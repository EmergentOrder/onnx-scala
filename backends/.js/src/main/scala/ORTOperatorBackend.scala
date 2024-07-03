package org.emergentorder.onnx.backends

import scala.concurrent.duration._
//import typings.onnxruntimeWeb.tensorMod
import org.emergentorder.onnx.onnxruntimeCommon.tensorMod
//import typings.onnxruntimeWeb.tensorMod.Tensor.DataType
//import typings.onnxjs.libTensorMod.Tensor.DataTypeMap.DataTypeMapOps
import org.emergentorder.onnx.onnxruntimeNode.mod.binding.{InferenceSession => OrtSession}
import org.emergentorder.onnx.onnxruntimeCommon.mod.Tensor.{^ => OnnxTensor}
//import typings.onnxruntimeWeb.ort.InferenceSession.{^ => InferenceSess}
//import typings.onnxjs.onnxMod.Onnx
import scala.scalajs.js.typedarray
//import typings.onnxruntimeWeb.onnxImplMod._

//import scala.scalajs.js.Thenable.Implicits._
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import scala.language.postfixOps
import scala.scalajs.js
import scalajs.js.JSConverters._
import scala.scalajs.js.typedarray._

import cats.implicits._
import cats.effect.{IO}
import ORTTensorUtils._
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.compiletime._
import onnxruntimeCommon.inferenceSessionMod.InferenceSession
import org.emergentorder.io.kjaer.compiletime._

//TODO: fix redundant computation due to cats-effect on the JS side
//Still happening, though partially fixed by changes in core
trait ORTOperatorBackend extends OpToONNXBytesConverter {

   def getSession(bytes: Array[Byte]) = {

      val bytesArrayBuffer = bytes.toTypedArray.buffer
      val session: IO[
        InferenceSession
      ] = IO.fromFuture(IO {
         val infSess = new OrtSession()
         infSess.loadModel(
           bytesArrayBuffer,
           0,
           bytesArrayBuffer.byteLength, {
              val opts = InferenceSession.SessionOptions()
              opts.executionProviders = scala.scalajs.js.Array("cpu")
              opts
           }
         )
         Future(
           infSess.asInstanceOf[
             org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
           ]
         )
      })
      session
   }

   // Idea: prepopulate models for ops with no params
   def callByteArrayOp[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       inputs: Tuple,
       input_node_names: List[String],
       opName: String,
       attrs: Map[String, Any]
   )(using
       s: ShapeOf[S],
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      /*
     val input_node_names = inputs.toArray.zipWithIndex.map { (e, i) =>
         val incr: String = if inputs.toArray.distinct.size == inputs.size then "" else i.toString
         val tensE = e.asInstanceOf[Tensor[T, Tuple3[Tt, Td, S]]]
         tensE.map{x =>
           val t = ((x.toString + incr).hashCode).toString
           println("ANESMMMS " + t + " " + i)
           t
         }
      }.toList.sequence
       */

      // TODO: more outputs
      val output_node_names = List(input_node_names.toString)

      // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
      // TODO: don't mix up Options and Tensors here
      @annotation.nowarn
      val inputTensors: IO[Array[OnnxTensor[T]]] = {

         inputs.toArray
            .flatMap { elem =>
               elem match {
                  case opt: Option[Tensor[T, Tuple3[Tt, Td, S]]] =>
                     opt match {
                        case Some(x) =>
                           Some(x.map { y =>
                              getOnnxTensor(y._1, y._2._3.toSeq.toArray)
                           })
                        case None => None
                     }
                  case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                     Some(tens.map { x =>
                        getOnnxTensor(x._1, x._2._3.toSeq.toArray)
                     })
               }
            }
            .toList
            .sequence
            .map(_.toArray)
      }

      def res(
          opModelBytes: Array[Byte],
          inputTensorss: IO[Array[OnnxTensor[T]]]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         cats.effect.Resource
            .make(inputTensorss)(inTens => IO {})
            .use(x =>
               cats.effect.Resource
                  .make(IO.blocking(getSession(opModelBytes)))(sess => IO {})
                  .use(sess =>
                     runModel(
                       sess,
                       x,
                       input_node_names,
                       output_node_names
                     )
                  )
               // }
            )

      }

      val finalRes = for {
         tens <- inputTensors.memoize
         t    <- tens
      } yield res(
        opToModelProto(
          opName,
          (t.map(x =>
             x.asInstanceOf[tensorMod.Tensor].`type`.valueOf.toString match {
                // Can't access the enum int values here
                // But it's fine, doesn't match the ONNX spec anyway
                case "int8"    => 3
                case "int16"   => 5
                case "float64" => 11
                case "float32" => 1
                case "int32"   => 6
                case "int64"   => 7
                case "bool"    => 9
                case y         => y.toInt
             }
          )
             zip t.map(_.dims.map(_.toInt).toArray)),
          attrs
        ).toByteArray,
        tens
      )

      finalRes.flatten
      // res
   }

   def callOp[T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](
       name: String,
       opName: String,
       inputs: Tuple,
       //    outName: String,
       attrs: Map[String, Any]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      val inputNodeNames = (0 until inputs.size).toList.map(_.toString)
      val result: Tensor[T, Tuple3[Tt, Td, S]] =
         callByteArrayOp(
           inputs,
           inputNodeNames,
           opName,
           attrs
         )
      // unsafeRunSync is not avaible in the JS context
      // so behavior on the JS side remains lazy
      // and thus inefficient in case user code refers to Tensors more than once
      result // .flatMap(x => IO.println("opName = " + opName).as(x))
   }

   def runModel[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       sess: IO[
         org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
       ],
       input_tensor_values: Array[OnnxTensor[T]],
       inputNames: List[String],
       outputNames: List[String]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {

      val feeds: js.Dictionary[OnnxTensor[T]] = {
         val zipped = inputNames.toArray zip input_tensor_values
         js.Dictionary(zipped.map(z => z._1 -> z._2)*)
      }

      val output_tensors: IO[org.emergentorder.onnx.onnxruntimeCommon.tensorMod.Tensor] =
         IO.fromFuture {
            sess
               .flatMap { realSess =>
//                  feeds.flatMap { realFeeds =>
                  val res = IO.eval(cats.Eval.later {
                     realSess
                        .run(
                          feeds.asInstanceOf[
                            org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession.FeedsType
                          ]
                        )
                        .toFuture
                  })
//                     outputNames.flatMap { names =>
                  res.map { result =>
                     result.map { rr =>
//                    println(realSess.outputNames.toList)
                        rr
//                      .asInstanceOf[
//                        org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession.OnnxValueMapType
//                      ]
                           .get(realSess.outputNames.toList(0))
                           .get
                     }
                  }
               // }
               //                }
               }
         }

      output_tensors.flatMap { output_tensor =>
         {
            val firstOut                      = output_tensor
            val shape                         = firstOut.dims
            val shapeFromType: S              = s.value
            val tensorTypeDenotationFromType  = tt.value
            val tensorShapeDenotationFromType = td.value
            require(shape sameElements shapeFromType.toSeq)
            // TODO: Denotations
            val arr: Array[T] = getArrayFromOnnxTensor(firstOut)

            val result = Tensor(
              arr,
              tensorTypeDenotationFromType,
              tensorShapeDenotationFromType,
              shapeFromType
            )

            result
         }
      }
   }

   val ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED  = 0
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT      = 1
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8      = 2
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8       = 3
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16     = 4
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16      = 5
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32      = 6
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64      = 7
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING     = 8
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL       = 9
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16    = 10
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE     = 11
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32     = 12
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64     = 13
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64  = 14
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15
   val ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16   = 16

   def getArrayFromOnnxTensor[T](
       value: org.emergentorder.onnx.onnxruntimeCommon.tensorMod.Tensor
   ): Array[T] = {
      val data = value.data
      val arr = data match {

         case a: scala.scalajs.js.typedarray.Float32Array => {
            scala.scalajs.js.typedarray.float32Array2FloatArray(a)
         }
         case b: scala.scalajs.js.typedarray.Float64Array => {
            scala.scalajs.js.typedarray.float64Array2DoubleArray(b)
         }
         case c: scala.scalajs.js.typedarray.Int8Array => {
            scala.scalajs.js.typedarray.int8Array2ByteArray(c)
         }
         case d: scala.scalajs.js.typedarray.Int16Array => {
            scala.scalajs.js.typedarray.int16Array2ShortArray(d)
         }
         case e: scala.scalajs.js.typedarray.Int32Array => {
            scala.scalajs.js.typedarray.int32Array2IntArray(e)
         }
         case f: scala.scalajs.js.typedarray.Uint8Array => { // Conflating bool and uint8 here
            f.toArray.map(x => if x == 1 then true else false)
         }
         case g: scala.scalajs.js.typedarray.BigInt64Array => {
            g.toArray
         }
         case _ => ???
         /*
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
           ???
           //scala.scalajs.js.typedarray.int64Array2FloatArray(value.data)
         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
           ???
            //value.getByteBuffer.array().map(x => if x == 1 then true else false)
         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => ??? // TODO
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8  => ??? // TODO, Newly supported in ORT Java 1.9.x
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED | ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 | ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 | ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 | ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 =>
            ??? // Unsupported
          */
      }

      arr.asInstanceOf[Array[T]]
   }

   //  extends OpToONNXBytesConverter
   // with AutoCloseable {

   def test() = {

      val session: IO[
        org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
      ] = IO.fromFuture {
         IO {
            val infSess = new OrtSession()
            infSess.loadModel("squeezenet1.0-12.onnx", InferenceSession.SessionOptions())
            Future(
              infSess.asInstanceOf[
                org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
              ]
            )
         }
      }
//      val dataTypes = new FloatType {}

      val dataType = "float32"
      val dims     = scala.scalajs.js.Array(1.0, 3.0, 224.0, 224.0)

      val rawData = typedarray.floatArray2Float32Array((0 until 150528).map(_ => 42.0f).toArray)
      // Should be a float tensor, not a string tensor..
      val tensor: OnnxTensor[Float] =
         (new OnnxTensor(rawData, dims)).asInstanceOf[OnnxTensor[Float]]

      // println(tensor.data)

      // r.Tensor(dims, dataType.asInstanceOf[typings.onnxruntimeWeb.tensorMod.Tensor.DataType])
      // data.set(scala.scalajs.js.Array[Double](0.0),rawData)

      // println(tensor.data)

      val inputs = Array(tensor)

      val res = runModel[
        Float,
        "ImageNetClassification",
        "Batch" ##: "Class" ##: TSNil,
        1 #: 1000 #: 1 #: 1 #: SNil
      ](
        session,
        inputs,
        List("data_0"),
        List("squeezenet0_flatten0_reshape0")
      )

      // res.foreach(tens => tens.data.foreach(println))
      // println(res)
      // res.andThen(x => println(x))
//      res.foreach(tens => println(tens.shape))

      /*
      res.foreach { result =>
            println(
              "RESULT :" + result
                 .asInstanceOf[
                   typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession.OnnxValueMapType
                 ]
                 .get("squeezenet0_flatten0_reshape0")
                 .getOrElse(null)
                 .data
            )
         }
       */
      // TODO: FIX README, to clarify that squeezenet output unnormalized scores, not probabilities

      // Getting error:
      // worker.js onmessage() captured an uncaught exception: ReferenceError: ortWasmThreaded is not defined

      //
      //
      // println("before run")
      // val res = sess.run(scala.scalajs.js.Array(inputs: _*))
      // println("after run")
      // res
   }
}
