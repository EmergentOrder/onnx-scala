package org.emergentorder.onnx.backends

import scala.concurrent.duration._
//import typings.onnxruntimeWeb.tensorMod._
//import typings.onnxruntimeWeb.tensorMod.Tensor.FloatType
//import typings.onnxruntimeWeb.tensorMod.Tensor.DataType
//import typings.onnxjs.libTensorMod.Tensor.DataTypeMap.DataTypeMapOps
import org.emergentorder.onnx.onnxruntimeWeb.mod.{InferenceSession => OrtSession}
import org.emergentorder.onnx.onnxruntimeWeb.mod.Tensor.{^ => OnnxTensor}
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
import io.kjaer.compiletime._

//TODO: fix redundant computation due to cats-effect on the JS side
trait ORTWebOperatorBackend extends OpToONNXBytesConverter {

   def getSession(bytes: Array[Byte]) = {

      val bytesArrayBuffer = bytes.toTypedArray.buffer
      val session: IO[
        InferenceSession
      ] = IO.fromFuture(IO { OrtSession.create(bytesArrayBuffer, {
        val opts = InferenceSession.SessionOptions()
        opts.executionProviders = scala.scalajs.js.Array("wasm")
        opts
      }
      ).toFuture })
      session
   }

   // Idea: prepopulate models for ops with no params
   def callByteArrayOp[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       opModel: Array[Byte],
       inputs: Tuple,
       input_node_names: IO[List[String]]
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
      val output_node_names = input_node_names.map(x => { List(x.toString) })

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
                           Some(x.data.flatMap { y =>
                              x.shape.map { z =>
                                 getOnnxTensor(y, z)
                              }
                           })
                        case None => None
                     }
                  case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                     Some(tens.data.flatMap { x =>
                        tens.shape.map { y =>
                           getOnnxTensor(x, y)
                        }
                     })
               }
            }
            .toList
            .sequence
            .map(_.toArray)
      }

      val res: Tensor[T, Tuple3[Tt, Td, S]] = {
//        val resource = cats.effect.Resource.make(IO{getSession(opModel)})(sess => IO{sess.close})
         // resource.use( sess =>
         inputTensors.flatMap { x =>
            // input_node_names.flatMap{y =>
            cats.effect.Resource
               .make(IO(getSession(opModel)))(sess => IO {})
               .use(sess =>
                  runModel(
                    sess,
                    x,
                    input_node_names,
                    output_node_names
                  )
               )
            // }
         }

      }
      // res.flatMap(IO.println("Post run").as(_))
      res
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
      // TODO: prevent passing input to opToONNXBytes

//      println("ATTR " + attrs)
      val modelProto = opToModelProto(opName, inputs, attrs)

      val result: IO[Tensor[T, Tuple3[Tt, Td, S]]] =
         for {
            mp <- modelProto.flatMap(IO.println("OpName => " + opName).as(_))
         } yield {
//            println(mp)
            callByteArrayOp(
              mp.toByteArray,
              inputs,
              IO.pure {
                 mp.graph.map(_.input.map(_.name.getOrElse(""))).getOrElse(List[String]()).toList
              }
            )
         }

      result.flatten
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
       inputNames: IO[List[String]],
       outputNames: IO[List[String]]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {

      val feeds: IO[js.Dictionary[OnnxTensor[T]]] = inputNames.map(x => {
         val zipped = x.toArray zip input_tensor_values
         js.Dictionary(zipped.map(z => z._1 -> z._2): _*)
      })

      val output_tensors: IO[org.emergentorder.onnx.onnxruntimeCommon.tensorMod.Tensor] =
         IO.fromFuture {
            sess
               .flatMap { realSess =>
                  feeds.flatMap { realFeeds =>
                     val res = IO.eval(cats.Eval.later {
                        realSess
                           .run(
                             realFeeds.asInstanceOf[
                               org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession.FeedsType
                             ]
                           )
                           .toFuture
                     })
                     outputNames.flatMap { names =>
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
                     }
                  }
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
      ] = IO.fromFuture { IO { OrtSession.create("squeezenet1.0-12.onnx").toFuture } }
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
        IO.pure { List("data_0") },
        IO.pure { List("squeezenet0_flatten0_reshape0") }
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
