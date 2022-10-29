package org.emergentorder.onnx.backends

import scala.concurrent.duration._
//import typings.onnxruntimeWeb.tensorMod._
//import typings.onnxruntimeWeb.tensorMod.Tensor.FloatType
//import typings.onnxruntimeWeb.tensorMod.Tensor.DataType
//import typings.onnxjs.libTensorMod.Tensor.DataTypeMap.DataTypeMapOps
import typings.onnxruntimeNode.mod.{InferenceSession => OrtSession}
import typings.onnxruntimeNode.mod.Tensor.{^ => OnnxTensor}
//import typings.onnxruntimeWeb.ort.InferenceSession.{^ => InferenceSess}
//import typings.onnxjs.onnxMod.Onnx
import scala.scalajs.js.typedarray
//import typings.onnxruntimeWeb.onnxImplMod._

//import scala.scalajs.js.Thenable.Implicits._
import scala.concurrent.Future
import scala.language.postfixOps
import scala.scalajs.js

import ORTTensorUtils._
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

trait ORTWebOperatorBackend {


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
     ???
     //TODO
     //
     //
     /*
     // TODO: prevent passing input to opToONNXBytes

      val modelProto = opToModelProto(opName, inputs, attrs)

      val result: Tensor[T, Tuple3[Tt, Td, S]] = callByteArrayOp(modelProto.toByteArray, inputs)
      result
      */
   }

   def runModel[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       sess: scala.scalajs.js.Promise[
         typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession
       ],
       input_tensor_values: Array[OnnxTensor[T]],
       inputNames: List[String],
       outputNames: List[String]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Future[Tensor[T, Tuple3[Tt, Td, S]]] = {

      // Limited to 1 input right now
      val feeds   = js.Dictionary("data" -> input_tensor_values.head)
      val sessFut = sess.toFuture
      val output_tensors: Future[typings.onnxruntimeCommon.tensorMod.Tensor] =
         sessFut
            .map { realSess =>
               val res = realSess.run(
                 feeds.asInstanceOf[
                   typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession.FeedsType
                 ]
               )
               res.toFuture
            }
            .flatten
            .map { result =>
               result
                  .asInstanceOf[
                    typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession.OnnxValueMapType
                  ]
                  .get("squeezenet0_flatten0_reshape0")
                  .getOrElse(null)
            }

      output_tensors.map { output_tensor =>
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

   def getArrayFromOnnxTensor[T](value: typings.onnxruntimeCommon.tensorMod.Tensor): Array[T] = {
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
   implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
   def test() = {

      val session: scala.scalajs.js.Promise[
        typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession
      ] = OrtSession.create("squeezenet1.1.onnx")
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
        1 #: 1000 #: SNil
      ](session, inputs, List("data"), List("squeezenet0_flatten0_reshape0"))

      res.foreach(tens => tens.data.foreach(println))
      println(res)
      res.andThen(x => println(x))
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
