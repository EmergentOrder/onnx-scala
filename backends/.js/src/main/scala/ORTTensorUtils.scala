package org.emergentorder.onnx.backends

//import scala.scalajs.js.Array
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.onnxruntimeNode.mod.Tensor.{^ => OnnxTensor}
import scalajs.js.JSConverters._
import scala.scalajs.js.typedarray

object ORTTensorUtils {

   // CAUTION: not even Scala.js can fix JS strangeness around numbers
   // They may end up changing types based on the value, i.e. 1.0 becomes an Int, while 1.1 remains a Float
   //
   def getOnnxTensor[T](arr: scala.Array[T], shape: scala.Array[Int]): OnnxTensor[T] = {
      arr match {
         case b: Array[Byte] =>
            getTensorByte(arr.asInstanceOf[Array[Byte]], shape).asInstanceOf[OnnxTensor[T]]
         case s: Array[Short] =>
            getTensorShort(arr.asInstanceOf[Array[Short]], shape).asInstanceOf[OnnxTensor[T]]
         case d: Array[Double] =>
            getTensorDouble(arr.asInstanceOf[Array[Double]], shape).asInstanceOf[OnnxTensor[T]]
         case f: Array[Float] =>
            getTensorFloat(arr.asInstanceOf[Array[Float]], shape).asInstanceOf[OnnxTensor[T]]
         case i: Array[Int] =>
            getTensorInt(arr.asInstanceOf[Array[Int]], shape).asInstanceOf[OnnxTensor[T]]
         case l: Array[Long] =>
            getTensorLong(arr.asInstanceOf[Array[Long]], shape).asInstanceOf[OnnxTensor[T]]
         case b: Array[Boolean] =>
            getTensorBoolean(arr.asInstanceOf[Array[Boolean]], shape).asInstanceOf[OnnxTensor[T]]
         case st: Array[String] =>
            getTensorString(arr.asInstanceOf[Array[String]], shape).asInstanceOf[OnnxTensor[T]]
         case _ =>
            getTensorLong(arr.map(x => x.toString.toLong).toArray, shape)
               .asInstanceOf[OnnxTensor[T]]
      }
   }

   private def getTensorByte(
       arr: Array[Byte],
       shape: Array[Int]
   ): OnnxTensor[Byte] = {
      (new OnnxTensor(typedarray.byteArray2Int8Array(arr), shape.map(_.toDouble).toJSArray))
         .asInstanceOf[OnnxTensor[Byte]]
   }

   private def getTensorShort(
       arr: Array[Short],
       shape: Array[Int]
   ): OnnxTensor[Short] = {
      (new OnnxTensor(typedarray.shortArray2Int16Array(arr), shape.map(_.toDouble).toJSArray))
         .asInstanceOf[OnnxTensor[Short]]
   }

   private def getTensorDouble(
       arr: Array[Double],
       shape: Array[Int]
   ): OnnxTensor[Double] = {
      (new OnnxTensor(typedarray.doubleArray2Float64Array(arr), shape.map(_.toDouble).toJSArray))
         .asInstanceOf[OnnxTensor[Double]]
   }

   private def getTensorInt(arr: Array[Int], shape: Array[Int]): OnnxTensor[Int] = {
      (new OnnxTensor(typedarray.intArray2Int32Array(arr), shape.map(_.toDouble).toJSArray))
         .asInstanceOf[OnnxTensor[Int]]
   }

   private def getTensorLong(
       arr: Array[Long],
       shape: Array[Int]
   ): OnnxTensor[Long] = {
      (new OnnxTensor("int64", arr.toJSArray, shape.map(_.toDouble).toJSArray))
         .asInstanceOf[OnnxTensor[Long]]
   }

   private def getTensorFloat(
       arr: Array[Float],
       shape: Array[Int]
   ): OnnxTensor[Float] = {
      (new OnnxTensor(typedarray.floatArray2Float32Array(arr), shape.map(_.toDouble).toJSArray))
         .asInstanceOf[OnnxTensor[Float]]
   }

   private def getTensorString(
       arr: Array[String],
       shape: Array[Int]
   ): OnnxTensor[String] = {
      (new OnnxTensor("string", arr.toJSArray, shape.map(_.toDouble).toJSArray))
         .asInstanceOf[OnnxTensor[String]]
   }

   private def getTensorBoolean(
       arr: Array[Boolean],
       shape: Array[Int]
   ): OnnxTensor[Boolean] = {
      (new OnnxTensor("bool", arr.toJSArray, shape.map(_.toDouble).toJSArray))
         .asInstanceOf[OnnxTensor[Boolean]]
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

//   def getArrayFromOnnxTensor[T](value: OnnxTensor[T]): Array[T] = {
//      value.data.asInstanceOf[Array[T]]
//   }
}
