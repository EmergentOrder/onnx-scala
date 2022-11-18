package org.emergentorder.onnx.backends

import java.nio._
import org.emergentorder.onnx.Tensors._
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtUtil
import ai.onnxruntime.TensorInfo.OnnxTensorType._

object ORTTensorUtils {

   def getOnnxTensor[T](arr: Array[T], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
      arr(0) match {
         case b: Byte    => getTensorByte(arr.asInstanceOf[Array[Byte]], shape, env)
         case s: Short   => getTensorShort(arr.asInstanceOf[Array[Short]], shape, env)
         case d: Double  => getTensorDouble(arr.asInstanceOf[Array[Double]], shape, env)
         case f: Float   => getTensorFloat(arr.asInstanceOf[Array[Float]], shape, env)
         case i: Int     => getTensorInt(arr.asInstanceOf[Array[Int]], shape, env)
         case l: Long    => getTensorLong(arr.asInstanceOf[Array[Long]], shape, env)
         case b: Boolean => getTensorBoolean(arr.asInstanceOf[Array[Boolean]], shape, env)
         case st: String => getTensorString(arr.asInstanceOf[Array[String]], shape, env)
      }
   }

   private def getTensorByte(
       arr: Array[Byte],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buff = ByteBuffer.wrap(arr)
      OnnxTensor.createTensor(env, buff, shape.map(_.toLong))
   }

   private def getTensorShort(
       arr: Array[Short],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buff = ShortBuffer.wrap(arr)
      OnnxTensor.createTensor(env, buff, shape.map(_.toLong))
   }

   private def getTensorDouble(
       arr: Array[Double],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buff = DoubleBuffer.wrap(arr)
      OnnxTensor.createTensor(env, buff, shape.map(_.toLong))
   }

   private def getTensorInt(arr: Array[Int], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
      val buff = IntBuffer.wrap(arr)
      OnnxTensor.createTensor(env, buff, shape.map(_.toLong))
   }

   private def getTensorLong(
       arr: Array[Long],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buff = LongBuffer.wrap(arr)
      OnnxTensor.createTensor(env, buff, shape.map(_.toLong))
   }

   private def getTensorFloat(
       arr: Array[Float],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buff = FloatBuffer.wrap(arr)
      OnnxTensor.createTensor(env, buff, shape.map(_.toLong))
   }

   private def getTensorString(
       arr: Array[String],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      // working around: https://github.com/microsoft/onnxruntime/issues/7358
      if shape.size == 0 || (shape.size == 1 && shape(0) == 1) then {
         OnnxTensor.createTensor(env, arr)
      } else {
         val tensorIn = OrtUtil.reshape(arr, shape.map(_.toLong))
         OnnxTensor.createTensor(env, tensorIn)
      }
   }

   private def getTensorBoolean(
       arr: Array[Boolean],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      // working around: https://github.com/microsoft/onnxruntime/issues/7358
      if shape.size == 0 || (shape.size == 1 && shape(0) == 1) then {
         OnnxTensor.createTensor(env, arr)
      } else {

         val tensorIn = OrtUtil.reshape(arr, shape.map(_.toLong))
         OnnxTensor.createTensor(env, tensorIn)

      }
   }

   def getArrayFromOnnxTensor[T](value: OnnxTensor): Array[T] = {
      val dtype = value.getInfo.onnxType
      val arr = dtype match {
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
            value.getFloatBuffer.array()
         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
            value.getDoubleBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
            value.getByteBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
            value.getShortBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            value.getIntBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            value.getLongBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {

            value.getByteBuffer.array().map(x => if x == 1 then true else false)
         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => ??? // TODO
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8  => ??? // TODO, Newly supported in ORT Java 1.9.x
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED | ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 | ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 | ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 | ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 =>
            ??? // Unsupported
      }
      value.close()
      arr.asInstanceOf[Array[T]]
   }
}
