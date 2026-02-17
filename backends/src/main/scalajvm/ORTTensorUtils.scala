package org.emergentorder.onnx.backends

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtUtil
import ai.onnxruntime.TensorInfo.OnnxTensorType.*

import scala.jdk.OptionConverters.RichOptional
import java.nio.*

import compiletime.asMatchable
//import scala.jdk.OptionConverters._

object ORTTensorUtils {

   def getOnnxTensor[T](arr: Array[T], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
      arr(0).asMatchable match {
         case _: Byte    => getTensorByte(arr.asInstanceOf[Array[Byte]], shape, env)
         case _: Short   => getTensorShort(arr.asInstanceOf[Array[Short]], shape, env)
         case _: Double  => getTensorDouble(arr.asInstanceOf[Array[Double]], shape, env)
         case _: Float   => getTensorFloat(arr.asInstanceOf[Array[Float]], shape, env)
         case _: Int     => getTensorInt(arr.asInstanceOf[Array[Int]], shape, env)
         case _: Long    => getTensorLong(arr.asInstanceOf[Array[Long]], shape, env)
         case _: Boolean => getTensorBoolean(arr.asInstanceOf[Array[Boolean]], shape, env)
         case _: String  => getTensorString(arr.asInstanceOf[Array[String]], shape, env)
      }
   }

   private def getTensorByte(
       arr: Array[Byte],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buffDirect: ByteBuffer = ByteBuffer.allocateDirect(arr.size).order(ByteOrder.nativeOrder())
      buffDirect.put(arr)
      buffDirect.rewind()
      OnnxTensor.createTensor(env, buffDirect, shape.map(_.toLong))
   }

   private def getTensorShort(
       arr: Array[Short],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buffDirect: ByteBuffer = ByteBuffer.allocateDirect(arr.size * 2).order(ByteOrder.nativeOrder())
      val buffSDirect: ShortBuffer = buffDirect.asShortBuffer
      buffSDirect.put(arr)
      buffSDirect.rewind()
      OnnxTensor.createTensor(env, buffSDirect, shape.map(_.toLong))
   }

   private def getTensorDouble(
       arr: Array[Double],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buffDirect: ByteBuffer = ByteBuffer.allocateDirect(arr.size * 8).order(ByteOrder.nativeOrder())
      val buffDDirect: DoubleBuffer = buffDirect.asDoubleBuffer
      buffDDirect.put(arr)
      buffDDirect.rewind()
      OnnxTensor.createTensor(env, buffDDirect, shape.map(_.toLong))
   }

   private def getTensorInt(arr: Array[Int], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
      val buffDirect: ByteBuffer = ByteBuffer.allocateDirect(arr.size * 4).order(ByteOrder.nativeOrder())
      val buffIDirect: IntBuffer = buffDirect.asIntBuffer
      buffIDirect.put(arr)
      buffIDirect.rewind()
      OnnxTensor.createTensor(env, buffIDirect, shape.map(_.toLong))
   }

   private def getTensorLong(
       arr: Array[Long],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buffDirect: ByteBuffer = ByteBuffer.allocateDirect(arr.size * 8).order(ByteOrder.nativeOrder())
      val buffLDirect: LongBuffer = buffDirect.asLongBuffer
      buffLDirect.put(arr)
      buffLDirect.rewind()
      OnnxTensor.createTensor(env, buffLDirect, shape.map(_.toLong))
   }

   private def getTensorFloat(
       arr: Array[Float],
       shape: Array[Int],
       env: OrtEnvironment
   ): OnnxTensor = {
      val buffFDirect: java.nio.FloatBuffer = ByteBuffer.allocateDirect(arr.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer
      buffFDirect.put(arr)
      buffFDirect.rewind()
      if (buffFDirect.isDirect) 
        val tens = OnnxTensor.createTensor(env, buffFDirect, shape.map(_.toLong))
        if (!tens.ownsBuffer())//.getFloatBuffer.isDirect)
          tens
        else
          throw new Exception("GGG")
      else
        throw new Exception("FFFF")
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
      val arr   = dtype match {
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => { 
           val outBuf: java.nio.FloatBuffer = value.getBufferRef().toScala match {
                   case Some(x) => x match {
                     case fb: java.nio.FloatBuffer => fb
                     case _ => throw new Exception("missing")
                   }
                   case None => throw new Exception("missing")
             }
            //value.getBufferRef().toScala
            //val fb = value.getFloatBuffer
            //val outArr = Array.ofDim[Float](outBuf.remaining)
              //new Array[Float](outBuf.remaining)
            if (outBuf.isDirect)
              value.getFloatBuffer.array()
              //outBuf.get(outArr)
              //outBuf.asReadOnlyBuffer.array()
              //outArr
            else
              throw new Exception("Buff is not direct!!!!!!!")
            //value.getBufferRef()
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
             ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 | ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ =>
            ??? // Unsupported
      }
      value.close
      arr.asInstanceOf[Array[T]]
   }
}
