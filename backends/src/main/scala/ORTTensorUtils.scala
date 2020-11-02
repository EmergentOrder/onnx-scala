package org.emergentorder.onnx.backends

import java.nio._
import org.emergentorder.onnx.Tensors._
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtUtil
import ai.onnxruntime.TensorInfo.OnnxTensorType._
import org.bytedeco.javacpp.BooleanPointer

object ORTTensorUtils{

    def getOnnxTensor[T <: Supported](arr: Array[T], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
    arr match {
      case b: Array[Byte] => getTensorByte(arr, shape, env)
      case s: Array[Short] => getTensorShort(arr, shape, env)
      case d: Array[Double] => getTensorDouble(arr, shape, env)
      case f: Array[Float] => getTensorFloat(arr, shape, env)
      case i: Array[Int]   => getTensorInt(arr, shape, env)
      case l: Array[Long]  => getTensorLong(arr, shape, env)
      case b: Array[Boolean] => getTensorBoolean(arr, shape, env)
    }
  }

  private def getTensorByte(arr: Array[Byte], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
    val buff = ByteBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorShort(arr: Array[Short], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
    val buff = ShortBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorDouble(arr: Array[Double], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
    val buff = DoubleBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorInt(arr: Array[Int], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
    val buff = IntBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorLong(arr: Array[Long], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
    val buff = LongBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorFloat(arr: Array[Float], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
    val buff = FloatBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorBoolean(arr: Array[Boolean], shape: Array[Int], env: OrtEnvironment): OnnxTensor = {
    val tensorIn = OrtUtil.reshape(arr, shape.map(_.toLong))
    OnnxTensor.createTensor(env,tensorIn)
  }

  def getArrayFromOnnxTensor[T <: Supported] (value: OnnxTensor): Array[T] = {
    val dtype = value.getInfo.onnxType
    val arr = dtype match {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT =>{
        value.getFloatBuffer.array()
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE =>{
        value.getDoubleBuffer.array()

      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 =>{
        value.getByteBuffer.array()

      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 =>{
        value.getShortBuffer.array()

      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 =>{
        value.getIntBuffer.array()

      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 =>{
        value.getLongBuffer.array()

      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL =>{
        val booleanPoint =
          new BooleanPointer(
            value.getByteBuffer
          ) //C++ bool size is not defined, could cause problems on some platforms
        (0l until (booleanPoint.capacity(): Long)).map { x =>
          booleanPoint.get(x)
        }.toArray
      }
    }
    value.close
    arr.asInstanceOf[Array[T]]
  }
}
