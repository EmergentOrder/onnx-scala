package org.emergentorder.onnx

import java.nio._
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric

import org.bytedeco.javacpp.BooleanPointer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtUtil
import ai.onnxruntime.TensorInfo.OnnxTensorType._

object Tensors{

  val env = OrtEnvironment.getEnvironment()

  type Supported = Int | Long | Float | Double | Byte | Short | UByte | UShort | UInt | ULong | 
                   Boolean | String | Float16 | Complex[Float] | Complex[Double]

  //TODO: Use IArray ? 
  case class Tensor[T <: Supported](data: Array[T], shape: Array[Int]){
    lazy val _1: Array[T] = data 
    lazy val _2: Array[Int] = shape 
    //TODO: move this out to an implicit conversion in the backend
    lazy val onnxTensor = getOnnxTensor(data,shape)
    require(data.size == shape.foldLeft(1)(_ * _))      
  }
 
  type SparseTensor[T <: Supported] = Tensor[T]

  private def getOnnxTensor[T <: Supported](arr: Array[T], shape: Array[Int]): OnnxTensor = {
    arr match {
      case b: Array[Byte] => getTensorByte(arr, shape)
      case s: Array[Short] => getTensorShort(arr, shape)
      case d: Array[Double] => getTensorDouble(arr, shape)
      case f: Array[Float] => getTensorFloat(arr, shape)
      case i: Array[Int]   => getTensorInt(arr, shape)
      case l: Array[Long]  => getTensorLong(arr, shape)
      case b: Array[Boolean] => getTensorBoolean(arr, shape)
    }
  }

  private def getTensorByte(arr: Array[Byte], shape: Array[Int]): OnnxTensor = {
    val buff = ByteBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorShort(arr: Array[Short], shape: Array[Int]): OnnxTensor = {
    val buff = ShortBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorDouble(arr: Array[Double], shape: Array[Int]): OnnxTensor = {
    val buff = DoubleBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorInt(arr: Array[Int], shape: Array[Int]): OnnxTensor = {
    val buff = IntBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorLong(arr: Array[Long], shape: Array[Int]): OnnxTensor = {
    val buff = LongBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorFloat(arr: Array[Float], shape: Array[Int]): OnnxTensor = {
    val buff = FloatBuffer.wrap(arr)
    OnnxTensor.createTensor(env,buff,shape.map(_.toLong))
  }

  private def getTensorBoolean(arr: Array[Boolean], shape: Array[Int]): OnnxTensor = {
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
