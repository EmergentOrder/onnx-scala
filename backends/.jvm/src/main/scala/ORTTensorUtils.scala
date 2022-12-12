package org.emergentorder.onnx.backends

import java.nio.*
import scala.util.Using
import compiletime.asMatchable
import org.emergentorder.onnx.Tensors.*

import com.jyuzawa.onnxruntime.Environment;
import com.jyuzawa.onnxruntime.NamedCollection;
import com.jyuzawa.onnxruntime.OnnxRuntime;
import com.jyuzawa.onnxruntime.OnnxValue;
import com.jyuzawa.onnxruntime.OnnxTensor;
import com.jyuzawa.onnxruntime.Session;
import com.jyuzawa.onnxruntime.Transaction;
//Doesn't work
//import com.jyuzawa.onnxruntime_extern.onnxruntime_c_api_h.*
import ai.onnxruntime.TensorInfo.OnnxTensorType.*

object ORTTensorUtils {

   def putArrayIntoOnnxTensor[T](arr: Array[T], shape: Array[Int], tens: OnnxTensor) = {
      arr(0).asMatchable match {
         case b: Byte    => getTensorByte(arr.asInstanceOf[Array[Byte]], shape, tens)
         case s: Short   => getTensorShort(arr.asInstanceOf[Array[Short]], shape, tens)
         case d: Double  => getTensorDouble(arr.asInstanceOf[Array[Double]], shape, tens)
         case f: Float   => getTensorFloat(arr.asInstanceOf[Array[Float]], shape, tens)
         case i: Int     => getTensorInt(arr.asInstanceOf[Array[Int]], shape, tens)
         case l: Long    => getTensorLong(arr.asInstanceOf[Array[Long]], shape, tens)
//         case b: Boolean => getTensorBoolean(arr.asInstanceOf[Array[Boolean]], shape, sess)
//         case st: String => getTensorString(arr.asInstanceOf[Array[String]], shape, sess)
      }
   }

   private def getTensorByte(
       arr: Array[Byte],
       shape: Array[Int],
       tens: OnnxTensor
   ): Unit = {
        tens.getByteBuffer().put(arr)
   }

   private def getTensorShort(
       arr: Array[Short],
       shape: Array[Int],
       tens: OnnxTensor
   ): Unit = {
        tens.getShortBuffer().put(arr)
   }

   private def getTensorDouble(
       arr: Array[Double],
       shape: Array[Int],
       tens: OnnxTensor 
   ): Unit = {
        tens.getDoubleBuffer().put(arr)
   }

   private def getTensorInt(arr: Array[Int], shape: Array[Int], tens: OnnxTensor): Unit = { 
        tens.getIntBuffer().put(arr)
   }

   private def getTensorLong(
       arr: Array[Long],
       shape: Array[Int],
       tens: OnnxTensor
   ): Unit = {
        tens.getLongBuffer().put(arr)
   }

   private def getTensorFloat(
       arr: Array[Float],
       shape: Array[Int],
       tens: OnnxTensor 
   ): Unit = {
        tens.getFloatBuffer().put(arr)
   }

   /*
   private def getTensorString(
       arr: Array[String],
       shape: Array[Int],
       env: Environment
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
       env: Environment
   ): OnnxTensor = {
      // working around: https://github.com/microsoft/onnxruntime/issues/7358
      if shape.size == 0 || (shape.size == 1 && shape(0) == 1) then {
         OnnxTensor.createTensor(env, arr)
      } else {

         val dims = shape.map(x => Dimension.newBuilder().setDimValue(x))
         val shapeProto = { 
           val prot = TensorShapeProto.newBuilder()
           dims.foreach(prot.addDim(_))
           prot
         }
         Tensor.newBuilder().setElemType(DataType.Boolean).setShape(shapeProto) 

      }
   }
*/
   def getArrayFromOnnxTensor[T](value: OnnxTensor): Array[T] = {
      val dtype = value.getInfo.getType.getNumber
      val arr = dtype match {
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT.value => {
            value.getFloatBuffer.array()
         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE.value => {
            value.getDoubleBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8.value => {
            value.getByteBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16.value => {
            value.getShortBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32.value => {
            value.getIntBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64.value => {
            value.getLongBuffer.array()

         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL.value => {

            value.getByteBuffer.array().map(x => if x == 1 then true else false)
         }
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING.value => ??? // TODO
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8.value  => ??? // TODO, Newly supported in ORT Java 1.9.x
         case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED.value | ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16.value |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32.value | ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64.value |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16.value | ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64.value |
             ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128.value | ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16.value =>
            ??? // Unsupported
      }
      arr.asInstanceOf[Array[T]]
   }
}
