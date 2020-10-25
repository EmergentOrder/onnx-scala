package org.emergentorder.onnx.backends

import java.nio._
import scala.jdk.CollectionConverters._
import scala.reflect.ClassTag
import scala.language.implicitConversions
import scala.language.existentials
//import org.bytedeco.javacpp._
//import org.bytedeco.onnxruntime._
//import org.bytedeco.onnxruntime.global.onnxruntime._
import ai.onnxruntime._
import ai.onnxruntime.TensorInfo.OnnxTensorType._
import org.emergentorder.onnx._
import org.bytedeco.javacpp.BooleanPointer
import org.emergentorder.onnx.Tensors._

object ORTOperatorBackend {  
val env = OrtEnvironment.getEnvironment()

}


trait ORTOperatorBackend
    extends OpToONNXBytesConverter
    with AutoCloseable {

  def getSession(bytes: Array[Byte]) = { 

//    val session_options: OrtSession.SessionOptions = (new OrtSession.SessionOptions())
//      session_options.addDnnl(true)
//    session_options.SetIntraOpNumThreads(1)
    //Using DNNL
//    OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options.asOrtSessionOptions(), 1)

    //Bug w/ session options? 
    ORTOperatorBackend.env.createSession(bytes) //, session_options)
  }

  def runModel(
      sess: OrtSession,
      input_tensor_values: Array[OnnxTensor],
      inputNames: List[String],
      outputNames: List[String] 
  ) = { 
    val inputs = (inputNames zip input_tensor_values).toMap
    val output_tensor = sess.run(inputs.asJava)

    //TODO: More outputs
    val firstOut = output_tensor.get(0)

    //val size: Long = firstOut.GetTensorTypeAndShapeInfo.GetElementCount
//    val shape: LongPointer = firstOut.GetTensorTypeAndShapeInfo.GetShape()
    val shape = firstOut.getInfo.asInstanceOf[TensorInfo].getShape
    val out = getTensorFromValue(firstOut.asInstanceOf[OnnxTensor], shape)
//    firstOut.close
    output_tensor.close
    out
  }


  def getTensorFromValue(value: OnnxTensor, shape: Array[Long]) = {
    val dtype = value.getInfo.asInstanceOf[TensorInfo].onnxType
//    val size = value.GetTensorTypeAndShapeInfo.GetElementCount
    //val shape = (0 until shapePointer.capacity.toInt).map(x => shapePointer.get(x).toInt).toArray
    val arr = dtype match {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT =>{      
        value.getFloatBuffer.array()

        //(0 until buff.capacity).map { x =>
        //  buff.get(x)
        //}.toArray    
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE =>{

        value.getDoubleBuffer.array()

//        (0 until buff.capacity).map { x =>
//          buff.get(x)
//        }.toArray
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
        (0 until booleanPoint.capacity().toInt).map { x =>
          booleanPoint.get(x)
        }.toArray
      }
    }
    TensorFactory.getTensor(arr, shape.map(_.toInt))
  }

  def getTensor[T: ClassTag](input: T): Option[OnnxTensor] = {
    input match { 
      case tensorOpt: Option[Tensor[_]] => {
        tensorOpt match {
          case Some(tens: Tensor[_]) => {
            val value: OnnxTensor = tens._1 match {
              case b: Array[Byte] => getTensorByte(tens)
              case s: Array[Short] => getTensorShort(tens)
              case d: Array[Double] => getTensorDouble(tens)
              case f: Array[Float] => getTensorFloat(tens)
              case i: Array[Int]   => getTensorInt(tens)
              case l: Array[Long]  => getTensorLong(tens)
              case b: Array[Boolean] => getTensorBoolean(tens)
            }
            Some(value)
          }
          case None =>  None//OnnxTensor.createTensor(ORTOperatorBackend.env, Array[Float]())
        } 
     } 
    case _ => None
    }
  }

  def getTensorByte(tens: Tensor[Byte]): OnnxTensor = {
    val buff = ByteBuffer.wrap(tens._1)
    OnnxTensor.createTensor(ORTOperatorBackend.env,buff,tens._2.map(_.toLong)) 
  }

  def getTensorShort(tens: Tensor[Short]): OnnxTensor = {
    val buff = ShortBuffer.wrap(tens._1)
    OnnxTensor.createTensor(ORTOperatorBackend.env,buff,tens._2.map(_.toLong)) 
  }

  def getTensorDouble(tens: Tensor[Double]): OnnxTensor = {
    val buff = DoubleBuffer.wrap(tens._1)
    OnnxTensor.createTensor(ORTOperatorBackend.env,buff,tens._2.map(_.toLong)) 
  }

  def getTensorInt(tens: Tensor[Int]): OnnxTensor = {
    val buff = IntBuffer.wrap(tens._1)
    OnnxTensor.createTensor(ORTOperatorBackend.env,buff,tens._2.map(_.toLong)) 
  }

  def getTensorLong(tens: Tensor[Long]): OnnxTensor = {
    val buff = LongBuffer.wrap(tens._1)
    OnnxTensor.createTensor(ORTOperatorBackend.env,buff,tens._2.map(_.toLong)) 
  }

  def getTensorFloat(tens: Tensor[Float]): OnnxTensor = {
    val buff = FloatBuffer.wrap(tens._1)
    OnnxTensor.createTensor(ORTOperatorBackend.env,buff,tens._2.map(_.toLong)) 
  }

  def getTensorBoolean(tens: Tensor[Boolean]): OnnxTensor = {
    val tensorIn = OrtUtil.reshape(tens._1, tens._2.map(_.toLong))
    OnnxTensor.createTensor(ORTOperatorBackend.env,tensorIn)  
  }

  def callByteArrayOp[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag,
      T9: ClassTag,
      T10: ClassTag,
      T11: ClassTag,
      T12: ClassTag,
      T13: ClassTag,
      T14: ClassTag,
      T15: ClassTag,
      T16: ClassTag,
      T17: ClassTag
  ](
      opModel: Array[Byte],
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8] 
  ): (T9) = {
    
    val sess = getSession(opModel)

    val input_node_names = List("A","B","C","D","E","F","G","H","I") 

    //TODO: more outputs
    val output_node_names = List("outName") 

    val inputArr = Array(
      inputs._1.asInstanceOf[Option[Tensor[_]]],
      inputs._2.asInstanceOf[Option[Tensor[_]]],
      inputs._3.asInstanceOf[Option[Tensor[_]]],
      inputs._4.asInstanceOf[Option[Tensor[_]]],
      inputs._5.asInstanceOf[Option[Tensor[_]]],
      inputs._6.asInstanceOf[Option[Tensor[_]]],
      inputs._7.asInstanceOf[Option[Tensor[_]]],
      inputs._8.asInstanceOf[Option[Tensor[_]]],
      inputs._9.asInstanceOf[Option[Tensor[_]]]
    )//.flatten

    val inputDimsAndValues: Array[OnnxTensor] =
      inputArr.map(x => getTensor(x)).flatten

    val output = runModel(
      sess,
      inputDimsAndValues,
      input_node_names,
      output_node_names
    )

    inputDimsAndValues.foreach(_.close) 
//    sess.close
    output.asInstanceOf[T9]
  }

    def callOp[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag,
      T9: ClassTag,
      T10: ClassTag,
      T11: ClassTag,
      T12: ClassTag,
      T13: ClassTag,
      T14: ClassTag,
      T15: ClassTag,
      T16: ClassTag,
      T17: ClassTag
  ](
      name: String,
      opName: String,
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
      //    outName: String,
      attrs: Map[String, Any]
  ): (T9) = {
    val bytes = opToONNXBytes(name, opName, inputs, "outName", attrs)
    callByteArrayOp[T, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17](
      bytes,
      inputs
    )
  }

  override def close(): Unit = {
//    super.close
  }
}
