package org.emergentorder.onnx.backends

import java.nio._
import scala.jdk.CollectionConverters._
import scala.reflect.ClassTag
import scala.language.implicitConversions
import org.bytedeco.javacpp._
import ai.onnxruntime._
import ai.onnxruntime.TensorInfo.OnnxTensorType._
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.bytedeco.javacpp.BooleanPointer

trait ORTOperatorBackend
    extends OpToONNXBytesConverter
    with AutoCloseable {

  val env = OrtEnvironment.getEnvironment()

  def getSession(bytes: Array[Byte]) = { 

//    val session_options = new OrtSession.SessionOptions()

//    session_options.addDnnl(true)
    //Using DNNL - TODO
//    OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options.asOrtSessionOptions(), 1)
    //Bug in ORT w/ sessions options (link error)
    env.createSession(bytes) //, session_options)
  }

  def runModel(
      sess: OrtSession,
      input_tensor_values: Array[OnnxTensor],
      inputNames: List[String],
      outputNames: List[String] 
  ) = { 
    val inputs = (inputNames zip input_tensor_values).toMap.asJava
    val output_tensor = sess.run(inputs)

    //TODO: More outputs
    val firstOut: OnnxTensor = output_tensor.get(0).asInstanceOf[OnnxTensor]

    val shape = firstOut.getInfo.asInstanceOf[TensorInfo].getShape
    val out = getTensorFromValue(firstOut, shape)
    output_tensor.close
    out
  }

  def getTensorFromValue(value: OnnxTensor, shape: Array[Long]) = {
    val dtype = value.getInfo.asInstanceOf[TensorInfo].onnxType
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
        (0 until booleanPoint.capacity().toInt).map { x =>
          booleanPoint.get(x)
        }.toArray
      }
    }
    TensorFactory.getTensor(arr, shape.map(_.toInt))
  }
  
  def getTensor[T <: Supported](tens: Tensor[T]): OnnxTensor = {

    tens._1 match {
                  
      case b: Array[Byte] => getTensorByte(tens)
      case s: Array[Short] => getTensorShort(tens)
      case d: Array[Double] => getTensorDouble(tens)            
      case f: Array[Float] => getTensorFloat(tens)
      case i: Array[Int]   => getTensorInt(tens)
      case l: Array[Long]  => getTensorLong(tens)
      case b: Array[Boolean] => getTensorBoolean(tens)
                
    }
  }

  def getTensorByte(tens: Tensor[Byte]): OnnxTensor = {
    val buff = ByteBuffer.wrap(tens._1)
    OnnxTensor.createTensor(env,buff,tens._2.map(_.toLong))
  }

  def getTensorShort(tens: Tensor[Short]): OnnxTensor = {
    val buff = ShortBuffer.wrap(tens._1)
    OnnxTensor.createTensor(env,buff,tens._2.map(_.toLong))
  }

  def getTensorDouble(tens: Tensor[Double]): OnnxTensor = {      
    val buff = DoubleBuffer.wrap(tens._1)
    OnnxTensor.createTensor(env,buff,tens._2.map(_.toLong))
  }

  def getTensorInt(tens: Tensor[Int]): OnnxTensor = {
    val buff = IntBuffer.wrap(tens._1)
    OnnxTensor.createTensor(env,buff,tens._2.map(_.toLong))
  }

  def getTensorLong(tens: Tensor[Long]): OnnxTensor = {
    val buff = LongBuffer.wrap(tens._1)
    OnnxTensor.createTensor(env,buff,tens._2.map(_.toLong))
  }

  def getTensorFloat(tens: Tensor[Float]): OnnxTensor = {
    val buff = FloatBuffer.wrap(tens._1)
    OnnxTensor.createTensor(env,buff,tens._2.map(_.toLong))
  }

  def getTensorBoolean(tens: Tensor[Boolean]): OnnxTensor = {
    val tensorIn = OrtUtil.reshape(tens._1, tens._2.map(_.toLong))
    OnnxTensor.createTensor(env,tensorIn)
  }

  def callByteArrayOp[
      T <: Supported
  ](
      opModel: Array[Byte],
      inputs: Tuple
  ): Tensor[T] = {

    val sess = getSession(opModel)  

    val input_node_names = List("0", "1", "2", "3", "4", "5", "6", "7", "8")
   
    //TODO: more outputs
    val output_node_names = List("outName") 

    //TODO: don't mix up Options and Tensors here
    val inputDimsAndValues: Array[OnnxTensor] = inputs.toArray.map{elem =>
      elem match {
            case opt: Option[Tensor[_]] =>
              opt match{
                case Some(x) => Some(getTensor(x))
                case None => None
              }
            case tens: Tensor[_] => Some(getTensor(tens))
          }
      }.flatten
   
  //  val filteredInputNodeNames = input_node_names.take(x.size)
    val output = runModel(
      sess, 
      inputDimsAndValues,
      input_node_names,
      output_node_names
    )
    inputDimsAndValues.foreach(_.close) 
    val out = output.asInstanceOf[Tensor[T]]
     
    sess.close
    
    out
  }

  def callOp[
      T <: Supported](
      name: String,
      opName: String,
      inputs: Tuple,
      //    outName: String,
      attrs: Map[String, Any]
  ): Tensor[T] = {
    //TODO: prevent passing input to opToONNXBytes
    val bytes = opToONNXBytes(name, opName, inputs, "outName", attrs)
    callByteArrayOp[T](
      bytes,
      inputs
    )
  }

  override def close(): Unit = {
      env.close
//    super.close
  }
}
