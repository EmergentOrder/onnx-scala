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

    //Using DNNL - TODO
//    OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options.asOrtSessionOptions(), 1)
    //Bug in ORT w/ sessions options (link error)
    env.createSession(bytes)
  }

  def runModel(
      sess: OrtSession,
      input_tensor_values: Array[OnnxTensor],
      inputNames: List[String],
      outputNames: List[String] 
  ) = { 
    val inputs = (inputNames zip input_tensor_values).toMap
    val output_tensor = sess.run(inputs.asJava, new OrtSession.RunOptions())

    //TODO: More outputs
    val firstOut = output_tensor.get(0)

    val shape = firstOut.getInfo.asInstanceOf[TensorInfo].getShape
    val out = getTensorFromValue(firstOut.asInstanceOf[OnnxTensor], shape)
//    firstOut.close
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
  
  def getTensor[T:ClassTag](input: T): OnnxTensor = {

    input match{
      case tensorOpt: Option[_] => {
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
            value
          }
          case None =>  OnnxTensor.createTensor(env, Array())
        }
      }

      case tensorOpt: Tensor[_] => {
        tensorOpt match {
          case tens => {
            val value: OnnxTensor = tens._1 match {
              case b: Array[Byte] => getTensorByte(tens)
              case s: Array[Short] => getTensorShort(tens)
              case d: Array[Double] => getTensorDouble(tens)
              case f: Array[Float] => getTensorFloat(tens)
              case i: Array[Int]   => getTensorInt(tens)
              case l: Array[Long]  => getTensorLong(tens)
              case b: Array[Boolean] => getTensorBoolean(tens)
            }
            value
          }
        }
      }

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
      T: ClassTag
  ](
      opModel: Array[Byte],
      inputs: Option[NonEmptyTuple]
  ): T = {

    val sess = getSession(opModel) 

    val out = inputs match{
      case Some(x) => {

    val input_node_names = List("0", "1", "2", "3", "4", "5", "6", "7", "8")
   
    //TODO: more outputs
    val output_node_names = List("outName") 

    val inputDimsAndValues: Array[OnnxTensor] = (0 until x.size).map{i => 
     
      val tens = x(i)

      tens match {
        case None => None
        case _ =>
          Some(getTensor(tens))
      }
    }.toArray.flatten
   
  //  val filteredInputNodeNames = input_node_names.take(x.size)
    val output = runModel(
      sess, 
      inputDimsAndValues,
      input_node_names,
      output_node_names
    )
    inputDimsAndValues.foreach(_.close) 
    output.asInstanceOf[T]
      } 
      case None => TensorFactory.getTensor(Array(), Array[Int]()).asInstanceOf[T]
    
    }
    sess.close
    
    out
  }

  def callOp[
      T: ClassTag](
      name: String,
      opName: String,
      inputs: Option[NonEmptyTuple],
      //    outName: String,
      attrs: Map[String, Any]
  ): T = {
    val bytes = opToONNXBytes(name, opName, inputs, "outName", attrs)
    callByteArrayOp[T](
      bytes,
      inputs
    )
  }

  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
