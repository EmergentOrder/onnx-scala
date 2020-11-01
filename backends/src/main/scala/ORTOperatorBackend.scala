package org.emergentorder.onnx.backends

import java.nio._
import scala.jdk.CollectionConverters._
import scala.reflect.ClassTag
import scala.language.implicitConversions
import scala.util.Using
import org.bytedeco.javacpp._
import ai.onnxruntime._
import ai.onnxruntime.TensorInfo.OnnxTensorType._
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.bytedeco.javacpp.BooleanPointer


trait ORTOperatorBackend
    extends OpToONNXBytesConverter
    with AutoCloseable {

  //Java map performs better
  //val sessionCache = new java.util.HashMap[Integer, OrtSession]

  val env = org.emergentorder.onnx.Tensors.env

  def getSession(bytes: Array[Byte]) = { 

//    val session_options = new OrtSession.SessionOptions()
//    session_options.addDnnl(true)
    env.createSession(bytes) //, session_options)
  }

  def runModel[T <: Supported](
      sess: OrtSession,
      input_tensor_values: Array[OnnxTensor],
      inputNames: List[String],
      outputNames: List[String] 
  ): Tensor[T] = { 
    val inputs = (inputNames zip input_tensor_values).toMap.asJava

    //TODO: More outputs / handle via ONNXSequence / ONNXMap
    val output_tensor = sess.run(inputs)
      val firstOut = output_tensor.get(0).asInstanceOf[OnnxTensor]
      val shape = firstOut.getInfo.getShape
      val result: Tensor[T] = new Tensor(Tensors.getArrayFromOnnxTensor(firstOut), shape.map(_.toInt))  
      result
  }
    
// def cachedSess(bytes: Array[Byte]) = sessionCache.computeIfAbsent(java.util.Arrays.hashCode(bytes), _ => getSession(bytes))

  def callByteArrayOp[
      T <: Supported
  ](
      opModel: Array[Byte],
      inputs: Tuple
  ): Tensor[T] = {
    val input_node_names = List("0", "1", "2", "3", "4", "5", "6", "7", "8")
    //TODO: more outputs
    val output_node_names = List("outName") 

    //TODO: don't mix up Options and Tensors here
    val inputTensors: Array[OnnxTensor] = inputs.toArray.map{elem =>
      elem match {
            case opt: Option[Tensor[_]] =>
              opt match{
                case Some(x) => Some(x.onnxTensor)
                case None => None
              }
            case tens: Tensor[_] => Some(tens.onnxTensor)
          }
      }.flatten

      val sess  = getSession(opModel)
        val res:Tensor[T] = runModel(
          sess, 
          inputTensors,
          input_node_names,
          output_node_names
        )
      sess.close
        res
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
    callByteArrayOp[T](bytes,inputs)
  }

  override def close(): Unit = {
      env.close
//    super.close
  }
}
