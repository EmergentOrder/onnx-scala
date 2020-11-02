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
import ORTTensorUtils._

trait ORTOperatorBackend
    extends OpToONNXBytesConverter
    with AutoCloseable {

  //Java map performs better
  //val sessionCache = new java.util.HashMap[Integer, OrtSession]
  
  val env = OrtEnvironment.getEnvironment()

  def getSession(bytes: Array[Byte]) = { 

//    val session_options = new OrtSession.SessionOptions()
//    session_options.addDnnl(true)
    env.createSession(bytes) //, session_options)
  }

  def runModel[T <: Supported, Ax <: Axes](
      sess: OrtSession,
      input_tensor_values: Array[OnnxTensor],
      inputNames: List[String],
      outputNames: List[String] 
  ): Tensor[T, Ax] = { 
    val inputs = (inputNames zip input_tensor_values).toMap.asJava

    //TODO: More outputs / handle via ONNXSequence / ONNXMap
    val output_tensor = sess.run(inputs)
      val firstOut = output_tensor.get(0).asInstanceOf[OnnxTensor]
      val shape = firstOut.getInfo.getShape.map(_.toInt)

      val result: Tensor[T, Ax] = Tensor.create(getArrayFromOnnxTensor[T](firstOut), shape).asInstanceOf[Tensor[T, Ax]] //dangerous
      result
  }
    
// def cachedSess(bytes: Array[Byte]) = sessionCache.computeIfAbsent(java.util.Arrays.hashCode(bytes), _ => getSession(bytes))

  def callByteArrayOp[
      T <: Supported,
      Ax <: Axes
  ](
      opModel: Array[Byte],
      inputs: Tuple
  ): Tensor[T, Ax] = {
    val input_node_names = List("0", "1", "2", "3", "4", "5", "6", "7", "8")
    //TODO: more outputs
    val output_node_names = List("outName") 

    //TODO: don't mix up Options and Tensors here
    val inputTensors: Array[OnnxTensor] = inputs.toArray.map{elem =>
      elem match {
            case opt: Option[Tensor[?, ?]] =>
              opt match{
                case Some(x) => Some(getOnnxTensor(x._1, x._2, env))
                case None => None
              }
            case tens: Tensor[?, ?] => Some(getOnnxTensor(tens._1, tens._2, env))
          }
      }.flatten

      val sess  = getSession(opModel)
        val res:Tensor[T, Ax] = runModel(
          sess, 
          inputTensors,
          input_node_names,
          output_node_names
        )
      sess.close
        res
  } 

  def callOp[
      T <: Supported, Ax <: Axes](
      name: String,
      opName: String,
      inputs: Tuple,
      //    outName: String,
      attrs: Map[String, Any]
  ): Tensor[T, Ax] = {
    //TODO: prevent passing input to opToONNXBytes

    val bytes = opToONNXBytes(name, opName, inputs, "outName", attrs)
    callByteArrayOp[T, Ax](bytes,inputs)
  }

  override def close(): Unit = {
      env.close
//    super.close
  }
}
