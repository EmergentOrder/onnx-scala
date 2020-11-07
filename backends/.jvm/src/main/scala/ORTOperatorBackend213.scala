package org.emergentorder.onnx.backends

import java.nio._
import scala.jdk.CollectionConverters._
import scala.reflect.ClassTag
import scala.language.implicitConversions
import scala.util.Using
import ai.onnxruntime._
import ai.onnxruntime.TensorInfo.OnnxTensorType._
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
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

  def runModel[T, Ax <: Axes](
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
      T,
      Ax <: Axes
  ](
      opModel: Array[Byte],
      inputs: Seq[_]
  ): Tensor[T, Ax] = {
    val input_node_names = List("0", "1", "2", "3", "4", "5", "6", "7", "8")
    //TODO: more outputs
    val output_node_names = List("outName") 

    //TODO: don't mix up Options and Tensors here
    val inputTensors: Array[OnnxTensor] = inputs.toArray.map{elem =>
      elem match {
            case opt: Option[Tensor[T, Ax]] =>
              opt match{
                case Some(x) => Some(getOnnxTensor(data(x), shape(x), env))
                case None => None
              }
            case tens: Tensor[T, Ax] => Some(getOnnxTensor(data(tens), shape(tens), env))
          }
      }.flatten

      val res: Tensor[T, Ax] = Using.resource(getSession(opModel)) { sess =>
        runModel(
          sess, 
          inputTensors,
          input_node_names,
          output_node_names
        )
      }
        res
  } 

  def callOp[
      T, Ax <: Axes](
      name: String,
      opName: String,
      inputs: Seq[_],
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
