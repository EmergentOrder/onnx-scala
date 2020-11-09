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
import org.emergentorder.compiletime._
import io.kjaer.compiletime._
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

  def runModel[T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorDenotation, S <: Shape](
      sess: OrtSession,
      input_tensor_values: Array[OnnxTensor],
      inputNames: List[String],
      outputNames: List[String]
  ): Tensor[T, Tuple3[Tt, Td, S]] = { 
    val inputs = (inputNames zip input_tensor_values).toMap.asJava

    //TODO: More outputs / handle via ONNXSequence / ONNXMap
      val output_tensor = sess.run(inputs)
      val firstOut = output_tensor.get(0).asInstanceOf[OnnxTensor]
      val shape = firstOut.getInfo.getShape.map(_.toInt) 
      //TODO: Denotations
      val result: Tensor[T, Tuple3[Tt, Td, S]] = Tensor(getArrayFromOnnxTensor(firstOut), "???", "???" ##: SSNil, Shape.fromSeq(shape)).asInstanceOf[Tensor[T,Tuple3[Tt, Td, S]]] //dangerous
      result
  }
    
// def cachedSess(bytes: Array[Byte]) = sessionCache.computeIfAbsent(java.util.Arrays.hashCode(bytes), _ => getSession(bytes))

  def callByteArrayOp[
      T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorDenotation, S <: Shape]( 
      opModel: Array[Byte],
      inputs: Tuple
  ): Tensor[T, Tuple3[Tt, Td, S]] = {
    val input_node_names = List("0", "1", "2", "3", "4", "5", "6", "7", "8")
    //TODO: more outputs
    val output_node_names = List("outName") 

    //TODO: don't mix up Options and Tensors here
    val inputTensors: Array[OnnxTensor] = inputs.toArray.map{elem =>
      elem match {
            case opt: Option[Tensor[T,  Tuple3[Tt, Td, S]]] =>
              opt match{
                case Some(x) => Some(getOnnxTensor(x.data, x.shape, env))
                case None => None
              }
            case tens: Tensor[T, Tuple3[Tt, Td, S]] => Some(getOnnxTensor(tens.data, tens.shape, env))
          }
      }.flatten

      val res: Tensor[T, Tuple3[Tt, Td, S]] = Using.resource(getSession(opModel)) { sess =>
        runModel[T, Tt, Td, S](
          sess, 
          inputTensors,
          input_node_names,
          output_node_names,
        )
      }
        res
  } 

  def callOp[
      T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorDenotation, S <: Shape](
      name: String,
      opName: String,
      inputs: Tuple,
      //    outName: String,
      attrs: Map[String, Any]): Tensor[T, Tuple3[Tt, Td, S]] = {
    //TODO: prevent passing input to opToONNXBytes
    
    val bytes = opToONNXBytes(name, opName, inputs, "outName", attrs)
    callByteArrayOp[T, Tt, Td, S](bytes,inputs)
  }

  override def close(): Unit = {
      env.close
//    super.close
  }
}
