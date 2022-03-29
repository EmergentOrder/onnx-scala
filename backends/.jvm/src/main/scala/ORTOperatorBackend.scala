package org.emergentorder.onnx.backends

import java.nio._
import scala.jdk.CollectionConverters._
import scala.language.implicitConversions
import scala.util.Using
import ai.onnxruntime._
import ai.onnxruntime.TensorInfo.OnnxTensorType
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._
import onnx.onnx._

import ORTTensorUtils._

trait ORTOperatorBackend extends OpToONNXBytesConverter with AutoCloseable {

   val env = OrtEnvironment.getEnvironment()

   val coreCount = java.lang.Runtime.getRuntime().availableProcessors()
   def getSession(bytes: Array[Byte]) = {
      // Can now set symbolic dimension values, but only at session creation time
      val session_options = new OrtSession.SessionOptions()
      session_options.setIntraOpNumThreads(coreCount)
//    session_options.addCUDA()
//    session_options.addDnnl(true)
      env.createSession(bytes, session_options)
   }

   def runModel[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       sess: OrtSession,
       input_tensor_values: Array[OnnxTensor],
       inputNames: List[String],
       outputNames: List[String]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      val inputs = (inputNames zip input_tensor_values).toMap.asJava

      // TODO: More outputs / handle via ONNXSequence / ONNXMap
      val output_tensor                 = sess.run(inputs)
      val firstOut                      = output_tensor.get(0).asInstanceOf[OnnxTensor]
      val shape                         = firstOut.getInfo.getShape.map(_.toInt)
      val shapeFromType: S              = s.value
      val tensorTypeDenotationFromType  = tt.value
      val tensorShapeDenotationFromType = td.value
      require(shape sameElements shapeFromType.toSeq)
      // TODO: Denotations
      val result: Tensor[T, Tuple3[Tt, Td, S]] = Tensor(
        getArrayFromOnnxTensor(firstOut),
        tensorTypeDenotationFromType,
        tensorShapeDenotationFromType,
        shapeFromType
      )
      result
   }

   def callByteArrayOp[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       opModel: Array[Byte],
       inputs: Tuple
   )(using
       s: ShapeOf[S],
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      val input_node_names = inputs.toArray.zipWithIndex.map { (e, i) =>
         val incr: String = if inputs.toArray.distinct.size == inputs.size then "" else i.toString
         ((e.toString + incr).hashCode).toString
      }.toList

      // TODO: more outputs
      val output_node_names = List(input_node_names.toString)

      // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
      // TODO: don't mix up Options and Tensors here
      @annotation.nowarn
      val inputTensors: Array[OnnxTensor] = inputs.toArray.map { elem =>
         elem match {
            case opt: Option[Tensor[T, Tuple3[Tt, Td, S]]] =>
               opt match {
                  case Some(x) => Some(getOnnxTensor(x.data, x.shape, env))
                  case None    => None
               }
            case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
               Some(getOnnxTensor(tens.data, tens.shape, env))
         }
      }.flatten

      val res: Tensor[T, Tuple3[Tt, Td, S]] = Using.resource(getSession(opModel)) { sess =>
         runModel(
           sess,
           inputTensors,
           input_node_names,
           output_node_names
         )
      }
      res
   }

   def callOp[T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](
       name: String,
       opName: String,
       inputs: Tuple,
       //    outName: String,
       attrs: Map[String, Any]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      // TODO: prevent passing input to opToONNXBytes

      val modelProto = opToModelProto(opName, inputs, attrs)

      val result: Tensor[T, Tuple3[Tt, Td, S]] = callByteArrayOp(modelProto.toByteArray, inputs)
      result
   }

   def modelToPersist(mod: ModelProto, outName: String) = {
      val outNode      = mod.getGraph.node(0).clearOutput.withOutput(Seq(outName))
      val outInfoProto = mod.getGraph.output(0).clearName.withName(outName)
      val graphToPersist =
         mod.getGraph.clearNode.withNode(Seq(outNode)).clearOutput.withOutput(Seq(outInfoProto))
      mod.clearGraph.withGraph(graphToPersist)
   }

   override def close(): Unit = {
      env.close
   }
}
