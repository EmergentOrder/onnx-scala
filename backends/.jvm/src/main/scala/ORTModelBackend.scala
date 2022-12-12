package org.emergentorder.onnx.backends

import scala.language.implicitConversions
import scala.concurrent.*
import scala.concurrent.duration.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.*
import compiletime.asMatchable
//import ai.onnxruntime.*
import scala.jdk.CollectionConverters.*

import com.jyuzawa.onnxruntime.Environment;
import com.jyuzawa.onnxruntime.NamedCollection;
import com.jyuzawa.onnxruntime.OnnxRuntime;
import com.jyuzawa.onnxruntime.OnnxValue;
import com.jyuzawa.onnxruntime.Session;
import com.jyuzawa.onnxruntime.Transaction;
import com.jyuzawa.onnxruntime.OnnxTensor

import cats.effect.IO
import cats.implicits.*
import cats.effect.unsafe.implicits.global
import org.emergentorder.onnx.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.Tensors.Tensor.*
import org.emergentorder.compiletime.*
import io.kjaer.compiletime.*

import ORTTensorUtils.*

//TODO: Clean up, remove asInstaceOf, etc.
class ORTModelBackend(onnxBytes: Array[Byte])
    extends Model()
    with ORTOperatorBackend
    with AutoCloseable {

   def getInputAndOutputNodeNamesAndDims(sess: Session) = {
      val input_node_names = session.getInputs.getList.asScala.map(_.getName)

      val inputNodeDims =
         session.getInputs.getList.asScala.map(_.getTypeInfo.getTensorInfo.getShape)

      val output_node_names = session.getOutputs.getList.asScala.map(_.getName)

      (input_node_names.toList, inputNodeDims.toArray, output_node_names.toList)
   }

   val session = getSession(onnxBytes)

   val allNodeNamesAndDims = getInputAndOutputNodeNamesAndDims(session)

   override def fullModel[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       inputs: Tuple
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {

      val size = inputs.size
      
      val output = 
            runModel[T, Tt, Td, S](
              session,
              inputs,
              allNodeNamesAndDims._1,
              allNodeNamesAndDims._3
            )
         
      output
   }

   override def close(): Unit = {}
}
