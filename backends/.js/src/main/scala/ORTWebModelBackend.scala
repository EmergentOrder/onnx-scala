package org.emergentorder.onnx.backends

import scala.language.implicitConversions
import scala.concurrent.Future
import scala.jdk.CollectionConverters._
import scala.scalajs.js.Array
import scalajs.js.JSConverters._

//import typings.onnxruntimeNode.mod.{InferenceSession => OrtSession}
import typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession
import typings.onnxruntimeNode.mod.Tensor.{^ => OnnxTensor}
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

import ORTTensorUtils._

//TODO: Clean up, remove asInstaceOf, etc.
class ORTWebModelBackend(session: InferenceSession)
    extends Model()
    with ORTWebOperatorBackend {

   def getInputAndOutputNodeNamesAndDims(sess: InferenceSession) = {
      val input_node_names = sess.inputNames

      val output_node_names = sess.outputNames

      (input_node_names.toList, None, output_node_names.toList)
   }

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
   ): Future[Tensor[T, Tuple3[Tt, Td, S]]] = {

      val size = inputs.size
      @annotation.nowarn
      val inputTensors = (0 until size).map { i =>
         val tup = inputs.drop(i).take(1)
         tup match { // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
            case t: Tuple1[_] =>
               t(0) match {
                  case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                     getOnnxTensor(tens.data, tens.shape).asInstanceOf[OnnxTensor[T]]
               }
         }
      }.toArray

      val output = runModel[T, Tt, Td, S](
        Future{session}.toJSPromise,
        inputTensors,
        allNodeNamesAndDims._1,
        allNodeNamesAndDims._3
      )

      output
   }

}
