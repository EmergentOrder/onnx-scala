package org.emergentorder.onnx.backends

import scala.language.implicitConversions
import scala.concurrent.Future
import scala.jdk.CollectionConverters._
import scala.scalajs.js.Array
import scalajs.js.JSConverters._

import cats.effect.IO
import cats.implicits._
//import typings.onnxruntimeNode.mod.{InferenceSession => OrtSession}
import org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
import org.emergentorder.onnx.onnxruntimeWeb.mod.Tensor.{^ => OnnxTensor}
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.compiletime._
import org.emergentorder.io.kjaer.compiletime._

import ORTTensorUtils._

//TODO: Clean up, remove asInstaceOf, etc.
class ORTWebModelBackend(session: IO[InferenceSession]) extends Model() with ORTOperatorBackend {

   def getInputAndOutputNodeNamesAndDims(sess: InferenceSession) = {
      val input_node_names = sess.inputNames

      val output_node_names = sess.outputNames

      (input_node_names.toList, output_node_names.toList)
   }

   val inputNames  = session.map(getInputAndOutputNodeNamesAndDims(_)._1)
   val outputNames = session.map(getInputAndOutputNodeNamesAndDims(_)._2)

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
      @annotation.nowarn
      val inputTensors = (0 until size)
         .map { i =>
            val tup = inputs.drop(i).take(1)
            tup match { // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
               case t: Tuple1[_] =>
                  t(0) match {
                     case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                        tens.data.flatMap(dat =>
                           tens.shape.map(shap =>
                              getOnnxTensor(dat, shap).asInstanceOf[OnnxTensor[T]]
                           )
                        )
                  }
            }
         }
         .toList
         .sequence
         .map(_.toArray)

      val output = inputTensors.flatMap { tns =>
         inputNames.flatMap { inNames =>
            outputNames.flatMap { outNames =>
               runModel[T, Tt, Td, S](
                 session,
                 tns,
                 inNames,
                 outNames
               )
            }
         }
      }
      output // .flatten
   }

}
