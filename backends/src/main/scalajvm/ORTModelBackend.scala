package org.emergentorder.onnx.backends

import ai.onnxruntime.*
import cats.effect.IO
//import cats.effect.unsafe.implicits.global
import cats.implicits.*
import org.emergentorder.compiletime.*
import org.emergentorder.io.kjaer.compiletime.*
import org.emergentorder.onnx.Tensors.Tensor.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.*

import scala.jdk.CollectionConverters.*
import scala.language.implicitConversions

import compiletime.asMatchable
import ORTTensorUtils.*

//TODO: Clean up, remove asInstaceOf, etc.
class ORTModelBackend(onnxBytes: Array[Byte])
    extends Model()
    with ORTOperatorBackend
    with AutoCloseable {

   def getInputAndOutputNodeNamesAndDims(
//       sess: OrtSession
   ): (List[String], Array[Array[Long]], List[String]) = {
      val input_node_names = session.getInputNames

      val inputNodeDims =
         session.getInputInfo.values.asScala.map(_.getInfo.asInstanceOf[TensorInfo].getShape)

      val output_node_names = session.getOutputNames

      (input_node_names.asScala.toList, inputNodeDims.toArray, output_node_names.asScala.toList)
   }

   val session: OrtSession = getSession(onnxBytes)

   val allNodeNamesAndDims: (List[String], Array[Array[Long]], List[String]) =
      getInputAndOutputNodeNamesAndDims()

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
      def inputTensors = (0 until size)
         .map { i =>
            val tup = inputs.drop(i).take(1)
            tup match { // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
               case t: Tuple1[?] =>
                  t(0).asMatchable match {
                     case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                        tens.data.flatMap(x => tens.shape.map(y => getOnnxTensor(x, y, env)))
                  }
            }
         }
         .toList
         .sequence
         .map(_.toArray)

      val output = cats.effect.Resource
         .make(inputTensors)(inTens => IO.blocking { inTens.map(_.close) })
         .use(inTens =>
            runModel[T, Tt, Td, S](
              session,
              inTens,
              allNodeNamesAndDims._1 
            )
         )
      output
      //output.memoize.unsafeRunSync()
   }

   override def close(): Unit = {env.close}
}
