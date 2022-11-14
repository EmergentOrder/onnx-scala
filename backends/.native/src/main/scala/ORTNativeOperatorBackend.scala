package org.emergentorder.onnx.backends

import scala.concurrent.duration._

import scala.concurrent.Future
import scala.language.postfixOps

//import ORTTensorUtils._
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

trait ORTNativeOperatorBackend {

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
      ???
      // TODO
      //
      //
      /*
     // TODO: prevent passing input to opToONNXBytes

      val modelProto = opToModelProto(opName, inputs, attrs)

      val result: Tensor[T, Tuple3[Tt, Td, S]] = callByteArrayOp(modelProto.toByteArray, inputs)
      result
       */
   }
}
