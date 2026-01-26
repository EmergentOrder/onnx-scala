package org.emergentorder.onnx

import org.emergentorder.compiletime.*
import org.emergentorder.io.kjaer.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.*

class ONNXBytesDataSource(onnxBytes: Array[Byte]) extends DataSource {

   val onnxHelper = new ONNXHelper(onnxBytes)

   // TODO: produce tensors with axes derived from denotations
   // TODO: return non-tensor params
   override def getParams[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](name: String)(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {

      "TEMP" ##: TSNil

      val shapeFromType: S              = s.value
      val tensorTypeDenotationFromType  = tt.value
      val tensorShapeDenotationFromType = td.value

      val params = onnxHelper.params.filter(x => x._1 == name).headOption
      params match {
         case Some(x) => {
            require(x._4 sameElements shapeFromType.toSeq)
            Tensor(
              x._3.asInstanceOf[Array[T]],
              tensorTypeDenotationFromType,
              tensorShapeDenotationFromType,
              shapeFromType
            )

         }
         case None =>
            throw new Exception("No params found for param name: " + name)
      }
   }
}
