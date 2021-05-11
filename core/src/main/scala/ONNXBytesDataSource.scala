package org.emergentorder.onnx

import scala.reflect.ClassTag
import spire.math.Numeric
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

class ONNXBytesDataSource(onnxBytes: Array[Byte]) extends AutoCloseable with DataSource {

   val onnxHelper = new ONNXHelper(onnxBytes)

   //TODO: produce tensors with axes derived from denotations
   //TODO: return non-tensor params
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

      val tensorTypeDenotation: TensorTypeDenotation = "TEMP"
      val tensorDenotation: TensorShapeDenotation    = "TEMP" ##: TSNil

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

   override def close(): Unit = {

//    onnxHelper.close

   }

}
