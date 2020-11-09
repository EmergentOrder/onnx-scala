package org.emergentorder.onnx

import scala.reflect.ClassTag
import spire.math.Numeric
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.compiletime._
import io.kjaer.compiletime.Shape

class ONNXBytesDataSource(onnxBytes: Array[Byte]) extends AutoCloseable with DataSource {

  val onnxHelper = new ONNXHelper(onnxBytes)


  
  //TODO: produce tensors with axes derived from denotations
  //TODO: return non-tensor params
  override def getParams[T <: Supported](name: String): Tensor[T,Tuple3[TensorTypeDenotation,TensorDenotation,Shape]] = {
 
  val tensorTypeDenotation: TensorTypeDenotation = "TEMP"
  val tensorDenotation: TensorDenotation = "TEMP" ##: SSNil

    val params = onnxHelper.params.filter(x => x._1 == name).headOption
    params match {
      case Some(x) => Tensor(x._3.asInstanceOf[Array[T]],tensorTypeDenotation, tensorDenotation, Shape.fromSeq(x._4))
      case None =>
        throw new Exception("No params found for param name: " + name)
    }
  }

  override def close(): Unit = {

//    onnxHelper.close

  }

}
