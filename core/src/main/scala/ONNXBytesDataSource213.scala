package org.emergentorder.onnx

import scala.reflect.ClassTag
import spire.math.Numeric
import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._

class ONNXBytesDataSource(onnxBytes: Array[Byte]) extends AutoCloseable with DataSource {

  val onnxHelper = new ONNXHelper(onnxBytes)


  //TODO: produce tensors with axes derived from denotations
  //TODO: return non-tensor params
  override def getParams[T, Ax <: Axes](name: String): Tensor[T,Ax] = {
    val params = onnxHelper.params.filter(x => x._1 == name).headOption
    params match {
      case Some(x) => Tensor.create(x._3.asInstanceOf[Array[T]], x._4)
      case None =>
        throw new Exception("No params found for param name: " + name)
    }
  }

  override def close(): Unit = {

//    onnxHelper.close

  }

}
