package org.emergentorder.onnx

import scala.reflect.ClassTag
import spire.math.Numeric
import org.emergentorder.onnx._

class ONNXBytesDataSource(onnxBytes: Array[Byte]) extends AutoCloseable with DataSource {

  val onnxHelper = new ONNXHelper(onnxBytes)

  val paramsMap =
    onnxHelper.params
      .map(x => x._1 -> (x._2, x._3, x._4))
      .toMap

  override def getParams[T: Numeric: ClassTag](name: String): Tensor[T] = {
    val params = paramsMap.get(name)
    params match {
      case Some(x) => TensorFactory.getTensor(x._2.asInstanceOf[Array[T]], x._3)
      case None =>
        throw new Exception("No params found for param name: " + name)
    }
  }

  override def close(): Unit = {

    onnxHelper.close

  }

}
