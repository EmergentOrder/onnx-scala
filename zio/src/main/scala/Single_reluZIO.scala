package org.emergentorder.onnxZIO

import scalaz.zio.Task
import org.emergentorder.onnx._
import org.emergentorder.union.UnionType._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import scala.language.higherKinds

object Single_reluZIO {
  val reluZIO: ReluZIO = new ONNXNGraphHandlers.ReluHandler()
  val dataSource: DataSourceZIO = new ONNXNGraphHandlers.DatasourceHandler()
  val dropoutZIO: DropoutZIO = new ONNXNGraphHandlers.DropoutHandler()
  def program
  : Task[Tensor[Float]]  = 
    for {
      nodex <- dataSource.inputDataZIO[Float].fork
      nodexjoined <- nodex.join
      nodey <- reluZIO.Relu6ZIO("y" ,X = Some(nodexjoined)).fork
      nodeyjoined <- nodey.join
      nodez <- dropoutZIO.Dropout7ZIO("z" , None, data = Some(nodeyjoined)).fork
      nodezjoined <- nodez.join
    } yield (nodezjoined._1)
}
