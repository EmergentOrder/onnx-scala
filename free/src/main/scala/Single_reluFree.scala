package org.emergentorder.onnxFree

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

object Single_reluFree {
  val reluFree: ReluFree = new ONNXNGraphHandlers.ReluHandler()
  val dataSource: DataSourceFree = new ONNXNGraphHandlers.DatasourceHandler()
  val dropoutFree: DropoutFree = new ONNXNGraphHandlers.DropoutHandler()
  def program
  : Task[Tensor[Float]]  = 
    for {
      nodex <- dataSource.inputDataFree[Float].fork
      nodexjoined <- nodex.join
      nodey <- reluFree.Relu6Free("y" ,X = Some(nodexjoined)).fork
      nodeyjoined <- nodey.join
      nodez <- dropoutFree.Dropout7Free("z" , None, data = Some(nodeyjoined)).fork
      nodezjoined <- nodez.join
    } yield (nodezjoined._1)
}
