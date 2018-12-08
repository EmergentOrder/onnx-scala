package org.emergentorder.onnx

import org.emergentorder.onnx.UnionType._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import singleton.ops._
import scala.language.higherKinds

trait AbsNet {
  val dataSource: DataSource
  val Abs: Abs
  def program[
      T: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check: Numeric: ClassTag]
    : List[Tensor[T]] =
    for {
      nodedata <- List(dataSource.inputData[T])
      nodeabs <- List(
        Abs.Abs6[T](
          "abs",
          X = Some(nodedata)
        ))
    } yield (nodeabs)
}
