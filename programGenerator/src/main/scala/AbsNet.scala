package org.emergentorder.onnx

import org.emergentorder.union.UnionType._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import scala.language.higherKinds

trait AbsNet {
  val dataSource: DataSource
  val Abs: Abs
  def program[
      T: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check: Numeric: ClassTag]
      (inputData: Tensor[T])
    : List[Tensor[T]] = {
    for {
      nodedata <- List(inputData)
      nodeabs <- List(
        Abs.Abs6[T](
          "abs",
          X = Some(nodedata)
        ))
    } yield (nodeabs)
    }
}
