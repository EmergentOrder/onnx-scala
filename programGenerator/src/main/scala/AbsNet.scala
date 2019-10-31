package org.emergentorder.onnx

import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UInt
import spire.math.ULong
import spire.math.UShort
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import scala.language.higherKinds
import org.emergentorder.union._

class AbsNet[T: Numeric: ClassTag] {
  val dataSource: DataSource = ???
  val Abs: Abs               = ???
  def program(inputData: Option[Tensor[T]])(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
          Short
        ]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): List[Tensor[T]] = {
    for {
      nodedata <- List(inputData)
      nodeabs <- List(
        Abs.Abs6[T](
          "abs",
          X = nodedata
        )
      )
    } yield (nodeabs)
  }
}
