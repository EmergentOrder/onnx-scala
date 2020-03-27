package org.emergentorder

import scala.language.higherKinds
import scala.{specialized => sp}
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import spire.implicits._
import spire.algebra.Field
import scala.reflect.ClassTag
import org.bytedeco.onnx.ModelProto
package object onnx {

  type ![A]  = A => Nothing
  type !![A] = ![![A]]

  trait Disjunction[T] {
    type or[S]  = Disjunction[T with ![S]]
    type create = ![T]
  }

  type Union[T] = {
    type or[S] = Disjunction[![T]]#or[S]
  }

  type Contains[S, T] = !![S] <:< T

  type UNil

  trait Dim

  sealed trait Axes

  sealed trait Scalar                                    extends Axes
  sealed trait Vec[T <: Dim]                             extends Axes
  sealed trait Mat[T <: Dim, U <: Dim]                   extends Axes
  sealed trait Tuple3OfDim[T <: Dim, U <: Dim, V <: Dim] extends Axes

  type TypesafeTensor[T, A <: Axes] = Tuple2[Array[T], Array[Int]]

  type Tensor[T]       = TypesafeTensor[T, Axes]
  type SparseTensor[T] = Tensor[T]

  type XInt = Int with Singleton

  object TensorFactory {
    def getTensor[T](data: Array[T], t: Array[Int]): Tensor[T] = {
      require(data.size == t.foldLeft(1)(_ * _))
      (data, t)
    }
  }

  trait Operator {
    def callOp[
        T: ClassTag,
        T1: ClassTag,
        T2: ClassTag,
        T3: ClassTag,
        T4: ClassTag,
        T5: ClassTag,
        T6: ClassTag,
        T7: ClassTag,
        T8: ClassTag,
        T9: ClassTag,
        T10: ClassTag,
        T11: ClassTag,
        T12: ClassTag,
        T13: ClassTag,
        T14: ClassTag,
        T15: ClassTag,
        T16: ClassTag,
        T17: ClassTag
    ](
        name: String,
        opName: String,
        inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
        //    outName: String,
        attrs: Map[String, Any]
    ): (T9)
  }

  abstract class Model(onnxBytes: Array[Byte]) extends Operator {
    def fullModel[
        T: ClassTag,
        T1: ClassTag,
        T2: ClassTag,
        T3: ClassTag,
        T4: ClassTag,
        T5: ClassTag,
        T6: ClassTag,
        T7: ClassTag,
        T8: ClassTag,
        T9: ClassTag,
        T10: ClassTag,
        T11: ClassTag,
        T12: ClassTag,
        T13: ClassTag,
        T14: ClassTag,
        T15: ClassTag,
        T16: ClassTag,
        T17: ClassTag
    ](
        inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8]
    ): (T9)
  }

  trait Graph
  trait DataSource {
    def getParams[T: Numeric: ClassTag](name: String): Tensor[T]
  }
}
