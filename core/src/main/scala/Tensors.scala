package org.emergentorder.onnx

import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric



object Tensors{

trait DimName
type Dimension = Int with Singleton
/*
type VecShape[I <: Dimension] = I #: SNil
type MatShape[I <: Dimension, J <: Dimension] = I #: J #: SNil
type TensorRank3Shape[I <: Dimension, J <: Dimension, K <: Dimension] = I #: J #: K #: SNil

sealed trait Axes


sealed case class Scalar()                             extends Axes
sealed case class Vec[I <: Dimension, T <: DimName, Q <: VecShape[I]](i: I, t: T) extends Axes
sealed case class Mat[I <: Dimension, T <: DimName, J <: Dimension, U <: DimName, Q <: MatShape[I,J]](i: I, t: T, j: J, u: U)
      extends Axes
sealed case class TensorRank3[I <: Dimension, T <: DimName, J <: Dimension, U <: DimName, K <: Dimension, V <: DimName, 
  Q <: TensorRank3Shape[I,J,K]](
      i: I,
      t: T,
      j: J,
      u: U,
      k: K,
      v: V
) extends Axes
//TODO: 4+ dimensional

object AxesFactory {
  //TODO: make more specific
    def getAxes[A <: Axes](shape: Array[Dimension], dims: Array[DimName]): Axes = {
      if (shape.length == 3) {
        val t0 = shape(0)
        val d0 = dims(0)
        val t1 = shape(1)
        val d1 = dims(1)
        val t2 = shape(2)
        val d2 = dims(2)
        new TensorRank3[t0.type, d0.type, t1.type, d1.type, t2.type, d2.type, TensorRank3Shape[t0.type, t1.type, t2.type]](
          t0,
          d0,
          t1,
          d1,
          t2,
          d2
        )
      } else if (shape.length == 1) {
        val t0 = shape(0)
        val d0 = dims(0)
        new Vec[t0.type, d0.type, VecShape[t0.type]](t0, d0)
      } else if (shape.length == 0) (new Scalar)
      else {
        val t0 = shape(0)
        val d0 = dims(0)
        val t1 = shape(1)
        val d1 = dims(1)
        new Mat[t0.type, d0.type, t1.type, d1.type, MatShape[t0.type, t1.type]](t0, d0, t1, d1)
      }

    }
  }
*/
  type Supported = Int | Long | Float | Double | Byte | Short | UByte | UShort | UInt | ULong | 
                   Boolean | String | Float16 | Complex[Float] | Complex[Double]

  type FloatSupported = Float | Double 

//  type TypesafeTensor[T, A <: Axes] = Tuple3[Array[T], Array[Int], A]

  //TODO: Use IArray
  type Tensor[T <: Supported]       = Tuple2[Array[T], Array[Int]] //TypesafeTensor[T, Axes]
  type SparseTensor[T <: Supported] = Tensor[T]

  object TensorFactory {

    def getTensor[T <: Supported](data: Array[T], t: Array[Int]): Tensor[T] = {
      val shape: Array[Dimension] = t.map(z => z: Dimension)
      require(data.size == shape.foldLeft(1)(_ * _))
      (data, t)
      //(data, t, AxesFactory.getAxes(shape, Array.fill(shape.size) { new DimName {} }))
    }
  /*
    def getTypesafeTensor[T, A <: Axes](data: Array[T], axes: A): TypesafeTensor[T, A] = {

//      val axes: A = AxesFactory.getAxes(shape, dims)
      val t: Array[Int] = axes match {
        case Scalar()                      => Array()
        case Vec(i, _)                     => Array(i)
        case Mat(i, _, j, _)               => Array(i, j)
        case TensorRank3(i, _, j, _, k, _) => Array(i, j, k)
      }

      val shape: Array[Dimension] = t.map(z => z: Dimension)

      require(data.size == shape.foldLeft(1)(_ * _))
      (data, t, axes)
    }
    */
  }
  
}
