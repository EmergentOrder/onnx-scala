package org.emergentorder.onnx

import scala.reflect.ClassTag
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import io.kjaer.compiletime._

object Tensors{

  type Supported = Int | Long | Float | Double | Byte | Short | UByte | UShort | UInt | ULong | 
                   Boolean | String | Float16 | Complex[Float] | Complex[Double]

  type Dimension = Int with Singleton
  type VecShape[I <: Dimension] = I #: SNil
  type MatShape[I <: Dimension, J <: Dimension] = I #: J #: SNil
  type TensorRank3Shape[I <: Dimension, J <: Dimension, K <: Dimension] = I #: J #: K #: SNil
  type TensorRank4Shape[I <: Dimension, J <: Dimension, K <: Dimension, L <: Dimension] = I #: J #: K #: L #: SNil


  sealed trait Axes
  sealed case class Scalar()                             extends Axes
  sealed case class Vec[I <: Dimension, Q <: VecShape[I]](i: I) extends Axes
  sealed case class Mat[I <: Dimension, J <: Dimension, Q <: MatShape[I,J]](i: I, j: J)
      extends Axes

  sealed case class TensorRank3[I <: Dimension, J <: Dimension, K <: Dimension,
  Q <: TensorRank3Shape[I,J,K]](
      i: I,
      j: J,
      k: K,
) extends Axes

  sealed case class TensorRank4[I <: Dimension, J <: Dimension, K <: Dimension, L <: Dimension,
  Q <: TensorRank4Shape[I,J,K, L]](
      i: I,
      j: J,
      k: K,
      l: L
) extends Axes

  type XInt

  //Need this alias to not conflict with other Tensors
  type Tensor[T <: Supported, A <: Axes] = OSTensor[T, A]  //(Array[T], Array[Int])

  type SparseTensor[T <: Supported, A <: Axes] = Tensor[T, A]

  //TODO: random nd access
  //Supports up to tensor rank 4 right now
  //TODO: use IArray ?
  //
//  case class OSTensor[T <: Supported, A <: Axes](data: Array[T], axes: A){
//    val _1: Array[T] = data
/*
    val _2: Array[Int] = axes match {
      case Scalar => Array()
      case Vec(i) => Array(i)
      case Mat(i, j) => Array(i,j)
      case TensorRank3(i,j,k) => Array(i,j,k) 
      case TensorRank4(i,j,k,l) => Array(i,j,k,l)
      case _ => ???
    }
*/
      //Array(dim0,dim1,dim2).map(_.asInstanceOf[Option[Int]]).flatten
  //  require(_2.size <= 4)
  //  require(data.size == _2.foldLeft(1)(_ * _)) 
//  }
//TODO: restore requires
//
//TODO: opaque
  type OSTensor[T <: Supported, A <: Axes] = (Array[T], A)

  object Tensor {
    extension[T <: Supported, A <: Axes](tens: OSTensor[T,A]) def data = tens._1
//    def apply[T <: Supported] (elem: T): OSTensor[T, Scalar] = new OSTensor[T, ?](Array(elem), Scalar())
    extension[T <: Supported, A <: Axes](tens: OSTensor[T,A]) def shape = tens._2 match {
      case Scalar => Array[Int]()
      case Vec(i) => Array(i)
      case Mat(i, j) => Array(i,j)
      case TensorRank3(i,j,k) => Array(i,j,k)
      case TensorRank4(i,j,k,l) => Array(i,j,k,l)
      case _ => ???
    } 
  def tensorRequires[T <: Supported, A <: Axes](tens: OSTensor[T,A]): OSTensor[T,A] = {
    require(tens.shape.size <= 4)
    require(tens._1.size == tens.shape.foldLeft(1)(_ * _))
    tens
  }

    def apply[T <: Supported, I <: Dimension](arr: Array[T], d0: I): OSTensor[T, Vec[I, VecShape[I]]] = tensorRequires(arr, Vec[I, VecShape[I]](d0))
    def apply[T <: Supported, I <: Dimension, J <: Dimension](arr: Array[T], d0: I, d1: J): OSTensor[T, Mat[I,J, MatShape[I,J]]] = (arr,Mat[I, J, MatShape[I,J]](d0, d1))
    def apply[T <: Supported, I <: Dimension, J <: Dimension, K <: Dimension](arr: Array[T], d0: I, d1: J, d2: K): OSTensor[T, TensorRank3[I,J,K, TensorRank3Shape[I,J,K]]] = (arr, TensorRank3[I,J,K, TensorRank3Shape[I,J,K]](d0,d1,d2))

    def apply[T <: Supported, I <: Dimension, J <: Dimension, K <: Dimension, L <: Dimension](arr: Array[T], d0: I, d1: J, d2: K, d3: L): OSTensor[T, TensorRank4[I,J,K, L, TensorRank4Shape[I,J,K, L]]] = (arr, TensorRank4[I,J,K,L, TensorRank4Shape[I,J,K, L]](d0,d1,d2, d3))

    def create[T <: Supported, Ax <: Axes](arr: Array[T], shape: Array[Int]): OSTensor[T, Ax] = {
      shape.size match {
 //     case 0 => apply(arr(0))
      case 1 => apply(arr, shape(0).asInstanceOf[Dimension]).asInstanceOf[OSTensor[T, Ax]]
      case 2 => apply(arr, shape(0).asInstanceOf[Dimension], shape(1).asInstanceOf[Dimension]).asInstanceOf[OSTensor[T, Ax]]
      case 3 => apply(arr, shape(0).asInstanceOf[Dimension], shape(1).asInstanceOf[Dimension], shape(2).asInstanceOf[Dimension]).asInstanceOf[OSTensor[T, Ax]]
      case 4 => apply(arr, shape(0).asInstanceOf[Dimension], shape(1).asInstanceOf[Dimension], shape(2).asInstanceOf[Dimension], shape(3).asInstanceOf[Dimension]).asInstanceOf[OSTensor[T, Ax]]
      case _ => ???
    }
    }
  }
}
