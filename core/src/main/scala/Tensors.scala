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

  type TensorTypeDenotation = String with Singleton

  type DimensionDenotation = String with Singleton 
//  type VecDenotation[I <: DimensionDenotation] = I #: SNil
//  type MatDenotation[I <: DimensionDenotation, J <: DimensionDenotation] = I #: J #: SNil
//  type TensorRank3Denotation[I <: DimensionDenotation, J <: DimensionDenotation, K <: DimensionDenotation] = I #: J #: K #: SNil
//  type TensorRank4Denotation[I <: DimensionDenotation, J <: DimensionDenotation, K <: DimensionDenotation, L <: DimensionDenotation] = I #: J #: K #: L #: SNil

//  type Dimension = Int with Singleton
  type VecShape[I <: Dimension] = I #: SNil
  type MatShape[I <: Dimension, J <: Dimension] = I #: J #: SNil
  type TensorRank3Shape[I <: Dimension, J <: Dimension, K <: Dimension] = I #: J #: K #: SNil
  type TensorRank4Shape[I <: Dimension, J <: Dimension, K <: Dimension, L <: Dimension] = I #: J #: K #: L #: SNil


  //TODO: use "super" trait here, to avoid incorrect type inference, and remove NEQ checks in NDScala
  sealed trait Axes
  sealed case class Undefined() extends Axes
  sealed case class Scalar[T <: TensorTypeDenotation, D <: DimensionDenotation]()                             extends Axes
  sealed case class Vec[T <: TensorTypeDenotation, D <: DimensionDenotation, Q <: Shape](i: Dimension) extends Axes
  sealed case class Mat[T <: TensorTypeDenotation, D <: DimensionDenotation, D1 <: DimensionDenotation, Q <: Shape](i: Dimension, j: Dimension)
      extends Axes

  sealed case class TensorRank3[T <: TensorTypeDenotation, D <: DimensionDenotation, D1 <: DimensionDenotation, D2 <: DimensionDenotation, Q <: Shape](
    i:Dimension,
    j:Dimension,
    k:Dimension
) extends Axes

  sealed case class TensorRank4[T <: TensorTypeDenotation, D <: DimensionDenotation, D1 <: DimensionDenotation, D2 <: DimensionDenotation, D3 <: DimensionDenotation,
  Q <: Shape](
    i:Dimension,
    j:Dimension,
    k:Dimension,
    l:Dimension
) extends Axes

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
  type OSTensor[T <: Supported, A <: Axes] = Tuple2[Array[T], A]

  object Tensor {
    extension[T <: Supported, A <: Axes](tens: OSTensor[T,A]) def data = tens._1
//    def apply[T <: Supported] (elem: T): OSTensor[T, Scalar] = new OSTensor[T, ?](Array(elem), Scalar())
    extension[T <: Supported, A <: Axes](tens: OSTensor[T,A]) def shape = tens._2 match {
      case Scalar() => Array[Int]()
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

    def apply[T <: Supported, Tt <: TensorTypeDenotation, D <: DimensionDenotation](arr: Array[T]): OSTensor[T, Scalar[Tt, D]] = tensorRequires(arr, Scalar[Tt, D]()) 

    def apply[T <: Supported, Tt <: TensorTypeDenotation, D <: DimensionDenotation](arr: Array[T], d0: Dimension): OSTensor[T, Vec[Tt, D, VecShape[d0.type]]] = tensorRequires(arr, Vec[Tt, D, VecShape[d0.type]](d0))
    def apply[T <: Supported, Tt <: TensorTypeDenotation, D <: DimensionDenotation, D1 <: DimensionDenotation](arr: Array[T], d0: Dimension, d1: Dimension): OSTensor[T, Mat[Tt, D, D1, MatShape[d0.type,d1.type]]] = (arr,Mat[Tt, D, D1, MatShape[d0.type,d1.type]](d0, d1))
    def apply[T <: Supported, Tt <: TensorTypeDenotation, D <: DimensionDenotation, D1 <: DimensionDenotation, D2 <: DimensionDenotation](arr: Array[T], d0: Dimension, d1: Dimension, d2: Dimension): OSTensor[T, TensorRank3[Tt, D, D1, D2, TensorRank3Shape[d0.type, d1.type, d2.type]]] = (arr, TensorRank3[Tt, D, D1, D2, TensorRank3Shape[d0.type,d1.type,d2.type]](d0,d1,d2))

    def apply[T <: Supported, Tt <: TensorTypeDenotation, D <: DimensionDenotation, D1 <: DimensionDenotation, D2 <: DimensionDenotation, D3 <: DimensionDenotation](arr: Array[T], d0: Dimension, d1: Dimension, d2: Dimension, d3: Dimension): OSTensor[T, TensorRank4[Tt, D, D1, D2, D3, TensorRank4Shape[d0.type, d1.type, d2.type, d3.type]]] = (arr, TensorRank4[Tt, D, D1, D2, D3, TensorRank4Shape[d0.type,d1.type,d2.type,d3.type]](d0,d1,d2, d3))

    def create[T <: Supported, Ax <: Axes](arr: Array[T], shape: Array[Int]): OSTensor[T, Ax] = {
      shape.size match {
      case 0 => apply(arr).asInstanceOf[OSTensor[T, Ax]]
      case 1 => apply(arr, shape(0).asInstanceOf[Dimension]).asInstanceOf[OSTensor[T, Ax]]
      case 2 => apply(arr, shape(0).asInstanceOf[Dimension], shape(1).asInstanceOf[Dimension]).asInstanceOf[OSTensor[T, Ax]]
      case 3 => apply(arr, shape(0).asInstanceOf[Dimension], shape(1).asInstanceOf[Dimension], shape(2).asInstanceOf[Dimension]).asInstanceOf[OSTensor[T, Ax]]
      case 4 => apply(arr, shape(0).asInstanceOf[Dimension], shape(1).asInstanceOf[Dimension], shape(2).asInstanceOf[Dimension], shape(3).asInstanceOf[Dimension]).asInstanceOf[OSTensor[T, Ax]]
      case _ => ???
    }
    }
  }
}
