package org.emergentorder.onnx

import scala.reflect.ClassTag
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import io.kjaer.compiletime._


import org.emergentorder.compiletime.DimensionDenotation
import org.emergentorder.compiletime.TensorDenotation
import org.emergentorder.compiletime.tensorDenotationOf

object Tensors{

  type Supported = Int | Long | Float | Double | Byte | Short | UByte | UShort | UInt | ULong | 
                   Boolean | String | Float16 | Complex[Float] | Complex[Double]

  type TensorTypeDenotation = String with Singleton

//  type VecDenotation[I <: DimensionDenotation] = I #: SNil
//  type MatDenotation[I <: DimensionDenotation, J <: DimensionDenotation] = I #: J #: SNil
//  type TensorRank3Denotation[I <: DimensionDenotation, J <: DimensionDenotation, K <: DimensionDenotation] = I #: J #: K #: SNil
//  type TensorRank4Denotation[I <: DimensionDenotation, J <: DimensionDenotation, K <: DimensionDenotation, L <: DimensionDenotation] = I #: J #: K #: L #: SNil

//  type Dimension = Int with Singleton
  type VecShape[I <: Dimension] = I #: SNil
  type MatShape[I <: Dimension, J <: Dimension] = I #: J #: SNil
  type TensorRank3Shape[I <: Dimension, J <: Dimension, K <: Dimension] = I #: J #: K #: SNil
  type TensorRank4Shape[I <: Dimension, J <: Dimension, K <: Dimension, L <: Dimension] = I #: J #: K #: L #: SNil

//  case class Axes(ttd: TensorTypeDenotation,td: TensorDenotation, shape: Shape)
  type Axes = Tuple3[TensorTypeDenotation, TensorDenotation, Shape]

//  type Axes = DenotedTensor[? <: TensorTypeDenotation, ? <: TensorDenotation, ? <: Shape]

  /*
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
*/
  //Need this alias to not conflict with other Tensors
  type Tensor[T <: Supported, Ax <: Axes] = OSTensor[T, Ax]  //(Array[T], Array[Int])

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
  type OSTensor[T <: Supported, Ax <: Axes] = Tuple2[Array[T], Ax]

  object Tensor {
    extension[T <: Supported,  Tt <: TensorTypeDenotation, Td <: TensorDenotation, S <: Shape](tens: OSTensor[T,Tuple3[Tt, Td, S]]) def data = tens._1
//    def apply[T <: Supported] (elem: T): OSTensor[T, Scalar] = new OSTensor[T, ?](Array(elem), Scalar())

    extension[T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorDenotation, S <: Shape](tens: OSTensor[T,Tuple3[Tt, Td, S]]) def shape: Array[Int] = tens._2._3.toSeq.toArray 
      
  def tensorRequires[T <: Supported,  Tt <: TensorTypeDenotation, Td <: TensorDenotation, S <: Shape](tens: OSTensor[T,Tuple3[Tt,Td,S]]): OSTensor[T,Tuple3[Tt, Td, S]] = {
    require(tens.shape.size <= 4)
    require(tens._1.size == tens.shape.foldLeft(1)(_ * _))
    tens
  }
    def apply[T <: Supported : scala.reflect.ClassTag, Tt <: TensorTypeDenotation](element: T, tt: Tt): OSTensor[T, Tuple3[Tt, org.emergentorder.compiletime.SSNil, SNil]] = tensorRequires((Array[T](element), (tt, org.emergentorder.compiletime.SSNil, SNil))) 

    def apply[T <: Supported, Tt <: TensorTypeDenotation, TD <: TensorDenotation, S <: Shape](arr: Array[T],tt: Tt, td0: TD, d0: S): OSTensor[T, Tuple3[Tt, TD, S]] = tensorRequires((arr, (tt, td0, d0)))

    //InstanceOf
    def create[T <: Supported, Tt <: TensorTypeDenotation, TD <: TensorDenotation](arr: Array[T],tt: Tt, td: TD, shape: Array[Int]): OSTensor[T, (Tt, TD, ? <: Shape)] = {

      apply(arr, tt, td, Shape.fromSeq(shape))
    
    }
  }
}
