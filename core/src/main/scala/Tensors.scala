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
import org.emergentorder.compiletime.TensorShapeDenotation
import org.emergentorder.compiletime.tensorShapeDenotationOf

object Tensors{

  type Supported = Int | Long | Float | Double | Byte | Short | UByte | UShort | UInt | ULong | 
                   Boolean | String | Float16 | Complex[Float] | Complex[Double]

  type TensorTypeDenotation = String & Singleton

//  case class Axes(ttd: TensorTypeDenotation,td: TensorShapeDenotation, shape: Shape)
  type Axes = Tuple3[TensorTypeDenotation, TensorShapeDenotation, Shape]


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
//TODO: Benchmark Array[T] vs ArraySeq[T] vs IArray[T]
  type OSTensor[T <: Supported, Ax <: Axes] = Tuple2[Array[T], Ax]

  object Tensor {
    extension[T <: Supported,  Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](tens: OSTensor[T,Tuple3[Tt, Td, S]]) def data = tens._1
//    def apply[T <: Supported] (elem: T): OSTensor[T, Scalar] = new OSTensor[T, ?](Array(elem), Scalar())

    extension[T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](tens: OSTensor[T,Tuple3[Tt, Td, S]]) def shape: Array[Int] = tens._2._3.toSeq.toArray 
      
  def tensorRequires[T <: Supported,  Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](tens: OSTensor[T,Tuple3[Tt,Td,S]]): OSTensor[T,Tuple3[Tt, Td, S]] = {
    require(tens.shape.size <= 4)
    require(tens._1.size == tens.shape.foldLeft(1)(_ * _))
    tens
  }
    def apply[T <: Supported : scala.reflect.ClassTag, Tt <: TensorTypeDenotation](element: T, tt: Tt): OSTensor[T, Tuple3[Tt, org.emergentorder.compiletime.TSNil, SNil]] = tensorRequires((Array[T](element), (tt, org.emergentorder.compiletime.TSNil, SNil))) 

    def apply[T <: Supported, Tt <: TensorTypeDenotation, TD <: TensorShapeDenotation, S <: Shape](arr: Array[T], tt0: Tt, td0: TD, d0: S): OSTensor[T, Tuple3[Tt, TD, S]] = tensorRequires((arr, (tt0, td0, d0)))
/*
    //InstanceOf
    def create[T <: Supported, Tt <: TensorTypeDenotation, TD <: TensorShapeDenotation, S <: Shape](arr: Array[T],tt: Tt, td: TD, shape: Array[Int]): OSTensor[T, (Tt, TD, S)] = {

      apply(arr, tt, td, Shape.fromSeq(shape))
    
    }
    */
  }
}
