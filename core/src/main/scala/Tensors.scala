package org.emergentorder.onnx

import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import io.kjaer.compiletime._
import scala.compiletime.S

import org.emergentorder.compiletime.DimensionDenotation
import org.emergentorder.compiletime.TensorShapeDenotation
import org.emergentorder.compiletime.tensorShapeDenotationOf
import org.emergentorder.compiletime.TSNil
import org.emergentorder.compiletime.##:

object Tensors{

  type Supported = Int | Long | Float | Double | Byte | Short | UByte | UShort | UInt | ULong | 
                   Boolean | String | Float16 | Complex[Float] | Complex[Double]

  type TensorTypeDenotation = String & Singleton

//  case class Axes(ttd: TensorTypeDenotation,td: TensorShapeDenotation, shape: Shape)
  type Axes = Tuple3[TensorTypeDenotation, TensorShapeDenotation, Shape]


  //Need this alias to not conflict with other Tensors
  opaque type Tensor[T <: Supported, Ax <: Axes] = Tuple2[Array[T], Ax]

  type SparseTensor[T <: Supported, A <: Axes] = Tensor[T, A]
 
  type KeepOrReduceDims[S <: Shape, AxisIndices <: None.type | Indices, KeepDims <: (Boolean & Singleton)] <: Shape = (KeepDims) match {
        case true => ReduceKeepDims[S, AxisIndices]
        case false => Shape.Reduce[S, AxisIndices]
  }

  type KeepOrReduceDimDenotations[Td <: TensorShapeDenotation, AxisIndices <: None.type | Indices, KeepDims <: (Boolean & Singleton)] <: TensorShapeDenotation = (KeepDims) match {
        case true => Td
        case false => TensorShapeDenotation.Reduce[Td, AxisIndices]
  }

  type ReduceKeepDims[S <: Shape, Axes <: None.type | Indices] <: Shape = Axes match {
    case None.type => SNil
    case Indices => ReduceKeepDimsLoop[S, Axes, 0]
  }

  protected type ReduceKeepDimsLoop[ReplaceFrom <: Shape, ToReplace <: Indices, I <: Index] <: Shape = ReplaceFrom match {
    case head #: tail => Indices.Contains[ToReplace, I] match {
      case true => 1 #: ReduceKeepDimsLoop[tail, Indices.RemoveValue[ToReplace, I], S[I]]
      case false => head #: ReduceKeepDimsLoop[tail, ToReplace, S[I]]
    }
    case SNil => ToReplace match {
      case INil => SNil 
    }
  }
  /*
  type ConcatLoop[ConcatFromA <: Shape, ConcatFromB <: Shape, ToConcat <: Indices, I <: Index] <: Shape = ConcatFromA match {
    case head #: tail => Indices.Contains[ConcatFromA, I] match {
      case true => ReduceLoop[tail, Indices.RemoveValue[ToRemove, I], S[I]]
      case false => head #: ReduceLoop[tail, ToRemove, S[I]]
    }
    case SNil => ToConcat match {
      case INil => SNil
      //     case head :: tail => Error[
      //         "The following indices are out of bounds: " + Indices.ToString[ToRemove]
      //     ]
    }
  }
*/

  //TODO: shape dimension values should be longs, not ints, but dotty compiletime ops only support ints
  //TODO: random nd access
  //TODO: opaque
  //TODO: Benchmark Array[T] vs ArraySeq[T] vs IArray[T]

  object Tensor {
    extension[T <: Supported,  Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](tens: Tensor[T,Tuple3[Tt, Td, S]]) def data = tens._1

    extension[T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](tens: Tensor[T,Tuple3[Tt, Td, S]]) def shape: Array[Int] = tens._2._3.toSeq.toArray 
 
  def tensorRequires[T <: Supported,  Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](tens: Tensor[T,Tuple3[Tt,Td,S]]): Tensor[T,Tuple3[Tt, Td, S]] = {
    //require(tens._2._2.toSeq.size == tens.shape.size) //We allow empty denotations
    require(tens.shape.size <= 4)
    require(tens.data.size == tens.shape.foldLeft(1)(_ * _))
    tens
  }
    def apply[T <: Supported : scala.reflect.ClassTag, Tt <: TensorTypeDenotation, TD <: TensorShapeDenotation](element: T, tt: Tt, td: TD): Tensor[T, Tuple3[Tt, TD, 1 #: SNil]] = tensorRequires((Array[T](element), (tt, td, 1 #: SNil))) 

    def apply[T <: Supported, Tt <: TensorTypeDenotation, TD <: TensorShapeDenotation, S <: Shape](arr: Array[T], tt0: Tt, td0: TD, d0: S): Tensor[T, Tuple3[Tt, TD, S]] = tensorRequires((arr, (tt0, td0, d0)))


    def apply[T <: Supported : scala.reflect.ClassTag](element: T): Tensor[T, Tuple3["", org.emergentorder.compiletime.TSNil, 1 #: SNil]] = tensorRequires((Array(element), ("", TSNil, 1 #: SNil))) 

    def apply[T <: Supported, TD <: TensorShapeDenotation, S <: Shape](arr: Array[T], td0: TD, d0: S): Tensor[T, Tuple3["", TD, S]] = tensorRequires((arr, ("", td0, d0)))

    def apply[T <: Supported, S <: Shape](arr: Array[T], d0: S): Tensor[T, Tuple3["", "" ##: TSNil, S]] = tensorRequires((arr, ("", "" ##: TSNil, d0)))

  }
}
