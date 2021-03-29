package org.emergentorder.onnx

import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import io.kjaer.compiletime._
import scala.compiletime.S
import scala.compiletime.ops.int.*

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
  //potential collision, use type name Axes elsewhere
  type Axes = Tuple3[TensorTypeDenotation, TensorShapeDenotation, Shape]


  //Need this alias to not conflict with other Tensors
  //TODO: consider using TF-Java ndarray as backing instead of Scala Array here
  //S is overloaded
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

  type ReduceKeepDims[S <: Shape, AxisIndices <: None.type | Indices] <: Shape = AxisIndices match {
    case None.type => SNil
    case Indices => ReduceKeepDimsLoop[S, AxisIndices, 0]
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

  type DoubleGivenAxisSize[S <: Shape, AxisIndices <: None.type | Indices] <: Shape = AxisIndices match {
    case None.type => SNil
    case Indices => DoubleGivenAxisSizeLoop[S, AxisIndices, 0]
  }

  protected type DoubleGivenAxisSizeLoop[ReplaceFrom <: Shape, ToReplace <: Indices, I <: Index] <: Shape = ReplaceFrom match {
    case head #: tail => Indices.Contains[ToReplace, I] match {
      case true => (2 * head) #: DoubleGivenAxisSizeLoop[tail, Indices.RemoveValue[ToReplace, I], S[I]]
      case false => head #: DoubleGivenAxisSizeLoop[tail, ToReplace, S[I]]
    }
    case SNil => ToReplace match {
      case INil => SNil
    }
  }

  type SlicedShape[AxisIndicesStarts <: None.type | Indices, AxisIndicesEnds <: None.type | Indices] <: Shape = AxisIndicesStarts match {
    case None.type => SNil
    case Indices => AxisIndicesEnds match {
      case None.type => SNil
      case Indices => SlicedShapeLoop[AxisIndicesStarts, AxisIndicesEnds]
    }
  }

  protected type SlicedShapeLoop[Starts <: Indices, Ends <: Indices] <: Shape = Starts match {
    case head ::: tail => Ends match{
      case endsHead ::: endsTail => (endsHead - head) #: SlicedShapeLoop[tail, endsTail]
      case INil => SNil
    }
    case INil => Ends match {
      case INil => SNil
    }
  }

  type PaddedShape[PadFrom <: Shape, AxisIndicesBefore <: None.type | Indices, AxisIndicesAfter <: None.type | Indices] <: Shape = AxisIndicesBefore match {
    case None.type => SNil
    case Indices => AxisIndicesAfter match {
      case None.type => SNil
      case Indices => PaddedShapeLoop[PadFrom, AxisIndicesBefore, AxisIndicesAfter]
    }
  }

  protected type PaddedShapeLoop[PadFrom <: Shape, Before <: Indices, After <: Indices] <: Shape = Before match {
    case head ::: tail => After match{
      case afterHead ::: afterTail => PadFrom match {
        case padFromHead #: padFromTail => (head + padFromHead + afterHead) #: PaddedShapeLoop[padFromTail, tail, afterTail]
        case SNil => SNil
      }
      case INil => SNil
    }
    case INil => After match {
      case INil => SNil
    }
  }

  type TiledShape[TileFrom <: Shape, AxisRepeats <: None.type | Indices] <: Shape = AxisRepeats match {
    case None.type => SNil
    case Indices => TiledShapeLoop[TileFrom, AxisRepeats]
  }

  protected type TiledShapeLoop[TileFrom <: Shape, Repeats <: Indices] <: Shape = Repeats match {
    case head ::: tail => TileFrom match {
      case tileFromHead #: tileFromTail => (head * tileFromHead) #: TiledShapeLoop[tileFromTail, tail]
      case SNil => SNil
    }
    case INil => SNil
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
    require(tens.data.size == tens.shape.foldLeft(1)(_ * _)) //This shouldn't fail at runtime, if so shape constraints need fixing
    tens
  }
    def apply[T <: Supported : scala.reflect.ClassTag, Tt <: TensorTypeDenotation, TD <: TensorShapeDenotation](element: T, tt: Tt, td: TD): Tensor[T, Tuple3[Tt, TD, 1 #: SNil]] = tensorRequires((Array[T](element), (tt, td, 1 #: SNil))) 

    def apply[T <: Supported, Tt <: TensorTypeDenotation, TD <: TensorShapeDenotation, S <: Shape](arr: Array[T], tt0: Tt, td0: TD, d0: S): Tensor[T, Tuple3[Tt, TD, S]] = tensorRequires((arr, (tt0, td0, d0)))


    def apply[T <: Supported : scala.reflect.ClassTag](element: T): Tensor[T, Tuple3["", org.emergentorder.compiletime.TSNil, 1 #: SNil]] = tensorRequires((Array(element), ("", TSNil, 1 #: SNil))) 

    def apply[T <: Supported, TD <: TensorShapeDenotation, S <: Shape](arr: Array[T], td0: TD, d0: S): Tensor[T, Tuple3["", TD, S]] = tensorRequires((arr, ("", td0, d0)))

    def apply[T <: Supported, S <: Shape](arr: Array[T], d0: S): Tensor[T, Tuple3["", "" ##: TSNil, S]] = tensorRequires((arr, ("", "" ##: TSNil, d0)))

  }
}
