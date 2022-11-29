package org.emergentorder.onnx

import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import io.kjaer.compiletime.*
import io.kjaer.compiletime.Shape.*
import scala.compiletime.ops.int.*

import cats.effect.IO
import org.emergentorder.compiletime.DimensionDenotation
import org.emergentorder.compiletime.TensorShapeDenotation
import org.emergentorder.compiletime.tensorShapeDenotationOf
import org.emergentorder.compiletime.TSNil
import org.emergentorder.compiletime.##:

object Tensors {

   type Supported = Int | Long | Float | Double | Byte | Short | UByte | UShort | UInt | ULong |
      Boolean | String | BFloat16 | Float16 | Complex[Float] | Complex[Double]

   type TensorTypeDenotation = String & Singleton

//  case class Axes(ttd: TensorTypeDenotation,td: TensorShapeDenotation, shape: Shape)
   // potential collision, use type name Axes elsewhere
   type Axes = Tuple3[TensorTypeDenotation, TensorShapeDenotation, Shape]

   // Need this alias to not conflict with other Tensors
   // TODO: consider using TF-Java ndarray as backing instead of Scala Array here
   // S is overloaded
   opaque type Tensor[T <: Supported, +Ax <: Axes] = IO[Tuple2[Array[T], Ax]]

   type SparseTensor[T <: Supported, A <: Axes] = Tensor[T, A]

   type KeepOrReduceDims[
       S <: Shape,
       AxisIndices <: None.type | Indices,
       KeepDims <: (Boolean & Singleton)
   ] <: Shape = (KeepDims) match {
      case true  => ReduceKeepDims[S, AxisIndices]
      case false => Shape.Reduce[S, AxisIndices]
   }

   type KeepOrReduceDimDenotations[
       Td <: TensorShapeDenotation,
       AxisIndices <: None.type | Indices,
       KeepDims <: (Boolean & Singleton)
   ] <: TensorShapeDenotation = (KeepDims) match {
      case true  => Td
      case false => TensorShapeDenotation.Reduce[Td, AxisIndices]
   }

   type ReduceKeepDims[S <: Shape, AxisIndices <: None.type | Indices] <: Shape =
      AxisIndices match {
         case None.type => SNil
         case Indices   => ReduceKeepDimsLoop[S, AxisIndices, 0]
      }

   protected type ReduceKeepDimsLoop[
       ReplaceFrom <: Shape,
       ToReplace <: Indices,
       I <: Index
   ] <: Shape = ReplaceFrom match {
      case head #: tail =>
         Indices.Contains[ToReplace, I] match {
            case true  => 1 #: ReduceKeepDimsLoop[tail, Indices.RemoveValue[ToReplace, I], S[I]]
            case false => head #: ReduceKeepDimsLoop[tail, ToReplace, S[I]]
         }
      case SNil =>
         ToReplace match {
            case INil => SNil
         }
   }

   type AddGivenAxisSize[S <: Shape, S1 <: Shape, AxisIndices <: None.type | Indices] <: Shape =
      AxisIndices match {
         case None.type => SNil
         case Indices   => AddGivenAxisSizeLoop[S, S1, AxisIndices, 0]
      }

   protected type AddGivenAxisSizeLoop[
       First <: Shape,
       Second <: Shape,
       AxisIndex <: Indices,
       I <: Index
   ] <: Shape = First match {
      case head #: tail =>
         Indices.Contains[AxisIndex, I] match {
            case true =>
               Second match {
                  case secondHead #: secondTail =>
                     (head + secondHead) #:
                        AddGivenAxisSizeLoop[
                          tail,
                          secondTail,
                          Indices.RemoveValue[AxisIndex, I],
                          S[I]
                        ]
                  case SNil =>
                     AxisIndex match {
                        case INil => SNil
                        case _    => Nothing
                     }
               }
            case false =>
               Second match {
                  case secondHead #: secondTail =>
                     (head) #: AddGivenAxisSizeLoop[tail, secondTail, AxisIndex, S[I]]
               }
         }
      case SNil =>
         AxisIndex match {
            case INil => SNil
            case _    => Nothing
         }
   }

   type UnsqueezeShape[S <: Shape, AxisIndex <: None.type | Indices] <: Shape = AxisIndex match {
      case None.type => SNil
      case Indices   => UnsqueezeShapeLoop[S, AxisIndex, 0]
   }

   protected type UnsqueezeShapeLoop[
       ToUnsqueeze <: Shape,
       AxisIndex <: Indices,
       I <: Index
   ] <: Shape = ToUnsqueeze match {
      case head #: tail =>
         Indices.Contains[AxisIndex, I] match {
            case true =>
               1 #: head #: UnsqueezeShapeLoop[tail, Indices.RemoveValue[AxisIndex, I], S[I]]
            case false => head #: UnsqueezeShapeLoop[tail, AxisIndex, S[I]]
         }
      case SNil =>
         AxisIndex match {
            case INil => SNil
         }
   }

   type GatheredShape[
       S <: Shape,
       AxisIndex <: None.type | Indices,
       AxisIndices <: Indices
   ] <: Shape = AxisIndex match {
      case None.type => SNil
      case Indices   => GatheredShapeLoop[S, AxisIndex, 0, AxisIndices]
   }

   protected type GatheredShapeLoop[
       ToGather <: Shape,
       AxisIndex <: Indices,
       I <: Index,
       AxisIndices <: Indices
   ] <: Shape = ToGather match {
      case head #: tail =>
         Indices.Contains[AxisIndex, I] match {
            case true =>
               IndicesSize[AxisIndices] #:
                  GatheredShapeLoop[
                    tail,
                    Indices.RemoveValue[AxisIndex, I],
                    S[I],
                    AxisIndices
                  ]
            case false => head #: GatheredShapeLoop[tail, AxisIndex, S[I], AxisIndices]
         }
      case SNil =>
         AxisIndex match {
            case INil => SNil
            case _    => SNil
            // FIXME
         }
   }

   type IndicesSize[AxisIndices <: Indices] = IndicesSizeLoop[AxisIndices, 0]

   type IndicesSizeLoop[AxisIndices <: Indices, Acc <: Dimension] = AxisIndices match {
      case head ::: tail => IndicesSizeLoop[tail, S[Acc]]
      case INil          => Acc
   }

   type FlattenedShape[S <: Shape, AxisIndex <: None.type | Indices] <: Shape = AxisIndex match {
      case None.type => SNil
      case Indices   => FlattenedShapeLoop[S, AxisIndex, 0, 1]
   }

   protected type FlattenedShapeLoop[
       ToFlatten <: Shape,
       AxisIndex <: Indices,
       I <: Index,
       Acc <: Index
   ] <: Shape = ToFlatten match {
      case head #: tail =>
         Indices.Contains[AxisIndex, I] match {
            case true =>
               Acc #: FlattenedShapeLoop[tail, Indices.RemoveValue[AxisIndex, I], S[I], head]
            case false => FlattenedShapeLoop[tail, AxisIndex, S[I], head * Acc]
         }
      case SNil =>
         AxisIndex match {
            case INil => Acc #: SNil
            case _    => Nothing
         }
   }

   type SlicedShape[
       AxisIndicesStarts <: None.type | Indices,
       AxisIndicesEnds <: None.type | Indices
   ] <: Shape = AxisIndicesStarts match {
      case None.type => SNil
      case Indices =>
         AxisIndicesEnds match {
            case None.type => SNil
            case Indices   => SlicedShapeLoop[AxisIndicesStarts, AxisIndicesEnds]
         }
   }

   protected type SlicedShapeLoop[Starts <: Indices, Ends <: Indices] <: Shape = Starts match {
      case head ::: tail =>
         Ends match {
            case endsHead ::: endsTail => (endsHead - head) #: SlicedShapeLoop[tail, endsTail]
            case INil                  => SNil
         }
      case INil =>
         Ends match {
            case INil => SNil
         }
   }

   type PaddedShape[
       PadFrom <: Shape,
       AxisBefore <: None.type | Shape,
       AxisAfter <: None.type | Shape
   ] <: Shape = AxisBefore match {
      case None.type => PadFrom
      case Shape =>
         AxisAfter match {
            case None.type => PadFrom
            case Shape =>
               Reverse[PaddedShapeLoop[Reverse[PadFrom], Reverse[AxisBefore], Reverse[AxisAfter]]]
         }
   }

   protected type PaddedShapeLoop[PadFrom <: Shape, Before <: Shape, After <: Shape] <: Shape =
      Before match {
         case head #: tail =>
            After match {
               case afterHead #: afterTail =>
                  PadFrom match {
                     case padFromHead #: padFromTail =>
                        (head + padFromHead + afterHead) #:
                           PaddedShapeLoop[
                             padFromTail,
                             tail,
                             afterTail
                           ]
                     case SNil => SNil
                  }
               case SNil => SNil
            }
         case SNil =>
            After match {
               case SNil =>
                  PadFrom match {
                     case padFromHead #: padFromTail =>
                        padFromHead #: PaddedShapeLoop[padFromTail, SNil, SNil]
                     case SNil => SNil
                  }
            }
      }

   type TiledShape[TileFrom <: Shape, AxisRepeats <: None.type | Indices] <: Shape =
      AxisRepeats match {
         case None.type => SNil
         case Indices   => TiledShapeLoop[TileFrom, AxisRepeats]
      }

   protected type TiledShapeLoop[TileFrom <: Shape, Repeats <: Indices] <: Shape = Repeats match {
      case head ::: tail =>
         TileFrom match {
            case tileFromHead #: tileFromTail =>
               (head * tileFromHead) #: TiledShapeLoop[tileFromTail, tail]
            case SNil => SNil
         }
      case INil => SNil
   }

   type PoolShape[From <: Shape, KernelShape <: None.type | Shape] <: Shape = KernelShape match {
      case None.type => SNil
      case Shape     => Reverse[PoolShapeLoop[Reverse[From], Reverse[KernelShape]]]
   }

   protected type PoolShapeLoop[From <: Shape, KernelShape <: Shape] <: Shape = KernelShape match {
      case head #: tail =>
         From match {
            case fromHead #: fromTail => ((fromHead - head + 1)) #: PoolShapeLoop[fromTail, tail]
            case SNil                 => SNil
         }
      case SNil => From
   }

   // TODO: shape dimension values should be longs, not ints, but dotty compiletime ops only support ints
   // TODO: random nd access
   // TODO: Benchmark Array[T] vs ArraySeq[T] vs IArray[T]

   object Tensor {
      extension [
          T <: Supported,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](tens: Tensor[T, Tuple3[Tt, Td, S]]) def data = tens.map(_._1)

      extension [
          T <: Supported,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](tens: Tensor[T, Tuple3[Tt, Td, S]])
         def shape: IO[Array[Int]] = tens.map(_._2._3.toSeq.toArray)

      def tensorRequires[
          T <: Supported,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](tens: Tensor[T, Tuple3[Tt, Td, S]]): Tensor[T, Tuple3[Tt, Td, S]] = {
         // require(tens._2._2.toSeq.size == tens.shape.size) //We allow empty denotations
         tens.map(x =>
            require(
              x._1.size == x._2._3.toSeq.toArray.foldLeft(1)(_ * _)
            ) // This shouldn't fail at runtime, if so shape constraints need fixing
            x
         )
      }
      def apply[
          T <: Supported: scala.reflect.ClassTag,
          Tt <: TensorTypeDenotation,
          TD <: TensorShapeDenotation
      ](element: T, tt: Tt, td: TD): Tensor[T, Tuple3[Tt, TD, 1 #: SNil]] = tensorRequires(
        IO.pure { (Array[T](element), (tt, td, 1 #: SNil)) }
      )

      def apply[
          T <: Supported,
          Tt <: TensorTypeDenotation,
          TD <: TensorShapeDenotation,
          S <: Shape
      ](arr: Array[T], tt0: Tt, td0: TD, d0: S): Tensor[T, Tuple3[Tt, TD, S]] = tensorRequires(
        IO.pure { (arr, (tt0, td0, d0)) }
      )

      def apply[T <: Supported: scala.reflect.ClassTag](
          element: T
      ): Tensor[T, Tuple3["", org.emergentorder.compiletime.TSNil, 1 #: SNil]] = tensorRequires(
        IO.pure { (Array(element), ("", TSNil, 1 #: SNil)) }
      )

      def apply[T <: Supported, TD <: TensorShapeDenotation, S <: Shape](
          arr: Array[T],
          td0: TD,
          d0: S
      ): Tensor[T, Tuple3["", TD, S]] = tensorRequires(IO.pure { (arr, ("", td0, d0)) })

      def apply[T <: Supported, S <: Shape](
          arr: Array[T],
          d0: S
      ): Tensor[T, Tuple3["", "" ##: TSNil, S]] = tensorRequires(IO.pure {
         (arr, ("", "" ##: TSNil, d0))
      })

   }
}
