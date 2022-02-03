//Author: Maxime Kjaer, taken from tf-dotty, Alexander Merritt

package org.emergentorder.compiletime

import scala.compiletime.ops.int.{S, +, <, <=, *}
import scala.compiletime.ops.boolean.&&
import io.kjaer.compiletime.Index
import io.kjaer.compiletime.Indices
import io.kjaer.compiletime.INil

type DimensionDenotation = String & Singleton

sealed trait TensorShapeDenotation extends Product with Serializable {
   import TensorShapeDenotation._

   /** Prepend the head to this */
   def ##:[H <: DimensionDenotation, This >: this.type <: TensorShapeDenotation](
       head: H
   ): H ##: This =
      org.emergentorder.compiletime.##:(head, this)

   /** Concat with another shape * */
   def ++(that: TensorShapeDenotation): this.type Concat that.type =
      TensorShapeDenotation.concat(this, that)

   /** Reverse the dimension list */
   def reverse: Reverse[this.type] = TensorShapeDenotation.reverse(this)

   def toSeq: Seq[String] = this match {
      case TSNil         => Nil
      case head ##: tail => head +: tail.toSeq
   }
}

final case class ##:[+H <: DimensionDenotation, +T <: TensorShapeDenotation](head: H, tail: T)
    extends TensorShapeDenotation {
   override def toString = head match {
      case _       => s"$head ##: $tail"
   }
}

sealed trait TSNil extends TensorShapeDenotation
case object TSNil  extends TSNil

object TensorShapeDenotation {
   def scalar: TSNil                                              = TSNil
   def vector(length: DimensionDenotation): length.type ##: TSNil = length ##: TSNil
   def matrix(
       rows: DimensionDenotation,
       columns: DimensionDenotation
   ): rows.type ##: columns.type ##: TSNil = rows ##: columns ##: TSNil

   def fromSeq(seq: Seq[String]): TensorShapeDenotation = seq match {
      case Nil          => TSNil
      case head +: tail => head ##: TensorShapeDenotation.fromSeq(tail)
   }

   type Concat[X <: TensorShapeDenotation, Y <: TensorShapeDenotation] <: TensorShapeDenotation =
      X match {
         case TSNil         => Y
         case head ##: tail => head ##: Concat[tail, Y]
      }

   def concat[X <: TensorShapeDenotation, Y <: TensorShapeDenotation](x: X, y: Y): Concat[X, Y] =
      x match {
         case _: TSNil        => y
         case cons: ##:[x, y] => cons.head ##: concat(cons.tail, y)
      }

   type Reverse[X <: TensorShapeDenotation] <: TensorShapeDenotation = X match {
      case TSNil         => TSNil
      case head ##: tail => Concat[Reverse[tail], head ##: TSNil]
   }

   def reverse[X <: TensorShapeDenotation](x: X): Reverse[X] = x match {
      case _: TSNil              => TSNil
      case cons: ##:[head, tail] => concat(reverse(cons.tail), cons.head ##: TSNil)
   }

   type IsEmpty[X <: TensorShapeDenotation] <: Boolean = X match {
      case TSNil   => true
      case _ ##: _ => false
   }

   type Head[X <: TensorShapeDenotation] <: DimensionDenotation = X match {
      case head ##: _ => head
   }

   type Tail[X <: TensorShapeDenotation] <: TensorShapeDenotation = X match {
      case _ ##: tail => tail
   }

   /** Represents reduction along axes, as defined in TensorFlow:
     *
     *   - None means reduce along all axes
     *   - List of indices contain which indices in the shape to remove
     *   - Empty list of indices means reduce along nothing
     *
     * @tparam S
     *   Shape to reduce
     * @tparam Axes
     *   List of indices to reduce along. `one` if reduction should be done along all axes. `SNil`
     *   if no reduction should be done.
     */
   type Reduce[S <: TensorShapeDenotation, Axes <: None.type | Indices] <: TensorShapeDenotation =
      Axes match {
         case None.type => TSNil
         case Indices   => ReduceLoop[S, Axes, 0]
      }

   /** Remove indices from a shape
     *
     * @tparam RemoveFrom
     *   Shape to remove from
     * @tparam ToRemove
     *   Indices to remove from `RemoveFrom`
     * @tparam I
     *   Current index (in the original shape)
     */
   protected type ReduceLoop[
       RemoveFrom <: TensorShapeDenotation,
       ToRemove <: Indices,
       I <: Index
   ] <: TensorShapeDenotation = RemoveFrom match {
      case head ##: tail =>
         Indices.Contains[ToRemove, I] match {
            case true  => ReduceLoop[tail, Indices.RemoveValue[ToRemove, I], S[I]]
            case false => head ##: ReduceLoop[tail, ToRemove, S[I]]
         }
      case TSNil =>
         ToRemove match {
            case INil => TSNil
            //     case head :: tail => Error[
            //         "The following indices are out of bounds: " + Indices.ToString[ToRemove]
            //     ]
         }
   }

   /** Apply a function to elements of a TensorShapeDenotation. Type-level representation of `def
     * map(f: (A) => A): List[A]`
     *
     * @tparam X
     *   TensorShapeDenotation to map over
     * @tparam F
     *   Function taking an value of the TensorShapeDenotation, returning another value
     */
   type Map[
       X <: TensorShapeDenotation,
       F[_ <: DimensionDenotation] <: DimensionDenotation
   ] <: TensorShapeDenotation = X match {
      case TSNil         => TSNil
      case head ##: tail => F[head] ##: Map[tail, F]
   }

   /** Apply a folding function to the elements of a TensorShapeDenotation Type-level representation
     * of `def foldLeft[B](z: B)(op: (B, A) => B): B`
     *
     * @tparam B
     *   Return type of the operation
     * @tparam X
     *   TensorShapeDenotation to fold over
     * @tparam Z
     *   Zero element
     * @tparam F
     *   Function taking an accumulator of type B, and an element of type String, returning B
     */
   type FoldLeft[B, X <: TensorShapeDenotation, Z <: B, F[_ <: B, _ <: String] <: B] <: B =
      X match {
         case TSNil         => Z
         case head ##: tail => FoldLeft[B, tail, F[Z, head], F]
      }
}
