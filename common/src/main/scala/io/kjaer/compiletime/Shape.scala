//Author: Maxime Kjaer, taken from tf-dotty
package io.kjaer.compiletime

import scala.compiletime.S
import scala.compiletime.ops.int.{<, <=, *}
import scala.compiletime.ops.boolean.&&

type Dimension = Int & Singleton

sealed trait Shape extends Product with Serializable {
  import Shape._

  /** Prepend the head to this */
  def #:[H <: Dimension, This >: this.type <: Shape](head: H): H #: This =
    io.kjaer.compiletime.#:(head, this)

  /** Concat with another shape **/
  def ++(that: Shape): this.type Concat that.type = {
    val res: Shape = this match {
      case SNil => that
      case x #: xs => x #: (xs ++ that)
    }
    res.asInstanceOf[this.type Concat that.type]
  }

  def reverse: Reverse[this.type] = {
    val res: Shape = this match {
      case SNil => SNil
      case x #: xs => xs.reverse ++ (x #: SNil)
    }
    res.asInstanceOf[Reverse[this.type]]
  }

  def rank: Size[this.type] = {
    val res: Int = this match {
      case SNil => 0
      case head #: tail => 1 + tail.rank
    }
    res.asInstanceOf[Size[this.type]]
  }

  def numElements: NumElements[this.type] = {
    val res: Int = this match {
      case SNil => 1
      case head #: tail => head * tail.numElements
    }
    res.asInstanceOf[NumElements[this.type]]
  }

  def toSeq: Seq[Int] = this match {
    case SNil => Nil
    case head #: tail => head +: tail.toSeq
  }
}

final case class #:[+H <: Dimension, +T <: Shape](head: H, tail: T) extends Shape {
  override def toString = head match {
    case _ #: _ => s"($head) #: $tail"
    case _      => s"$head #: $tail"
  }
}

sealed trait SNil extends Shape
case object SNil extends SNil

object Shape {
  def fromSeq(seq: Seq[Int]): Shape = seq match {
    case Nil => SNil
    case head +: tail => head #: Shape.fromSeq(tail)
  }

  /**
   * Apply a function to elements of a Shape.
   * Type-level representation of  `def map(f: (A) => A): List[A]`
   *
   * @tparam X Shape to map over
   * @tparam F Function taking an value of the Shape, returning another value
   */
  type Map[X <: Shape, F[_ <: Dimension] <: Dimension] <: Shape = X match {
    case SNil => SNil
    case head #: tail => F[head] #: Map[tail, F]
  }

  /**
   * Apply a folding function to the elements of a Shape
   * Type-level representation of `def foldLeft[B](z: B)(op: (B, A) => B): B`
   *
   * @tparam B Return type of the operation
   * @tparam X Shape to fold over
   * @tparam Z Zero element
   * @tparam F Function taking an accumulator of type B, and an element of type Int, returning B
   */
  type FoldLeft[B, X <: Shape, Z <: B, F[_ <: B, _ <: Int] <: B] <: B = X match {
    case SNil => Z
    case head #: tail => FoldLeft[B, tail, F[Z, head], F]
  }

  type Size[X <: Shape] <: Int = X match {
    case SNil => 0
    case head #: tail => S[Size[tail]]
  }

  type NumElements[X <: Shape] <: Int = X match {
    case SNil => 1
    case head #: tail => head * NumElements[tail]
  }

  type Concat[X <: Shape, Y <: Shape] <: Shape = X match {
    case SNil => Y
    case head #: tail => head #: Concat[tail, Y]
  }

  type Reverse[X <: Shape] <: Shape = X match {
    case SNil => SNil
    case head #: tail => Concat[Reverse[tail], head #: SNil]
  }

  type IsEmpty[X <: Shape] <: Boolean = X match {
    case SNil => true
    case _ #: _ => false
  }

  type Head[X <: Shape] <: Dimension = X match {
    case head #: _ => head
  }

  type Tail[X <: Shape] <: Shape = X match {
    case _ #: tail => tail
  }

  /**
   * Represents reduction along axes, as defined in TensorFlow:
   *
   *   - None means reduce along all axes
   *   - List of indices contain which indices in the shape to remove
   *   - Empty list of indices means reduce along nothing
   *
   * @tparam S           Shape to reduce
   * @tparam Axes        List of indices to reduce along.
   *                     `one` if reduction should be done along all axes.
   *                     `SNil` if no reduction should be done.
   */
  type Reduce[S <: Shape, Axes <: None.type | Indices] <: Shape = Axes match {
    case None.type => SNil
    case Indices => ReduceLoop[S, Axes, 0]
  }

  /**
   * Remove indices from a shape
   *
   * @tparam RemoveFrom   Shape to remove from
   * @tparam ToRemove     Indices to remove from `RemoveFrom`
   * @tparam I            Current index (in the original shape)
   */
  protected type ReduceLoop[RemoveFrom <: Shape, ToRemove <: Indices, I <: Index] <: Shape = RemoveFrom match {
    case head #: tail => Indices.Contains[ToRemove, I] match {
      case true => ReduceLoop[tail, Indices.RemoveValue[ToRemove, I], S[I]]
      case false => head #: ReduceLoop[tail, ToRemove, S[I]]
    }
    case SNil => ToRemove match {
      case INil => SNil
      //     case head :: tail => Error[
      //         "The following indices are out of bounds: " + Indices.ToString[ToRemove]
      //     ]
    }
  }

  /** Returns whether index `I` is within bounds of `S` */
  type WithinBounds[I <: Index, S <: Shape] = (0 <= I && I < Size[S])

  /**
   * Remove the element at index `I` in `RemoveFrom`.
   *
   * @tparam RemoveFrom  Shape to remove from
   * @tparam I           Index to remove
   */
  type RemoveIndex[RemoveFrom <: Shape, I <: Index] <: Shape = WithinBounds[I, RemoveFrom] match {
    case true => RemoveIndexLoop[RemoveFrom, I, 0]
    // case false => Error[
    //     "Index " + int.ToString[I] +
    //     " is out of bounds for shape of rank " + int.ToString[Size[RemoveFrom]]
    // ]
  }

  /**
   * Removes element at index `I` from `RemoveFrom`. Assumes `I` is within bounds.
   *
   * @tparam RemoveFrom  Shape to remove index `I` from
   * @tparam I           Index to remove from `RemoveFrom`
   * @tparam Current     Current index in the loop
   */
  protected type RemoveIndexLoop[RemoveFrom <: Shape, I <: Index, Current <: Index] <: Shape = RemoveFrom match {
    case head #: tail => Current match {
      case I => tail
      case _ => head #: RemoveIndexLoop[tail, I, S[Current]]
    }
  }

  def scalar: SNil = SNil
  def vector(length: Dimension): length.type #: SNil = length #: SNil
  def matrix(rows: Dimension, columns: Dimension): rows.type #: columns.type #: SNil = rows #: columns #: SNil
}

final class ShapeOf[T <: Shape](val value: T)

object ShapeOf {
  given shapeOfSNilType as ShapeOf[SNil.type] = ShapeOf(SNil)
  given shapeOfSNil as ShapeOf[SNil] = ShapeOf(SNil)
  given shapeOfCons[H <: Dimension, T <: Shape](using head: ValueOf[H], tail: ShapeOf[T]) as ShapeOf[H #: T] =
    ShapeOf(head.value #: tail.value)
}

inline def shapeOf[S <: Shape](using s: ShapeOf[S]): S = s.value
