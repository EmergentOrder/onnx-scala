//Author: Maxime Kjaer, taken from tf-dotty, Alexander Merritt
package org.emergentorder.compiletime

import scala.compiletime.S
import scala.compiletime.ops.int.{<, <=, *}
import scala.compiletime.ops.boolean.&&

type DimensionDenotation = String & Singleton

sealed trait TensorDenotation extends Product with Serializable {
  import TensorDenotation._

  /** Prepend the head to this */
  def ##:[H <: DimensionDenotation, This >: this.type <: TensorDenotation](head: H): H ##: This =
    org.emergentorder.compiletime.##:(head, this)

  /** Concat with another tensor denotation **/
  def ++(that: TensorDenotation): this.type Concat that.type = {
    val res: TensorDenotation = this match {
      case SSNil => that
      case x ##: xs => x ##: (xs ++ that)
    }
    res.asInstanceOf[this.type Concat that.type]
  }

  def reverse: Reverse[this.type] = {
    val res: TensorDenotation = this match {
      case SSNil => SSNil
      case x ##: xs => xs.reverse ++ (x ##: SSNil)
    }
    res.asInstanceOf[Reverse[this.type]]
  }

  def toSeq: Seq[String] = this match {
    case SSNil => Nil
    case head ##: tail => head +: tail.toSeq
  }
}

final case class ##:[+H <: DimensionDenotation, +T <: TensorDenotation](head: H, tail: T) extends TensorDenotation {
  override def toString = head match {
    case _ ##: _ => s"($head) ##: $tail"
    case _      => s"$head ##: $tail"
  }
}

sealed trait SSNil extends TensorDenotation
case object SSNil extends SSNil

object TensorDenotation {
  def fromSeq(seq: Seq[String]): TensorDenotation = seq match {
    case Nil => SSNil
    case head +: tail => head ##: TensorDenotation.fromSeq(tail)
  }

  /**
   * Apply a function to elements of a TensorDenotation.
   * Type-level representation of  `def map(f: (A) => A): List[A]`
   *
   * @tparam X TensorDenotation to map over
   * @tparam F Function taking an value of the TensorDenotation, returning another value
   */
  type Map[X <: TensorDenotation, F[_ <: DimensionDenotation] <: DimensionDenotation] <: TensorDenotation = X match {
    case SSNil => SSNil
    case head ##: tail => F[head] ##: Map[tail, F]
  }

  /**
   * Apply a folding function to the elements of a TensorDenotation
   * Type-level representation of `def foldLeft[B](z: B)(op: (B, A) => B): B`
   *
   * @tparam B Return type of the operation
   * @tparam X TensorDenotation to fold over
   * @tparam Z Zero element
   * @tparam F Function taking an accumulator of type B, and an element of type String, returning B
   */
  type FoldLeft[B, X <: TensorDenotation, Z <: B, F[_ <: B, _ <: String] <: B] <: B = X match {
    case SSNil => Z
    case head ##: tail => FoldLeft[B, tail, F[Z, head], F]
  }

  type Concat[X <: TensorDenotation, Y <: TensorDenotation] <: TensorDenotation = X match {
    case SSNil => Y
    case head ##: tail => head ##: Concat[tail, Y]
  }

  type Reverse[X <: TensorDenotation] <: TensorDenotation = X match {
    case SSNil => SSNil
    case head ##: tail => Concat[Reverse[tail], head ##: SSNil]
  }

  type IsEmpty[X <: TensorDenotation] <: Boolean = X match {
    case SSNil => true
    case _ ##: _ => false
  }

  type Head[X <: TensorDenotation] <: DimensionDenotation = X match {
    case head ##: _ => head
  }

  type Tail[X <: TensorDenotation] <: TensorDenotation = X match {
    case _ ##: tail => tail
  } 
}

final class TensorDenotationOf[T <: TensorDenotation](val value: T)

object TensorDenotationOf {
  given tensorDenotationOfSSNilType as TensorDenotationOf[SSNil.type] = TensorDenotationOf(SSNil)
  given tensorDenotationOfSSNil as TensorDenotationOf[SSNil] = TensorDenotationOf(SSNil)
  given tensorDenotationOfCons[H <: DimensionDenotation, T <: TensorDenotation](using head: ValueOf[H], tail: TensorDenotationOf[T]) as TensorDenotationOf[H ##: T] =
    TensorDenotationOf(head.value ##: tail.value)
}

inline def tensorDenotationOf[S <: TensorDenotation](using s: TensorDenotationOf[S]): S = s.value
