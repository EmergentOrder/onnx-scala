//Author: Maxime Kjaer, taken from tf-dotty
package io.kjaer.compiletime

import scala.compiletime.ops.string.+
import scala.compiletime.ops.int

type Index = Int & Singleton

sealed trait Indices {
  def ::[H <: Index, This >: this.type <: Indices](head: H): H :: This =
    io.kjaer.compiletime.::(head, this)

  def indices: Set[Int] = this match {
    case head :: tail => tail.indices + head
    case INil => Set.empty
  }
}

final case class ::[H <: Index, T <: Indices](head: H, tail: T) extends Indices {
  override def toString = s"$head :: $tail"
}

sealed trait INil extends Indices
case object INil extends INil

object Indices {
  type ToString[X <: Indices] <: String = X match {
    case INil => "INil"
    case head :: tail => int.ToString[head] + " :: " + ToString[tail]
  }

  type Contains[Haystack <: Indices, Needle <: Index] <: Boolean = Haystack match {
    case head :: tail => head match {
      case Needle => true
      case _ => Contains[tail, Needle]
    }
    case INil => false
  }

  type RemoveValue[RemoveFrom <: Indices, Value <: Index] <: Indices = RemoveFrom match {
    case INil => INil
    case head :: tail => head match {
      case Value => RemoveValue[tail, Value]
      case _ => head :: RemoveValue[tail, Value]
    }
  }
}
