//Author: Maxime Kjaer, taken from tf-dotty
package io.kjaer.compiletime

/**
  * Type-class used to materialize the singleton type of an [[Indices]].
  * 
  * @see ShapeOf
  */
final class IndicesOf[T <: Indices](val value: T)

object IndicesOf {
  given indicesOfINilType: IndicesOf[INil.type] = IndicesOf(INil)
  given indicesOfINil: IndicesOf[INil] = IndicesOf(INil)
  given indicesOfCons[H <: Index, T <: Indices](using head: ValueOf[H], tail: IndicesOf[T]): IndicesOf[H ::: T] =
    IndicesOf(head.value ::: tail.value)
}

inline def indicesOf[I <: Indices](using i: IndicesOf[I]): I = i.value
