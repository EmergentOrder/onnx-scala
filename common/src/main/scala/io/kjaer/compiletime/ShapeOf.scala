//Author: Maxime Kjaer, taken from tf-dotty
package io.kjaer.compiletime

/** 
  * Type-class used to materialize the singleton type of a [[Shape]].
  * 
  * This is useful to implicitly convert a type-level representation of a
  * [[Shape]] to a term representing the same [[Shape]], for instance by using
  * the [[shapeOf]] method:
  * 
  * {{{
  * shapeOf[SNil.type]      //=> SNil
  * shapeOf[1 #: 2 #: SNil] //=> 1 #: 2 #: SNil
  * }}}
  */
final class ShapeOf[T <: Shape](val value: T)

object ShapeOf {
  given shapeOfSNilType as ShapeOf[SNil.type] = ShapeOf(SNil)
  given shapeOfSNil as ShapeOf[SNil] = ShapeOf(SNil)
  given shapeOfCons[H <: Dimension, T <: Shape](using head: ValueOf[H], tail: ShapeOf[T]) as ShapeOf[H #: T] =
    ShapeOf(head.value #: tail.value)
}

inline def shapeOf[S <: Shape](using s: ShapeOf[S]): S = s.value
