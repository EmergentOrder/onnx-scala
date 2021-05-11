//Author: Maxime Kjaer, taken from tf-dotty, Alexander Merritt

package org.emergentorder.compiletime

/** Type-class used to materialize the singleton type of a [[TensorShapeDenotation]].
  *
  * This is useful to implicitly convert a type-level representation of a [[TensorShapeDenotation]]
  * to a term representing the same [[TensorShapeDenotation]], for instance by using the [[shapeOf]]
  * method:
  *
  * {{{
  * shapeOf[TSNil.type]      //=> TSNil
  * shapeOf[1 ##: 2 ##: TSNil] //=> 1 ##: 2 ##: TSNil
  * }}}
  */
final class TensorShapeDenotationOf[T <: TensorShapeDenotation](val value: T)

object TensorShapeDenotationOf {
   given tensorShapeDenotationOfTSNilType: TensorShapeDenotationOf[TSNil.type] =
      TensorShapeDenotationOf(TSNil)
   given tensorShapeDenotationOfTSNil: TensorShapeDenotationOf[TSNil] = TensorShapeDenotationOf(
     TSNil
   )
   given tensorShapeDenotationOfCons[H <: DimensionDenotation, T <: TensorShapeDenotation](using
       head: ValueOf[H],
       tail: TensorShapeDenotationOf[T]
   ): TensorShapeDenotationOf[H ##: T] =
      TensorShapeDenotationOf(head.value ##: tail.value)
}

inline def tensorShapeDenotationOf[Td <: TensorShapeDenotation](using
    td: TensorShapeDenotationOf[Td]
): Td = td.value
