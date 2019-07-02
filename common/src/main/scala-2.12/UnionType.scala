package org.emergentorder.union

object UnionType {

  trait inv[-A] {}

  sealed trait OrR {
    type L <: OrR
    type R
    type invIntersect
    type intersect
  }

  sealed class TypeOr[A <: OrR, B] extends OrR {
    type L = A
    type R = B

    type intersect    = (L#intersect with R)
    type invIntersect = (L#invIntersect with inv[R])
    type check[X]     = invIntersect <:< inv[X]
  }

  object UNil extends OrR {
    type intersect    = Any
    type invIntersect = inv[Nothing]
  }
  type UNil = UNil.type

}
