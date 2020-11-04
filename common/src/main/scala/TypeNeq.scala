package org.emergentorder

//Annotation doesn't seem to work in dotty
@annotation.implicitNotFound(msg = "Cannot prove that ${A} =!= ${B}.")
trait =!=[A, B]
object =!= {
  class Impl[A, B]
  object Impl {
    implicit def neq[A, B] : A Impl B = null
    implicit def neqAmbig1[A] : A Impl A = null
    implicit def neqAmbig2[A] : A Impl A = null
  }

  implicit def foo[A, B](implicit e: A Impl B): A =!= B = null
}
