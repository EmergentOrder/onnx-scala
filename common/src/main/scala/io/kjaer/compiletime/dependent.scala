//Author: Maxime Kjaer, taken from tf-dotty
package org.emergentorder.io.kjaer.compiletime

import scala.compiletime.ops.long.*

// Extensions on longs that allow scala.compiletime.ops to be dependently typed
extension [X <: Long, Y <: Long](x: Long) {
   infix def add(y: Y): X + Y = (x + y).asInstanceOf[X + Y]
   infix def sub(y: Y): X - Y = (x - y).asInstanceOf[X - Y]
   infix def mul(y: Y): X * Y = (x * y).asInstanceOf[X * Y]
   infix def lt(y: Y): X < Y  = (x < y).asInstanceOf[X < Y]
   infix def le(y: Y): X <= Y = (x <= y).asInstanceOf[X <= Y]
}
