/*
 * Copyright 2019 The Agate Authors
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//Source: https://github.com/stripe/agate/blob/master/core/src/main/scala/com/stripe/agate/tensor/Float16.scala

package org.emergentorder.onnx

import java.lang.{Float => JFloat}
import java.lang.Math.pow
import scala.language.implicitConversions

/** BFloat16 represents 16-bit floating-point values.
  *
  * This type does not actually support arithmetic directly. The expected use case is to convert to
  * Float to perform any actual arithmetic, then convert back to a BFloat16 if needed.
  *
  * Binary representation:
  *
  * sign (1 bit) \| \| exponent (8 bits)
  * |  |
  * |:-|
  * |  |
  * mantissa (7 bits) \| | | x xxxxxxxx xxxxxxx
  *
  * Value interpretation (in order of precedence, with _ wild):
  *
  * 0 00000000 0000000 (positive) zero 1 00000000 0000000 negative zero _ 00000000 _______ subnormal
  * number _ 11111111 0000000 +/- infinity _ 11111111 _______ not-a-number _ ________ _______ normal
  * number
  *
  * An exponent of all 1s signals a sentinel (NaN or infinity), and all 0s signals a subnormal
  * number. So the working "real" range of exponents we can express is [-126, +127].
  *
  * For non-zero exponents, the mantissa has an implied leading 1 bit, so 7 bits of data provide 8
  * bits of precision for normal numbers.
  *
  * For normal numbers:
  *
  * x = (1 - sign*2) * 2^exponent * (1 + mantissa/128)
  *
  * For subnormal numbers, the implied leading 1 bit is absent. Thus, subnormal numbers have the
  * same exponent as the smallest normal numbers, but without an implied 1 bit.
  *
  * So for subnormal numbers:
  *
  * x = (1 - sign*2) * 2^(-127) * (mantissa/128)
  */
class BFloat16(val raw: Short) extends AnyVal { lhs =>

   def isNaN: Boolean  = (raw & 0x7fff) > 0x7f80
   def nonNaN: Boolean = (raw & 0x7fff) <= 0x7f80

   /** Returns if this is a zero value (positive or negative).
     */
   def isZero: Boolean = (raw & 0x7fff) == 0

   def nonZero: Boolean = (raw & 0x7fff) != 0

   def isPositiveZero: Boolean = raw == 0x0000
   def isNegativeZero: Boolean =
      // raw == 0x8000 but negated since 0x8000 is an integer.
      // comparing to the raw is clearer
      raw == BFloat16.NegativeZero.raw

   def isInfinite: Boolean         = (raw & 0x7fff) == 0x7f80
   def isPositiveInfinity: Boolean = raw == 0x7f80
   def isNegativeInfinity: Boolean =
      // raw == 0xff80 but negated since 0xff80 is an integer.
      // comparing to the raw is clearer
      (raw == BFloat16.NegativeInfinity.raw)

   def isSubnormal: Boolean = (raw & 0x7f80) == 0

   /** Whether this BFloat16 value is finite or not.
     *
     * For the purposes of this method, infinities and NaNs are considered non-finite. For those
     * values it returns false and for all other values it returns true.
     */
   def isFinite: Boolean = (raw & 0x7f80) != 0x7f80

   /** Return the sign of a BFloat16 value as a Float.
     *
     * There are five possible return values:
     *
     * * NaN: the value is BFloat16.NaN (and has no sign) * -1F: the value is a non-zero negative
     * number * -0F: the value is BFloat16.NegativeZero * 0F: the value is BFloat16.Zero * 1F: the
     * value is a non-zero positive number
     *
     * PositiveInfinity and NegativeInfinity return their expected signs.
     */
   def signum: Float =
      java.lang.Math.signum(toFloat)

   /** Reverse the sign of this BFloat16 value.
     *
     * This just involves toggling the sign bit with XOR.
     *
     * -BFloat16.NaN has no meaningful effect.
     * -BFloat16.Zero returns BFloat16.NegativeZero.
     */
   def unary_- : BFloat16 =
      new BFloat16((raw ^ 0x8000).toShort)

   def +(rhs: BFloat16): BFloat16 =
      BFloat16.fromFloat(lhs.toFloat + rhs.toFloat)
   def -(rhs: BFloat16): BFloat16 =
      BFloat16.fromFloat(lhs.toFloat - rhs.toFloat)
   def *(rhs: BFloat16): BFloat16 =
      BFloat16.fromFloat(lhs.toFloat * rhs.toFloat)
   def /(rhs: BFloat16): BFloat16 =
      BFloat16.fromFloat(lhs.toFloat / rhs.toFloat)
   def **(rhs: Int): BFloat16 =
      BFloat16.fromFloat(pow(lhs.toFloat, rhs).toFloat)

   def compare(rhs: BFloat16): Int =
      java.lang.Float.compare(lhs.toFloat, rhs.toFloat)

   def <(rhs: BFloat16): Boolean =
      lhs.toFloat < rhs.toFloat

   def <=(rhs: BFloat16): Boolean =
      lhs.toFloat <= rhs.toFloat

   def >(rhs: BFloat16): Boolean =
      lhs.toFloat > rhs.toFloat

   def >=(rhs: BFloat16): Boolean =
      lhs.toFloat >= rhs.toFloat

   def ==(rhs: BFloat16): Boolean =
      lhs.toFloat == rhs.toFloat

   /** Convert this BFloat16 value to the nearest Float.
     *
     * Unlike Float16, since BFloat16 has the same size exponents as Float32 it means that all we
     * have to do is add some extra zeros to the mantissa.
     */
   def toFloat: Float =
      JFloat.intBitsToFloat((raw & 0xffff) << 16)

   def toDouble: Double =
      toFloat.toDouble

   /** String representation of this BFloat16 value.
     */
   override def toString: String =
      toFloat.toString
}

object BFloat16 {
   // interesting BFloat16 constants
   // with the exception of NaN, values go from smallest to largest
   val NaN = new BFloat16(0x7f81.toShort)

   val NegativeInfinity = new BFloat16(0xff80.toShort)
   val PositiveInfinity = new BFloat16(0x7f80.toShort)

   val MinValue = new BFloat16(0xff7f.toShort)
   val MaxValue = new BFloat16(0x7f7f.toShort)

   val MinusOne = new BFloat16(0xbf80.toShort)
   val One      = new BFloat16(0x3f80.toShort)

   val MaxNegativeNormal = new BFloat16(0x8080.toShort)
   val MinPositiveNormal = new BFloat16(0x0080.toShort)

   val MaxNegative = new BFloat16(0x8001.toShort)
   val MinPositive = new BFloat16(0x0001.toShort)

   val NegativeZero = new BFloat16(0x8000.toShort)
   val Zero         = new BFloat16(0x0000.toShort)

   /** Create a BFloat16 value from a Float.
     *
     * This value is guaranteed to be the closest possible BFloat16 value. However, because there
     * are many more possible Float values, rounding will occur, and very large or very small values
     * will end up as infinities.
     *
     * Given any (x: BFloat16), BFloat16.fromFloat(x.toFloat) = x
     *
     * The reverse is not necessarily true, since there are many Float values which are not
     * precisely representable as BFloat16 values.
     */
   def fromFloat(n: Float): BFloat16 = {
      val nbits = JFloat.floatToRawIntBits(n)
      // 32 bits has 1 sign bit, 8 exponent bits, 23 mantissa bits
      val s = (nbits >>> 16) & 0x8000
      val e = (nbits >>> 16) & 0x7f80
      val m = (nbits & 0x7fffff)

      if (e != 0x7f80) {
         // handle normal and subnormal numbers (i.e. not sentinels).
         //
         // m1 will be in [0, 128]; 0 means we rounded down to 0, 128
         // means we rounded up, and will have a zero mantissa left (plus
         // one exponent bit).
         //
         // in any of these cases it turns out m1 has the correct
         // exponent + mantissa bits set. what luck!
         val m1 = Float16.round(m, 16)
         val e1 = e + m1
         new BFloat16((s | e1).toShort)
      } else {
         // handle sentinels
         //
         // if m != 0, we need to be sure to return a NaN. otherwise,
         // truncating will preserve the correctly-signed infinity value.
         if (m != 0) BFloat16.NaN else new BFloat16((nbits >>> 16).toShort)
      }
   }

   def fromDouble(x: Double): BFloat16 =
      fromFloat(x.toFloat)

   implicit val orderingForBFloat16: Ordering[BFloat16] =
      new Ordering[BFloat16] {
         def compare(l: BFloat16, r: BFloat16) = l.compare(r)
      }

   // if either argument is NaN, return NaN. this matches java.lang.Float.min
   def min(x: BFloat16, y: BFloat16): BFloat16 =
      if (x.isNaN || y.isNaN) BFloat16.NaN
      else if (x <= y) x
      else y

   // if either argument is NaN, return NaN. this matches java.lang.Float.max
   def max(x: BFloat16, y: BFloat16): BFloat16 =
      if (x.isNaN || y.isNaN) BFloat16.NaN
      else if (x >= y) x
      else y
}
