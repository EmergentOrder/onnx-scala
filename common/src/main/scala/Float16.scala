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

import java.lang.Integer.numberOfLeadingZeros
import java.lang.Math.pow
import java.lang.{Float => JFloat}
import scala.language.implicitConversions

/** Float16 represents 16-bit floating-point values.
  *
  * This type does not actually support arithmetic directly. The expected use case is to convert to
  * Float to perform any actual arithmetic, then convert back to a Float16 if needed.
  *
  * Binary representation:
  *
  * sign (1 bit) \| \| exponent (5 bits)
  * |  |
  * |:-|
  * |  |
  * mantissa (10 bits) \| | | x xxxxx xxxxxxxxxx
  *
  * Value interpretation (in order of precedence, with _ wild):
  *
  * 0 00000 0000000000 (positive) zero 1 00000 0000000000 negative zero _ 00000 __________ subnormal
  * number _ 11111 0000000000 +/- infinity _ 11111 __________ not-a-number _ _____ __________ normal
  * number
  *
  * An exponent of all 1s signals a sentinel (NaN or infinity), and all 0s signals a subnormal
  * number. So the working "real" range of exponents we can express is [-14, +15].
  *
  * For non-zero exponents, the mantissa has an implied leading 1 bit, so 10 bits of data provide 11
  * bits of precision for normal numbers.
  *
  * For normal numbers:
  *
  * x = (1 - sign*2) * 2^exponent * (1 + mantissa/1024)
  *
  * For subnormal numbers, the implied leading 1 bit is absent. Thus, subnormal numbers have the
  * same exponent as the smallest normal numbers, but without an implied 1 bit.
  *
  * So for subnormal numbers:
  *
  * x = (1 - sign*2) * 2^(-14) * (mantissa/1024)
  */
class Float16(val raw: Short) extends AnyVal { lhs =>

   def isNaN: Boolean  = (raw & 0x7fff) > 0x7c00
   def nonNaN: Boolean = (raw & 0x7fff) <= 0x7c00

   /** Returns if this is a zero value (positive or negative).
     */
   def isZero: Boolean = (raw & 0x7fff) == 0

   def nonZero: Boolean = (raw & 0x7fff) != 0

   def isPositiveZero: Boolean = raw == 0x0000
   def isNegativeZero: Boolean =
      // raw == 0x8000 but negated since 0x8000 is an integer.
      // comparing to the raw is clearer
      raw == Float16.NegativeZero.raw

   def isInfinite: Boolean         = (raw & 0x7fff) == 0x7c00
   def isPositiveInfinity: Boolean = raw == 0x7c00
   def isNegativeInfinity: Boolean =
      // raw == 0xfc00 but negated since 0xfc00 is an integer.
      // comparing to the raw is clearer
      (raw == Float16.NegativeInfinity.raw)

   def isSubnormal: Boolean = (raw & 0x7c00) == 0

   /** Whether this Float16 value is finite or not.
     *
     * For the purposes of this method, infinities and NaNs are considered non-finite. For those
     * values it returns false and for all other values it returns true.
     */
   def isFinite: Boolean = (raw & 0x7c00) != 0x7c00

   /** Return the sign of a Float16 value as a Float.
     *
     * There are five possible return values:
     *
     * * NaN: the value is Float16.NaN (and has no sign) * -1F: the value is a non-zero negative
     * number * -0F: the value is Float16.NegativeZero * 0F: the value is Float16.Zero * 1F: the
     * value is a non-zero positive number
     *
     * PositiveInfinity and NegativeInfinity return their expected signs.
     */
   def signum: Float =
      if (raw == -0x8000) -0f
      else if (raw == 0x0000) 0f
      else if (isNaN) Float.NaN
      else 1f - ((raw >>> 14) & 2)

   /** Reverse the sign of this Float16 value.
     *
     * This just involves toggling the sign bit with XOR.
     *
     * -Float16.NaN has no meaningful effect. -Float16.Zero returns Float16.NegativeZero.
     */
   def unary_- : Float16 =
      new Float16((raw ^ 0x8000).toShort)

   def +(rhs: Float16): Float16 =
      Float16.fromFloat(lhs.toFloat + rhs.toFloat)
   def -(rhs: Float16): Float16 =
      Float16.fromFloat(lhs.toFloat - rhs.toFloat)
   def *(rhs: Float16): Float16 =
      Float16.fromFloat(lhs.toFloat * rhs.toFloat)
   def /(rhs: Float16): Float16 =
      Float16.fromFloat(lhs.toFloat / rhs.toFloat)
   def **(rhs: Int): Float16 =
      Float16.fromFloat(pow(lhs.toFloat, rhs).toFloat)

   def compare(rhs: Float16): Int =
      if (lhs.raw == rhs.raw) 0
      else {
         val le = lhs.raw & 0x7c00
         val re = rhs.raw & 0x7c00

         if (le == 0x7c00) {
            // lhs is inf or nan
            if (re == 0x7c00) {
               // rhs is inf or nan
               val lm = lhs.raw & 0x03ff
               val rm = rhs.raw & 0x03ff
               if (lm != 0) {
                  // lhs is nan
                  if (rm != 0) 0 else 1
               } else {
                  // lhs is +/- inf
                  if (rm != 0) -1
                  else (rhs.raw & 0x8000) - (lhs.raw & 0x8000)
               }
            } else {
               if (lhs == Float16.NegativeInfinity) -1 else 1
            }
         } else if (re == 0x7c00) {
            // rhs is inf or nan, lhs is finite
            if (rhs == Float16.NegativeInfinity) 1 else -1
         } else {
            val ls = lhs.raw & 0x8000
            val rs = rhs.raw & 0x8000
            if (ls != rs) {
               (rs - ls) // if rs > ls, then rhs is negative and lhs is positive, so return +
            } else {
               val n = (1 - (ls >>> 14)) // 0x8000 -> -1, 0x0000 -> +1
               if (le != re) {
                  (le - re) * n
               } else {
                  val lm = lhs.raw & 0x03ff
                  val rm = rhs.raw & 0x03ff
                  (lm - rm) * n
               }
            }
         }
      }

   def <(rhs: Float16): Boolean = {
      if (lhs.raw == rhs.raw || lhs.isNaN || rhs.isNaN) return false
      if (lhs.isZero && rhs.isZero) return false
      val ls = lhs.raw & 0x8000
      val rs = rhs.raw & 0x8000
      if (ls < rs) return false
      if (ls > rs) return true
      val le = lhs.raw & 0x7c00
      val re = rhs.raw & 0x7c00
      if (le < re) return ls == 0
      if (le > re) return ls != 0
      val lm = lhs.raw & 0x03ff
      val rm = rhs.raw & 0x03ff
      if (ls == 0) lm < rm else rm < lm
   }

   def <=(rhs: Float16): Boolean = {
      if (lhs.isNaN || rhs.isNaN) return false
      if (lhs.raw == rhs.raw || lhs.isZero && rhs.isZero) return true
      val ls = lhs.raw & 0x8000
      val rs = rhs.raw & 0x8000
      if (ls < rs) return false
      if (ls > rs) return true
      val le = lhs.raw & 0x7c00
      val re = rhs.raw & 0x7c00
      if (le < re) return ls == 0
      if (le > re) return ls != 0
      val lm = lhs.raw & 0x03ff
      val rm = rhs.raw & 0x03ff
      if (ls == 0) lm < rm else rm < lm
   }

   def >(rhs: Float16): Boolean =
      !(lhs.isNaN || rhs.isNaN || lhs <= rhs)

   def >=(rhs: Float16): Boolean =
      !(lhs.isNaN || rhs.isNaN || lhs < rhs)

   def ==(rhs: Float16): Boolean =
      if (lhs.isNaN || rhs.isNaN) false
      else if (lhs.isZero && rhs.isZero) true
      else lhs.raw == rhs.raw

   def !=(rhs: Float16): Boolean =
      !(lhs == rhs)

   /** Convert this Float16 value to the nearest Float.
     *
     * Non-finite values and zero values will be mapped to the corresponding Float value.
     *
     * All other finite values will be handled depending on whether they are normal or subnormal.
     * The relevant formulas are:
     *
     * * normal: (sign*2-1) * 2^(exponent-15) * (1 + mantissa/1024) * subnormal: (sign*2-1) * 2^-14
     * * (mantissa/1024)
     *
     * Given any (x: Float16), Float16.fromFloat(x.toFloat) = x
     *
     * The reverse is not necessarily true, since there are many Float values which are not
     * precisely representable as Float16 values.
     */
   def toFloat: Float = {
      val s = raw & 0x8000
      val e = (raw >>> 10) & 0x1f // exponent
      val m = (raw & 0x03ff)      // mantissa
      if (e == 0) {
         // either zero or a subnormal number
         if (m == 0) {
            if (s == 0) 0f else -0f
         } else {
            // a 10-bit mantissa always has 22 leading zeros
            val shifts = numberOfLeadingZeros(m) - 21 // between 1-10 shifts

            // +127 is used to bias to 32-bit, -14 is used to unbias from
            // 16-bit, so our net is +113.
            val e32 = e + 113 - shifts

            // we are going from a 10-bit mantissa to a 23-bit mantissa,
            // so in addition to shifts we need 13 extra shifts. we also
            // have to mask the leading one bit.
            val m32 = (m << (shifts + 13)) & 0x7fffff

            val bits32 = (s << 16) | (e32 << 23) | m32
            JFloat.intBitsToFloat(bits32)
         }
      } else if (e != 31) {
         // normal number
         // a normal float is
         // 1 bit of sign, 8 bits of exponent biased by -126, 23 bits of mantissa
         // 127 - 15 = 112, which is the bias adjustment
         val bits32 = (s << 16) | ((e + 112) << 23) | (m << 13)
         JFloat.intBitsToFloat(bits32)
      } else if (m != 0) {
         Float.NaN
      } else if (s == 0) {
         Float.PositiveInfinity
      } else {
         Float.NegativeInfinity
      }
   }

   def toDouble: Double =
      toFloat.toDouble

   /** String representation of this Float16 value.
     */
   override def toString: String =
      toFloat.toString
}

object Float16 {
   // interesting Float16 constants
   // with the exception of NaN, values go from smallest to largest
   val NaN = new Float16(0x7c01.toShort)

   val NegativeInfinity = new Float16(0xfc00.toShort)
   val PositiveInfinity = new Float16(0x7c00.toShort)

   val MinValue = new Float16(0xfbff.toShort)
   val MaxValue = new Float16(0x7bff.toShort)

   val MinusOne = new Float16(0xbc00.toShort)
   val One      = new Float16(0x3c00.toShort)

   val MaxNegativeNormal = new Float16(0x8400.toShort)
   val MinPositiveNormal = new Float16(0x0400.toShort)

   val MaxNegative = new Float16(0x8001.toShort)
   val MinPositive = new Float16(0x0001.toShort)

   val NegativeZero = new Float16(0x8000.toShort)
   val Zero         = new Float16(0x0000.toShort)

   /** Implement left bit-shift with rounding.
     *
     * val shifts = ? val mask = (1 << shifts) - 1 val n = (? & mask) val res = round(n, shifts)
     * assert(res <= (mask + 1))
     */
   def round(m: Int, shifts: Int): Int = {
      val mid    = 1 << (shifts - 1)
      val mask   = (1 << shifts) - 1
      val mshift = m >> shifts
      val masked = m & mask
      val cmp    = masked - mid
      // we are losing more than 1/2
      if (cmp > 0) mshift + 1
      // we are losing < 1/2
      else if (cmp < 0) mshift
      else {
         // we are losing exactly 1/2
         // we round to the nearest even
         // 2.5 => 2, 3.5 => 4, 4.5 => 4
         // -2.5 => -2, -3.5 => -4, -4.5 => -4
         val isOdd = (mshift & 1) != 0
         if (isOdd) mshift + 1
         else mshift
      }
   }

   /** Create a Float16 value from a Float.
     *
     * This value is guaranteed to be the closest possible Float16 value. However, because there are
     * many more possible Float values, rounding will occur, and very large or very small values
     * will end up as infinities.
     *
     * Given any (x: Float16), Float16.fromFloat(x.toFloat) = x
     *
     * The reverse is not necessarily true, since there are many Float values which are not
     * precisely representable as Float16 values.
     */
   def fromFloat(n: Float): Float16 = {
      val nbits = JFloat.floatToRawIntBits(n)
      // 32 bits has 1 sign bit, 8 exponent bits, 23 mantissa bits
      val s = ((nbits >>> 16) & 0x8000)
      val e = (nbits >>> 23) & 0xff
      val m = (nbits & 0x7fffff)

      if (e == 0) {
         // subnormal, all 0 for float16
         new Float16(s.toShort)
      } else if (e != 0xff) { // e < 255
         val ereal = e - 127 // [127, -126]
         // for 16 bits, we have 5 bits of exponent
         // which are [15, -14], and we bias by adding 15
         //
         // we ebias16 = ereal + 15, if that is 0
         //
         if (ereal > 15) {
            // we can't fit in the new exponent, so either +/- inf
            new Float16((s | 0x7c00).toShort)
         } else if (ereal < -25) {
            // 2^(-25) * mant = 2*(-14) * ((1 + mant)/2^11)
            // but (1 + mant) >> 11 == 0
            // but we may need to round up or down to the smallest
            // subnormal numbers
            new Float16(s.toShort)
         }
         // past here ereal [-24, 15] = [-24, -15] | [-14, 15]
         else if (ereal >= -14) {
            // this is a regular normal 16 bit number
            val newm = round(m, 13)
            new Float16((s | (((ereal + 15) << 10) + newm)).toShort)
         } else {
            // ereal [-24, -15] which are all subnormal
            // need to write
            // 2^(ereal) * (1 + m) =
            // 2^(-14) * (1 + m)/2^n
            // note we can do 10 or less shifts, but 10 would reach -24
            // -n0 - 14 = ereal
            // n0 = -14 - ereal
            // n = n0 + 13
            // n = -1 - ereal
            val n    = -1 - ereal
            val newm = round(0x800000 | m, n)
            new Float16((s | newm).toShort)
         }
      } else if (m != 0) {
         Float16.NaN
      } else {
         // +/- infinity
         new Float16((s | 0x7c00).toShort)
      }
   }

   def fromDouble(x: Double): Float16 =
      fromFloat(x.toFloat)

   implicit val orderingForFloat16: Ordering[Float16] =
      new Ordering[Float16] {
         def compare(l: Float16, r: Float16): Int = l.compare(r)
      }

   // if either argument is NaN, return NaN. this matches java.lang.Float.min
   def min(x: Float16, y: Float16): Float16 =
      if (x.isNaN || y.isNaN) Float16.NaN
      else if (x <= y) x
      else y

   // if either argument is NaN, return NaN. this matches java.lang.Float.max
   def max(x: Float16, y: Float16): Float16 =
      if (x.isNaN || y.isNaN) Float16.NaN
      else if (x >= y) x
      else y
}
