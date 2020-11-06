package org.emergentorder

import scala.language.higherKinds
import scala.{specialized => sp}
//import java.util.Map
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import spire.implicits._
import spire.algebra.Field
import org.emergentorder.onnx.Tensors._
import org.emergentorder.union.|

package object onnx {

  //TODO: Push constraints on binary ops using =!= to here
  //TODO: match types to avoid instance of?
  //TODO:Typed axis semantics, JS support
  //Note: shape constraints will disallow broadcasting
  //In progress: Add shapes, constraints (at first only to NDScala-exposed ops)
  //TODO: add ORT contrib ops
  //TODO: Remove requirement to be Numeric for ops with non-numeric outputs / inputs
  //TODO: Encode node names as types
  //TODO: fix encoding of type constraints, use Tensor as part of definition of types
  //TODO: Use  monadless(except dead, find followup) / scala-async (with -Xasync?) / dotty-cps-async to replace for comprehensions

  sealed trait Operator {
    def callOp[T, Ax <: Axes](
        name: String,
        opName: String,
        inputs: Seq[_],
        //    outName: String,
        attrs: Map[String, Any]
    ): Tensor[T, Ax]
  }

  abstract class Model(onnxBytes: Array[Byte]) extends Operator {
    def fullModel[
        T
    , Ax <: Axes](
        inputs: Seq[_]
    ): Tensor[T, Ax]
  }

  trait Graph
  trait DataSource {
    def getParams[T, Ax <: Axes](name: String): Tensor[T, Ax]
  }
  trait AbsV6 extends Operator {
    def AbsV6[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, X: Tensor[T, Ax]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Abs", allInputs, map))
    }
  }

  trait AbsV1 extends Operator {
    def AbsV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Abs", allInputs, map))
    }
  }

  trait AcosV7 extends Operator {
    def AcosV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Acos", allInputs, map))
    }
  }

  trait AcoshV9 extends Operator {
    def AcoshV9[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Acosh", allInputs, map))
    }
  }

  //Not yet supported, training has yet to GA
  /*
  trait AdagradV1 extends Operator {
    def AdagradV1[
        @sp T1 <: Float | Double: Numeric,
        @sp T2 <: Long: Numeric,
        @sp T3 <: Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes](
        name: String,
        decay_factor: Option[(Float)] = None,
        epsilon: Option[(Float)] = None,
        norm_coefficient: Option[(Float)] = None,
        R: Tensor[T1,Ax],
        T: Tensor[T2, Bx],
        inputs: Seq[Tensor[T3, Cx]]
    ): Tensor[T3, Dx] = {
      val map: Map[String, Any] = Map(
        "decay_factor"     -> decay_factor,
        "epsilon"          -> epsilon,
        "norm_coefficient" -> norm_coefficient
      )
      val allInputs =
        Seq(R, T) ++ (Tuple.fromArray(inputs.toArray).asInstanceOf[Tuple])
      (callOp(name, "Adagrad", allInputs, map))
    }
  }
*/
  trait AddV7 extends Operator {
    def AddV7[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Add", allInputs, map))
    }
  }

  trait AddV6 extends Operator {
    def AddV6[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Add", allInputs, map))
    }
  }

  trait AddV1 extends Operator {
    def AddV1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "broadcast" -> broadcast, "consumed_inputs" -> consumed_inputs)
      val allInputs = Seq(A,B)
      (callOp(name, "Add", allInputs, map))
    }
  }

  trait AndV7 extends Operator {
    def AndV7[@sp T <: Boolean, @sp T1 <: Boolean, Ax <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1,Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "And", allInputs, map))
    }
  }

  trait AndV1 extends Operator {
    def AndV1[@sp T <: Boolean, @sp T1 <: Boolean, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "And", allInputs, map))
    }
  }

  //tf-dotty reduce eligible
  trait ArgMaxV12 extends Operator {
    def ArgMaxV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        select_last_index: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[Long, Bx] = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "keepdims" -> keepdims, "select_last_index" -> select_last_index)
      val allInputs = Seq(data)
      (callOp(name, "ArgMax", allInputs, map))
    }
  }

  trait ArgMaxV11 extends Operator {
    def ArgMaxV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[Long, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ArgMax", allInputs, map))
    }
  }

  trait ArgMaxV1 extends Operator {
    def ArgMaxV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[Long, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ArgMax", allInputs, map))
    }
  }

  //tf-dotty reduce eligible
  trait ArgMinV12 extends Operator {
    def ArgMinV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        select_last_index: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[Long, Bx] = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "keepdims" -> keepdims, "select_last_index" -> select_last_index)
      val allInputs = Seq(data)
      (callOp(name, "ArgMin", allInputs, map))
    }
  }

  trait ArgMinV11 extends Operator {
    def ArgMinV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[Long, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ArgMin", allInputs, map))
    }
  }

  trait ArgMinV1 extends Operator {
    def ArgMinV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[Long, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ArgMin", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait ArrayFeatureExtractorV1 extends Operator {
    def ArrayFeatureExtractorV1[@sp T <: Float | Double | Long | Int | String: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        X: Tensor[T, Ax],
        Y: Tensor[Long, Bx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X,Y)
      (callOp(name, "ArrayFeatureExtractor", allInputs, map))
    }
  }
*/
  trait AsinV7 extends Operator {
    def AsinV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Asin", allInputs, map))
    }
  }

  trait AsinhV9 extends Operator {
    def AsinhV9[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Asinh", allInputs, map))
    }
  }

  trait AtanV7 extends Operator {
    def AtanV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Atan", allInputs, map))
    }
  }

  trait AtanhV9 extends Operator {
    def AtanhV9[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Atanh", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait AveragePoolV11 extends Operator {
    def AveragePoolV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"          -> auto_pad,
        "ceil_mode"         -> ceil_mode,
        "count_include_pad" -> count_include_pad,
        "kernel_shape"      -> kernel_shape,
        "pads"              -> pads,
        "strides"           -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "AveragePool", allInputs, map))
    }
  }
*/
  trait AveragePoolV10 extends Operator {
    def AveragePoolV10[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"          -> auto_pad,
        "ceil_mode"         -> ceil_mode,
        "count_include_pad" -> count_include_pad,
        "kernel_shape"      -> kernel_shape,
        "pads"              -> pads,
        "strides"           -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "AveragePool", allInputs, map))
    }
  }

  trait AveragePoolV7 extends Operator {
    def AveragePoolV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"          -> auto_pad,
        "count_include_pad" -> count_include_pad,
        "kernel_shape"      -> kernel_shape,
        "pads"              -> pads,
        "strides"           -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "AveragePool", allInputs, map))
    }
  }

  trait AveragePoolV1 extends Operator {
    def AveragePoolV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "AveragePool", allInputs, map))
    }
  }

  trait BatchNormalizationV9 extends Operator {
    def BatchNormalizationV9[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        epsilon: Option[(Float)] = None,
        momentum: Option[(Float)] = None,
        X: Tensor[T, Ax],
        scale: Tensor[T, Bx],
        B: Tensor[T, Bx],
        mean: Tensor[T, Bx],
        someVar: Tensor[T, Bx]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("epsilon" -> epsilon, "momentum" -> momentum)
      val allInputs             = Seq(X, scale, B, mean, someVar)
      (callOp(name, "BatchNormalization", allInputs, map))
    }
  }

  trait BatchNormalizationV7 extends Operator {
    def BatchNormalizationV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        epsilon: Option[(Float)] = None,
        momentum: Option[(Float)] = None,
        spatial: Option[(Int)] = None,
        X: Tensor[T, Ax],
        scale: Tensor[T, Bx],
        B: Tensor[T, Bx],
        mean: Tensor[T, Bx],
        someVar: Tensor[T, Bx]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("epsilon" -> epsilon, "momentum" -> momentum, "spatial" -> spatial)
      val allInputs = Seq(X, scale, B, mean, someVar)
      (callOp(name, "BatchNormalization", allInputs, map))
    }
  }

  trait BatchNormalizationV6 extends Operator {
    def BatchNormalizationV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        epsilon: Option[(Float)] = None,
        is_test: Option[(Int)] = None,
        momentum: Option[(Float)] = None,
        spatial: Option[(Int)] = None,
        X: Tensor[T, Ax],
        scale: Tensor[T, Bx],
        B: Tensor[T, Bx],
        mean: Tensor[T, Bx],
        someVar: Tensor[T, Bx]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map(
        "epsilon"  -> epsilon,
        "is_test"  -> is_test,
        "momentum" -> momentum,
        "spatial"  -> spatial
      )
      val allInputs = Seq(X, scale, B, mean, someVar)
      (callOp(name, "BatchNormalization", allInputs, map))
    }
  }

  trait BatchNormalizationV1 extends Operator {
    def BatchNormalizationV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        consumed_inputs: (Array[Int]),
        epsilon: Option[(Float)] = None,
        is_test: Option[(Int)] = None,
        momentum: Option[(Float)] = None,
        spatial: Option[(Int)] = None,
        X: Tensor[T, Ax],
        scale: Tensor[T, Bx],
        B: Tensor[T, Bx],
        mean: Tensor[T, Bx],
        someVar: Tensor[T, Bx]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map(
        "consumed_inputs" -> consumed_inputs,
        "epsilon"         -> epsilon,
        "is_test"         -> is_test,
        "momentum"        -> momentum,
        "spatial"         -> spatial
      )
      val allInputs = Seq(X, scale, B, mean, someVar)
      (callOp(name, "BatchNormalization", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait BinarizerV1 extends Operator {
    def BinarizerV1[@sp T <: Float | Double | Long | Int: Numeric, Ax <: Axes](
        name: String,
        threshold: Option[(Float)] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("threshold" -> threshold)
      val allInputs             = Seq(X)
      (callOp(name, "Binarizer", allInputs, map))
    }
  }
*/
  //Not supported, missing from ONNXJS
  trait BitShiftV11 extends Operator {
    def BitShiftV11[@sp T <: UByte | UShort | UInt | ULong: Numeric, Ax <: Axes](
        name: String,
        direction: (String),
        X: Tensor[T, Ax],
        Y: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("direction" -> direction)
      val allInputs             = Seq(X,Y)
      (callOp(name, "BitShift", allInputs, map))
    }
  }

  //Not supported, ONNX ML\
  /*
  trait CastMapV1 extends Operator {
    def CastMapV1[@sp T1 <: Map[Long, String] | Map[
      Long,
      Float
    ]: Numeric, @sp T2 <: String | Float | Long: Numeric, Ax <: Axes](
        name: String,
        cast_to: Option[(String)] = None,
        map_form: Option[(String)] = None,
        max_map: Option[(Int)] = None,
        X: T1
    ): Tensor[T2, Ax] = {
      val map: Map[String, Any] =
        Map("cast_to" -> cast_to, "map_form" -> map_form, "max_map" -> max_map)
      val allInputs = Seq(X)
      (callOp(name, "CastMap", allInputs, map))
    }
  }
*/
  trait CastV9 extends Operator {
    def CastV9[
        @sp T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String: Numeric,
        @sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String: Numeric
    , Ax <: Axes](name: String, to: (Int), input: Tensor[T1,Ax]): Tensor[T2, Ax] = {
      val map: Map[String, Any] = Map("to" -> to)
      val allInputs             = Seq(input)
      (callOp(name, "Cast", allInputs, map))
    }
  }

  trait CastV6 extends Operator {
    def CastV6[
        @sp T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String: Numeric,
        @sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String: Numeric
    , Ax <: Axes](name: String, to: (Int), input: Tensor[T1, Ax]): Tensor[T2, Ax] = {
      val map: Map[String, Any] = Map("to" -> to)
      val allInputs             = Seq(input)
      (callOp(name, "Cast", allInputs, map))
    }
  }

  trait CastV1 extends Operator {
    def CastV1[
        @sp T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String: Numeric,
        @sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String: Numeric
    , Ax <: Axes](name: String, to: (String), input: Tensor[T1, Ax]): Tensor[T2, Ax] = {
      val map: Map[String, Any] = Map("to" -> to)
      val allInputs             = Seq(input)
      (callOp(name, "Cast", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait CategoryMapperV1 extends Operator {
    def CategoryMapperV1[
        @sp T1 <: String | Long: Numeric,
        @sp T2 <: String | Long: Numeric
    , Ax <: Axes](
        name: String,
        cats_int64s: Option[(Array[Int])] = None,
        cats_strings: Option[(Array[String])] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        X: Tensor[T1, Ax]
    ): Tensor[T2, Ax] = {
      val map: Map[String, Any] = Map(
        "cats_int64s"    -> cats_int64s,
        "cats_strings"   -> cats_strings,
        "default_int64"  -> default_int64,
        "default_string" -> default_string
      )
      val allInputs = Seq(X)
      (callOp(name, "CategoryMapper", allInputs, map))
    }
  }
*/
  trait CeilV6 extends Operator {
    def CeilV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Ceil", allInputs, map))
    }
  }

  trait CeilV1 extends Operator {
    def CeilV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Ceil", allInputs, map))
    }
  }

  trait CeluV12 extends Operator {
    def CeluV12[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("alpha" -> alpha)
      val allInputs             = Seq(X)
      (callOp(name, "Celu", allInputs, map))
    }
  }

  //Not supported, not in ONNXJS
  /*
  trait ClipV12 extends Operator {
    def ClipV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Tt <: TensorTypeDenotation, Dd <: DimensionDenotation](
        name: String,
        input: Tensor[T, Ax],
        min: Option[Tensor[T, Scalar[Tt, Dd]]] = None,
        max: Option[Tensor[T, Scalar[Tt, Dd]]] = None
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input, min, max)
      (callOp(name, "Clip", allInputs, map))
    }
  }

  trait ClipV11 extends Operator {
    def ClipV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Tt <: TensorTypeDenotation, Dd <: DimensionDenotation](
        name: String,
        input: Tensor[T, Ax],
        min: Option[Tensor[T, Scalar[TensorTypeDenotation, Dd]]] = None,
        max: Option[Tensor[T, Scalar[TensorTypeDenotation, Dd]]] = None
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input, min, max)
      (callOp(name, "Clip", allInputs, map))
    }
  }
*/
  trait ClipV6 extends Operator {
    def ClipV6[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        max: Option[(Float)] = None,
        min: Option[(Float)] = None,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("max" -> max, "min" -> min)
      val allInputs             = Seq(input)
      (callOp(name, "Clip", allInputs, map))
    }
  }

  trait ClipV1 extends Operator {
    def ClipV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        max: Option[(Float)] = None,
        min: Option[(Float)] = None,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("consumed_inputs" -> consumed_inputs, "max" -> max, "min" -> min)
      val allInputs = Seq(input)
      (callOp(name, "Clip", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait CompressV11 extends Operator {
    def CompressV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, Ax],
        condition: Tensor[T1,Bx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input, condition)
      (callOp(name, "Compress", allInputs, map))
    }
  }

  trait CompressV9 extends Operator {
    def CompressV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, Ax],
        condition: Tensor[T1, Bx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input, condition)
      (callOp(name, "Compress", allInputs, map))
    }
  }
  */

  //Not supported, sequence op
  /*
  trait ConcatFromSequenceV11 extends Operator {
    def ConcatFromSequenceV11[@sp S <: Seq[Tensor[UByte, _]] | Seq[Tensor[UShort, _]] | Seq[
      Tensor[UInt, _]
    ] | Seq[
      Tensor[ULong, _]
    ] | Seq[Tensor[Byte, _]] | Seq[Tensor[Short, _]] | Seq[Tensor[Int, _]] | Seq[Tensor[Long, _]] | Seq[
      Tensor[Float16, _]
    ] | Seq[Tensor[Float,_]] | Seq[Tensor[Double, _]] | Seq[Tensor[String, _]] | Seq[Tensor[Boolean, _]] | Seq[
      Tensor[Complex[Float], _]
    ] | Seq[
      Tensor[Complex[Double], _]
    ]: Numeric, @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
      Float
    ] | Complex[Double]: Numeric, Ax <: Axes](
        name: String,
        axis: (Int),
        new_axis: Option[(Int)] = None,
        input_sequence: S
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "new_axis" -> new_axis)
      val allInputs             = Seq(input_sequence)
      (callOp(name, "ConcatFromSequence", allInputs, map))
    }
  }
*/

  //TODO: constraint 
  trait ConcatV11 extends Operator {
    def ConcatV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]
    , Ax <: Axes, Bx <: Axes](name: String, axis: (Int), inputs: Seq[Tensor[T, Ax]]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = inputs 
      (callOp(name, "Concat", allInputs, map))
    }
  }

  trait ConcatV4 extends Operator {
    def ConcatV4[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]
    , Ax <: Axes, Bx <: Axes](name: String, axis: (Int), inputs: Seq[Tensor[T, Ax]]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = inputs 
      (callOp(name, "Concat", allInputs, map))
    }
  }

  trait ConcatV1 extends Operator {
    def ConcatV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]
    , Ax <: Axes, Bx <: Axes](name: String, axis: Option[(Int)] = None, inputs: Seq[Tensor[T, Ax]]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = inputs 
      (callOp(name, "Concat", allInputs, map))
    }
  }

  trait ConstantOfShapeV9 extends Operator {
    def ConstantOfShapeV9[
        @sp T1 <: Long: Numeric,
        @sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](name: String, value: Option[(Tensor[T2, Ax])] = None, input: Tensor[T1, Bx]): Tensor[T2, Cx] = {
      val map: Map[String, Any] = Map("value" -> value)
      val allInputs             = Seq(input)
      (callOp(name, "ConstantOfShape", allInputs, map))
    }
  }

  trait ConstantV12 extends Operator {
    def ConstantV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        sparse_value: Option[(SparseTensor[T, Ax])] = None,
        value: Option[(Tensor[T, Bx])] = None,
        value_float: Option[(Float)] = None,
        value_floats: Option[(Array[Float])] = None,
        value_int: Option[(Int)] = None,
        value_ints: Option[(Array[Int])] = None,
        value_string: Option[(String)] = None,
        value_strings: Option[(Array[String])] = None
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map(
        "sparse_value"  -> sparse_value,
        "value"         -> value,
        "value_float"   -> value_float,
        "value_floats"  -> value_floats,
        "value_int"     -> value_int,
        "value_ints"    -> value_ints,
        "value_string"  -> value_string,
        "value_strings" -> value_strings
      )
      val allInputs = Seq()
      (callOp(name, "Constant", allInputs, map))
    }
  }

  trait ConstantV11 extends Operator {
    def ConstantV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        sparse_value: Option[(SparseTensor[T, Ax])] = None,
        value: Option[(Tensor[T, Bx])] = None
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map("sparse_value" -> sparse_value, "value" -> value)
      val allInputs             = Seq()
      (callOp(name, "Constant", allInputs, map))
    }
  }

  trait ConstantV9 extends Operator {
    def ConstantV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, value: (Tensor[T, Ax])): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("value" -> value)
      val allInputs             = Seq()
      (callOp(name, "Constant", allInputs, map))
    }
  }

  trait ConstantV1 extends Operator {
    def ConstantV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, value: (Tensor[T, Ax])): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("value" -> value)
      val allInputs             = Seq()
      (callOp(name, "Constant", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait ConvIntegerV10 extends Operator {
    def ConvIntegerV10[
        @sp T1 <: Byte | UByte: Numeric,
        @sp T2 <: Byte | UByte: Numeric,
        @sp T3 <: Int: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes, Ex <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        x: Tensor[T1, Ax],
        w: Tensor[T2, Bx],
        x_zero_point: Option[Tensor[T1, Cx]] = None,
        w_zero_point: Option[Tensor[T2, Dx]] = None
    ): Tensor[T3, Ex] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Seq(x, w, x_zero_point, w_zero_point)
      (callOp(name, "ConvInteger", allInputs, map))
    }
  }

  trait ConvTransposeV11 extends Operator {
    def ConvTransposeV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        output_padding: Option[(Array[Int])] = None,
        output_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax],
        W: Tensor[T, Bx],
        B: Option[Tensor[T, Cx]] = None
    ): Tensor[T, Dx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"       -> auto_pad,
        "dilations"      -> dilations,
        "group"          -> group,
        "kernel_shape"   -> kernel_shape,
        "output_padding" -> output_padding,
        "output_shape"   -> output_shape,
        "pads"           -> pads,
        "strides"        -> strides
      )
      val allInputs = Seq(X, W, B)
      (callOp(name, "ConvTranspose", allInputs, map))
    }
  }

  trait ConvTransposeV1 extends Operator {
    def ConvTransposeV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        output_padding: Option[(Array[Int])] = None,
        output_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax],
        W: Tensor[T, Bx],
        B: Option[Tensor[T, Cx]] = None
    ): Tensor[T, Dx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"       -> auto_pad,
        "dilations"      -> dilations,
        "group"          -> group,
        "kernel_shape"   -> kernel_shape,
        "output_padding" -> output_padding,
        "output_shape"   -> output_shape,
        "pads"           -> pads,
        "strides"        -> strides
      )
      val allInputs = Seq(X, W, B)
      (callOp(name, "ConvTranspose", allInputs, map))
    }
  }
*/
  //TODO: Constraints
  trait ConvV11 extends Operator {
    def ConvV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax],
        W: Tensor[T, Bx],
        B: Option[Tensor[T, Cx]] = None
    ): Tensor[T, Dx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Seq(X, W, B)
      (callOp(name, "Conv", allInputs, map))
    }
  }

  trait ConvV1 extends Operator {
    def ConvV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax],
        W: Tensor[T, Bx],
        B: Option[Tensor[T, Cx]] = None
    ): Tensor[T, Dx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Seq(X, W, B)
      (callOp(name, "Conv", allInputs, map))
    }
  }

  trait CosV7 extends Operator {
    def CosV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Cos", allInputs, map))
    }
  }

  trait CoshV9 extends Operator {
    def CoshV9[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Cosh", allInputs, map))
    }
  }

  //Not supported for now, missing from ONNXJS
  /*
  trait CumSumV11 extends Operator {
    def CumSumV11[
        @sp T <: UInt | ULong | Int | Long | Float | Double: Numeric,
        @sp T2 <: Int | Long: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        exclusive: Option[(Int)] = None,
        reverse: Option[(Int)] = None,
        x: Tensor[T, Ax],
        axis: Tensor[T2, Bx]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("exclusive" -> exclusive, "reverse" -> reverse)
      val allInputs             = Seq(x, axis)
      (callOp(name, "CumSum", allInputs, map))
    }
  }

  trait DepthToSpaceV11 extends Operator {
    def DepthToSpaceV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        blocksize: (Int),
        mode: Option[(String)] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("blocksize" -> blocksize, "mode" -> mode)
      val allInputs             = Seq(input)
      (callOp(name, "DepthToSpace", allInputs, map))
    }
  }

  trait DepthToSpaceV1 extends Operator {
    def DepthToSpaceV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, blocksize: (Int), input: Tensor[T, _]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("blocksize" -> blocksize)
      val allInputs             = Seq(input)
      (callOp(name, "DepthToSpace", allInputs, map))
    }
  }

  trait DequantizeLinearV10 extends Operator {
    def DequantizeLinearV10[@sp T <: Byte | UByte | Int: Numeric, Ax <: Axes](
        name: String,
        x: Tensor[T, _],
        x_scale: Tensor[Float,_],
        x_zero_point: Option[Tensor[T, _]] = None
    ): Tensor[Float,_] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(x, x_scale, x_zero_point)
      (callOp(name, "DequantizeLinear", allInputs, map))
    }
  }

  trait DetV11 extends Operator {
    def DetV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Det", allInputs, map))
    }
  }

  trait DictVectorizerV1 extends Operator {
    def DictVectorizerV1[@sp T1 <: Map[String, Long] | Map[Long, String] | Map[Long, Float] | Map[
      Long,
      Double
    ] | Map[String, Float] | Map[
      String,
      Double
    ]: Numeric, @sp T2 <: Long | Float | Double | String: Numeric, Ax <: Axes](
        name: String,
        int64_vocabulary: Option[(Array[Int])] = None,
        string_vocabulary: Option[(Array[String])] = None,
        X: T1
    ): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("int64_vocabulary" -> int64_vocabulary, "string_vocabulary" -> string_vocabulary)
      val allInputs = Seq(X)
      (callOp(name, "DictVectorizer", allInputs, map))
    }
  }
*/
  trait DivV7 extends Operator {
    def DivV7[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Div", allInputs, map))
    }
  }

  trait DivV6 extends Operator {
    def DivV6[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Div", allInputs, map))
    }
  }

  trait DivV1 extends Operator {
    def DivV1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "broadcast" -> broadcast, "consumed_inputs" -> consumed_inputs)
      val allInputs = Seq(A,B)
      (callOp(name, "Div", allInputs, map))
    }
  }

  trait DropoutV12 extends Operator {
    def DropoutV12[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Float16 | Float | Double | Boolean,
        @sp T2 <: Boolean
    , Ax <: Axes,Bx <: Axes](
        name: String,
        seed: Option[(Int)] = None,
        data: Tensor[T, Ax],
        ratio: Option[Tensor[T1,Bx]] = None
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("seed" -> seed)
      val allInputs             = Seq(data, ratio)
      (callOp(name, "Dropout", allInputs, map))
    }
  }

  trait DropoutV10 extends Operator {
    def DropoutV10[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Float16 | Float | Double | Boolean
    , Ax <: Axes](name: String, ratio: Option[(Float)] = None, data: Tensor[T, Ax]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("ratio" -> ratio)
      val allInputs             = Seq(data)
      (callOp(name, "Dropout", allInputs, map))
    }
  }

  trait DropoutV7 extends Operator {
    def DropoutV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        ratio: Option[(Float)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("ratio" -> ratio)
      val allInputs             = Seq(data)
      (callOp(name, "Dropout", allInputs, map))
    }
  }

  trait DropoutV6 extends Operator {
    def DropoutV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("is_test" -> is_test, "ratio" -> ratio)
      val allInputs             = Seq(data)
      (callOp(name, "Dropout", allInputs, map))
    }
  }

  trait DropoutV1 extends Operator {
    def DropoutV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("consumed_inputs" -> consumed_inputs, "is_test" -> is_test, "ratio" -> ratio)
      val allInputs = Seq(data)
      (callOp(name, "Dropout", allInputs, map))
    }
  }

  //Not supported for now, missing from ONNXJS
  /*
  trait DynamicQuantizeLinearV11 extends Operator {
    def DynamicQuantizeLinearV11[
        @sp T1 <: Float: Numeric,
        @sp T2 <: UByte: Numeric
    , Ax <: Axes](name: String, x: Tensor[T1,_]): Tensor[T2, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(x)
      (callOp(name, "DynamicQuantizeLinear", allInputs, map))
    }
  }

  trait EinsumV12 extends Operator {
    def EinsumV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, equation: (String), Inputs: Seq[Tensor[T, _]]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("equation" -> equation)
      val allInputs             = Tuple.fromArray(Inputs.toArray).asInstanceOf[Tuple]
      (callOp(name, "Einsum", allInputs, map))
    }
  }
*/
  trait EluV6 extends Operator {
    def EluV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("alpha" -> alpha)
      val allInputs             = Seq(X)
      (callOp(name, "Elu", allInputs, map))
    }
  }

  trait EluV1 extends Operator {
    def EluV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Elu", allInputs, map))
    }
  }

  trait EqualV11 extends Operator {
    def EqualV11[
        @sp T <: Boolean | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double,
        @sp T1 <: Boolean
    , Ax <: Axes](name: String, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Equal", allInputs, map))
    }
  }

  trait EqualV7 extends Operator {
    def EqualV7[
        @sp T <: Boolean | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](name: String, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Equal", allInputs, map))
    }
  }

  trait EqualV1 extends Operator {
    def EqualV1[
        @sp T <: Boolean | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1,Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Equal", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait ErfV9 extends Operator {
    def ErfV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, input: Tensor[T, _]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Erf", allInputs, map))
    }
  }
*/

  trait ExpV6 extends Operator {
    def ExpV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Exp", allInputs, map))
    }
  }

  trait ExpV1 extends Operator {
    def ExpV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(input)
      (callOp(name, "Exp", allInputs, map))
    }
  }

  trait ExpandV8 extends Operator {
    def ExpandV8[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](name: String, input: Tensor[T, Ax], shapeInput: Tensor[Long, Bx]): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input, shapeInput)
      (callOp(name, "Expand", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait EyeLikeV9 extends Operator {
    def EyeLikeV9[
        @sp T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean,
        @sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean
    , Ax <: Axes](
        name: String,
        dtype: Option[(Int)] = None,
        k: Option[(Int)] = None,
        input: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] = Map("dtype" -> dtype, "k" -> k)
      val allInputs             = Seq(input)
      (callOp(name, "EyeLike", allInputs, map))
    }
  }
*/
  //Not supported, ONNX ML
  /*
  trait FeatureVectorizerV1 extends Operator {
    def FeatureVectorizerV1[@sp T1 <: Int | Long | Float | Double: Numeric, Ax <: Axes](
        name: String,
        inputdimensions: Option[(Array[Int])] = None,
        X: Seq[Tensor[T1,_]]
    ): Tensor[Float,_] = {
      val map: Map[String, Any] = Map("inputdimensions" -> inputdimensions)
      val allInputs             = Tuple.fromArray(X.toArray).asInstanceOf[Tuple]
      (callOp(name, "FeatureVectorizer", allInputs, map))
    }
  }
*/

  trait FlattenV11 extends Operator {
    def FlattenV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, axis: Option[(Int)] = None, input: Tensor[T, Ax]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "Flatten", allInputs, map))
    }
  }

  trait FlattenV9 extends Operator {
    def FlattenV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, axis: Option[(Int)] = None, input: Tensor[T, Ax]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "Flatten", allInputs, map))
    }
  }

  trait FlattenV1 extends Operator {
    def FlattenV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, axis: Option[(Int)] = None, input: Tensor[T, Ax]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "Flatten", allInputs, map))
    }
  }

  trait FloorV6 extends Operator {
    def FloorV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Floor", allInputs, map))
    }
  }

  trait FloorV1 extends Operator {
    def FloorV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Floor", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait GRUV7 extends Operator {
    def GRUV7[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        linear_before_reset: Option[(Int)] = None,
        X: Tensor[T, _],
        W: Tensor[T, _],
        R: Tensor[T, _],
        B: Option[Tensor[T, _]] = None,
        sequence_lens: Option[Tensor[T1,_]] = None,
        initial_h: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha"    -> activation_alpha,
        "activation_beta"     -> activation_beta,
        "activations"         -> activations,
        "clip"                -> clip,
        "direction"           -> direction,
        "hidden_size"         -> hidden_size,
        "linear_before_reset" -> linear_before_reset
      )
      val allInputs = Seq(X, W, R, B, sequence_lens, initial_h)
      (callOp(name, "GRU", allInputs, map))
    }
  }

  trait GRUV3 extends Operator {
    def GRUV3[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        linear_before_reset: Option[(Int)] = None,
        output_sequence: Option[(Int)] = None,
        X: Tensor[T, _],
        W: Tensor[T, _],
        R: Tensor[T, _],
        B: Option[Tensor[T, _]] = None,
        sequence_lens: Option[Tensor[T1,_]] = None,
        initial_h: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha"    -> activation_alpha,
        "activation_beta"     -> activation_beta,
        "activations"         -> activations,
        "clip"                -> clip,
        "direction"           -> direction,
        "hidden_size"         -> hidden_size,
        "linear_before_reset" -> linear_before_reset,
        "output_sequence"     -> output_sequence
      )
      val allInputs = Seq(X, W, R, B, sequence_lens, initial_h)
      (callOp(name, "GRU", allInputs, map))
    }
  }

  trait GRUV1 extends Operator {
    def GRUV1[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        output_sequence: Option[(Int)] = None,
        X: Tensor[T, _],
        W: Tensor[T, _],
        R: Tensor[T, _],
        B: Option[Tensor[T, _]] = None,
        sequence_lens: Option[Tensor[T1,_]] = None,
        initial_h: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size,
        "output_sequence"  -> output_sequence
      )
      val allInputs = Seq(X, W, R, B, sequence_lens, initial_h)
      (callOp(name, "GRU", allInputs, map))
    }
  }

  trait GatherElementsV11 extends Operator {
    def GatherElementsV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        data: Tensor[T, _],
        indices: Tensor[Tind, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(data, indices)
      (callOp(name, "GatherElements", allInputs, map))
    }
  }

  trait GatherNDV12 extends Operator {
    def GatherNDV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        batch_dims: Option[(Int)] = None,
        data: Tensor[T, _],
        indices: Tensor[Long, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("batch_dims" -> batch_dims)
      val allInputs             = Seq(data, indices)
      (callOp(name, "GatherND", allInputs, map))
    }
  }

  trait GatherNDV11 extends Operator {
    def GatherNDV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, data: Tensor[T, _], indices: Tensor[Long, _]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(data, indices)
      (callOp(name, "GatherND", allInputs, map))
    }
  }
*/
  trait GatherV11 extends Operator {
    def GatherV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        data: Tensor[T, Ax],
        indices: Tensor[Tind, Bx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(data, indices)
      (callOp(name, "Gather", allInputs, map))
    }
  }

  trait GatherV1 extends Operator {
    def GatherV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        data: Tensor[T, Ax],
        indices: Tensor[Tind, Bx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(data, indices)
      (callOp(name, "Gather", allInputs, map))
    }
  }

  trait GemmV11 extends Operator {
    def GemmV11[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Bx],
        C: Option[Tensor[T, Cx]] = None
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "transA" -> transA, "transB" -> transB)
      val allInputs = Seq(A, B, C)
      (callOp(name, "Gemm", allInputs, map))
    }
  }

  trait GemmV9 extends Operator {
    def GemmV9[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Bx],
        C: Tensor[T, Cx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "transA" -> transA, "transB" -> transB)
      val allInputs = Seq(A, B, C)
      (callOp(name, "Gemm", allInputs, map))
    }
  }

  trait GemmV7 extends Operator {
    def GemmV7[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Bx],
        C: Tensor[T, Cx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "transA" -> transA, "transB" -> transB)
      val allInputs = Seq(A, B, C)
      (callOp(name, "Gemm", allInputs, map))
    }
  }

  trait GemmV6 extends Operator {
    def GemmV6[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        broadcast: Option[(Int)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Bx],
        C: Tensor[T, Cx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map(
        "alpha"     -> alpha,
        "beta"      -> beta,
        "broadcast" -> broadcast,
        "transA"    -> transA,
        "transB"    -> transB
      )
      val allInputs = Seq(A, B, C)
      (callOp(name, "Gemm", allInputs, map))
    }
  }

  trait GemmV1 extends Operator {
    def GemmV1[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        broadcast: Option[(Int)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Bx],
        C: Tensor[T, Cx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map(
        "alpha"     -> alpha,
        "beta"      -> beta,
        "broadcast" -> broadcast,
        "transA"    -> transA,
        "transB"    -> transB
      )
      val allInputs = Seq(A, B, C)
      (callOp(name, "Gemm", allInputs, map))
    }
  }

  trait GlobalAveragePoolV1 extends Operator {
    def GlobalAveragePoolV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "GlobalAveragePool", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait GlobalLpPoolV2 extends Operator {
    def GlobalLpPoolV2[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        p: Option[(Int)] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("p" -> p)
      val allInputs             = Seq(X)
      (callOp(name, "GlobalLpPool", allInputs, map))
    }
  }

  trait GlobalLpPoolV1 extends Operator {
    def GlobalLpPoolV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        p: Option[(Float)] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("p" -> p)
      val allInputs             = Seq(X)
      (callOp(name, "GlobalLpPool", allInputs, map))
    }
  }
*/

  trait GlobalMaxPoolV1 extends Operator {
    def GlobalMaxPoolV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "GlobalMaxPool", allInputs, map))
    }
  }

  //Not supported, training is not yet GA
  /*
  trait GradientV1 extends Operator {
    def GradientV1[
        @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T2 <: Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        xs: (Array[String]),
        y: (String),
        zs: Option[(Array[String])] = None,
        Inputs: Seq[Tensor[T1,_]]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] = Map("xs" -> xs, "y" -> y, "zs" -> zs)
      val allInputs             = Tuple.fromArray(Inputs.toArray).asInstanceOf[Tuple]
      (callOp(name, "Gradient", allInputs, map))
    }
  }

  trait GraphCallV1 extends Operator {
    def GraphCallV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, graph_name: (String), Inputs: Seq[Tensor[T, _]]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("graph_name" -> graph_name)
      val allInputs             = Tuple.fromArray(Inputs.toArray).asInstanceOf[Tuple]
      (callOp(name, "GraphCall", allInputs, map))
    }
  }
*/

  trait GreaterOrEqualV12 extends Operator {
    def GreaterOrEqualV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](name: String, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "GreaterOrEqual", allInputs, map))
    }
  }

  trait GreaterV9 extends Operator {
    def GreaterV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](name: String, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Greater", allInputs, map))
    }
  }

  trait GreaterV7 extends Operator {
    def GreaterV7[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](name: String, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Greater", allInputs, map))
    }
  }

  trait GreaterV1 extends Operator {
    def GreaterV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Greater", allInputs, map))
    }
  }

  //Not supported, missing in ONNXJS
  /*
  trait HardSigmoidV6 extends Operator {
    def HardSigmoidV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "beta" -> beta)
      val allInputs             = Seq(X)
      (callOp(name, "HardSigmoid", allInputs, map))
    }
  }

  trait HardSigmoidV1 extends Operator {
    def HardSigmoidV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "consumed_inputs" -> consumed_inputs)
      val allInputs = Seq(X)
      (callOp(name, "HardSigmoid", allInputs, map))
    }
  }

  trait HardmaxV11 extends Operator {
    def HardmaxV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "Hardmax", allInputs, map))
    }
  }

  trait HardmaxV1 extends Operator {
    def HardmaxV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "Hardmax", allInputs, map))
    }
  }

  trait IdentityV1 extends Operator {
    def IdentityV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, input: Tensor[T, _]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Identity", allInputs, map))
    }
  }

  trait IfV11 extends Operator {
    def IfV11[
        @sp B <: Boolean,
        @sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        else_branch: (Graph),
        then_branch: (Graph),
        cond: Tensor[B, _]
    ): Tensor[V, _] = {
      val map: Map[String, Any] = Map("else_branch" -> else_branch, "then_branch" -> then_branch)
      val allInputs             = Seq(cond)
      (callOp(name, "If", allInputs, map))
    }
  }

  trait IfV1 extends Operator {
    def IfV1[
        @sp B <: Boolean,
        @sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        else_branch: (Graph),
        then_branch: (Graph),
        cond: Tensor[B, _]
    ): Tensor[V, _] = {
      val map: Map[String, Any] = Map("else_branch" -> else_branch, "then_branch" -> then_branch)
      val allInputs             = Seq(cond)
      (callOp(name, "If", allInputs, map))
    }
  }

  trait ImputerV1 extends Operator {
    def ImputerV1[@sp T <: Float | Double | Long | Int: Numeric, Ax <: Axes](
        name: String,
        imputed_value_floats: Option[(Array[Float])] = None,
        imputed_value_int64s: Option[(Array[Int])] = None,
        replaced_value_float: Option[(Float)] = None,
        replaced_value_int64: Option[(Int)] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "imputed_value_floats" -> imputed_value_floats,
        "imputed_value_int64s" -> imputed_value_int64s,
        "replaced_value_float" -> replaced_value_float,
        "replaced_value_int64" -> replaced_value_int64
      )
      val allInputs = Seq(X)
      (callOp(name, "Imputer", allInputs, map))
    }
  }
*/
  trait InstanceNormalizationV6 extends Operator {
    def InstanceNormalizationV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        epsilon: Option[(Float)] = None,
        input: Tensor[T, Ax],
        scale: Tensor[T, Bx],
        B: Tensor[T, Bx]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("epsilon" -> epsilon)
      val allInputs             = Seq(input, scale, B)
      (callOp(name, "InstanceNormalization", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait InstanceNormalizationV1 extends Operator {
    def InstanceNormalizationV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        epsilon: Option[(Float)] = None,
        input: Tensor[T, _],
        scale: Tensor[T, _],
        B: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs, "epsilon" -> epsilon)
      val allInputs             = Seq(input, scale, B)
      (callOp(name, "InstanceNormalization", allInputs, map))
    }
  }

  trait InverseV12 extends Operator {
    def InverseV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, X: Tensor[T, _]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Inverse", allInputs, map))
    }
  }

  trait IsInfV10 extends Operator {
    def IsInfV10[@sp T1 <: Float | Double: Numeric, @sp T2 <: Boolean, Ax <: Axes](
        name: String,
        detect_negative: Option[(Int)] = None,
        detect_positive: Option[(Int)] = None,
        X: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("detect_negative" -> detect_negative, "detect_positive" -> detect_positive)
      val allInputs = Seq(X)
      (callOp(name, "IsInf", allInputs, map))
    }
  }
*/
  trait IsNaNV9 extends Operator {
    def IsNaNV9[
        @sp T1 <: Float16 | Float | Double: Numeric,
        @sp T2 <: Boolean
    , Ax <: Axes](name: String, X: Tensor[T1, Ax]): Tensor[T2, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "IsNaN", allInputs, map))
    }
  }

  trait LRNV1 extends Operator {
    def LRNV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        bias: Option[(Float)] = None,
        size: (Int),
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "bias" -> bias, "size" -> size)
      val allInputs = Seq(X)
      (callOp(name, "LRN", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait LSTMV7 extends Operator {
    def LSTMV7[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        input_forget: Option[(Int)] = None,
        X: Tensor[T, _],
        W: Tensor[T, _],
        R: Tensor[T, _],
        B: Option[Tensor[T, _]] = None,
        sequence_lens: Option[Tensor[T1,_]] = None,
        initial_h: Option[Tensor[T, _]] = None,
        initial_c: Option[Tensor[T, _]] = None,
        P: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size,
        "input_forget"     -> input_forget
      )
      val allInputs = Seq(X, W, R, B, sequence_lens, initial_h, initial_c, P)
      (callOp(name, "LSTM", allInputs, map))
    }
  }

  trait LSTMV1 extends Operator {
    def LSTMV1[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        input_forget: Option[(Int)] = None,
        output_sequence: Option[(Int)] = None,
        X: Tensor[T, _],
        W: Tensor[T, _],
        R: Tensor[T, _],
        B: Option[Tensor[T, _]] = None,
        sequence_lens: Option[Tensor[T1,_]] = None,
        initial_h: Option[Tensor[T, _]] = None,
        initial_c: Option[Tensor[T, _]] = None,
        P: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size,
        "input_forget"     -> input_forget,
        "output_sequence"  -> output_sequence
      )
      val allInputs = Seq(X, W, R, B, sequence_lens, initial_h, initial_c, P)
      (callOp(name, "LSTM", allInputs, map))
    }
  }
*/
  //Not supported, ONNX ML
  /*
  trait LabelEncoderV2 extends Operator {
    def LabelEncoderV2[
        @sp T1 <: String | Long | Float: Numeric,
        @sp T2 <: String | Long | Float: Numeric
    , Ax <: Axes](
        name: String,
        default_float: Option[(Float)] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        keys_floats: Option[(Array[Float])] = None,
        keys_int64s: Option[(Array[Int])] = None,
        keys_strings: Option[(Array[String])] = None,
        values_floats: Option[(Array[Float])] = None,
        values_int64s: Option[(Array[Int])] = None,
        values_strings: Option[(Array[String])] = None,
        X: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] = Map(
        "default_float"  -> default_float,
        "default_int64"  -> default_int64,
        "default_string" -> default_string,
        "keys_floats"    -> keys_floats,
        "keys_int64s"    -> keys_int64s,
        "keys_strings"   -> keys_strings,
        "values_floats"  -> values_floats,
        "values_int64s"  -> values_int64s,
        "values_strings" -> values_strings
      )
      val allInputs = Seq(X)
      (callOp(name, "LabelEncoder", allInputs, map))
    }
  }

  trait LabelEncoderV1 extends Operator {
    def LabelEncoderV1[
        @sp T1 <: String | Long | Float: Numeric,
        @sp T2 <: String | Long | Float: Numeric
    , Ax <: Axes](
        name: String,
        classes_strings: Option[(Array[String])] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        X: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] = Map(
        "classes_strings" -> classes_strings,
        "default_int64"   -> default_int64,
        "default_string"  -> default_string
      )
      val allInputs = Seq(X)
      (callOp(name, "LabelEncoder", allInputs, map))
    }
  }
*/

  trait LeakyReluV6 extends Operator {
    def LeakyReluV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("alpha" -> alpha)
      val allInputs             = Seq(X)
      (callOp(name, "LeakyRelu", allInputs, map))
    }
  }

  trait LeakyReluV1 extends Operator {
    def LeakyReluV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "LeakyRelu", allInputs, map))
    }
  }

  trait LessOrEqualV12 extends Operator {
    def LessOrEqualV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](name: String, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "LessOrEqual", allInputs, map))
    }
  }

  trait LessV9 extends Operator {
    def LessV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](name: String, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Less", allInputs, map))
    }
  }

  trait LessV7 extends Operator {
    def LessV7[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](name: String, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Less", allInputs, map))
    }
  }

  trait LessV1 extends Operator {
    def LessV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: Boolean
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Less", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait LinearClassifierV1 extends Operator {
    def LinearClassifierV1[
        @sp T1 <: Float | Double | Long | Int: Numeric,
        @sp T2 <: String | Long: Numeric
    , Ax <: Axes](
        name: String,
        classlabels_ints: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        coefficients: (Array[Float]),
        intercepts: Option[(Array[Float])] = None,
        multi_class: Option[(Int)] = None,
        post_transform: Option[(String)] = None,
        X: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] = Map(
        "classlabels_ints"    -> classlabels_ints,
        "classlabels_strings" -> classlabels_strings,
        "coefficients"        -> coefficients,
        "intercepts"          -> intercepts,
        "multi_class"         -> multi_class,
        "post_transform"      -> post_transform
      )
      val allInputs = Seq(X)
      (callOp(name, "LinearClassifier", allInputs, map))
    }
  }

  trait LinearRegressorV1 extends Operator {
    def LinearRegressorV1[@sp T <: Float | Double | Long | Int: Numeric, Ax <: Axes](
        name: String,
        coefficients: Option[(Array[Float])] = None,
        intercepts: Option[(Array[Float])] = None,
        post_transform: Option[(String)] = None,
        targets: Option[(Int)] = None,
        X: Tensor[T, _]
    ): Tensor[Float,_] = {
      val map: Map[String, Any] = Map(
        "coefficients"   -> coefficients,
        "intercepts"     -> intercepts,
        "post_transform" -> post_transform,
        "targets"        -> targets
      )
      val allInputs = Seq(X)
      (callOp(name, "LinearRegressor", allInputs, map))
    }
  }
*/
  //Not supported, missing from ONNXJS
  /*
  trait LogSoftmaxV11 extends Operator {
    def LogSoftmaxV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "LogSoftmax", allInputs, map))
    }
  }

  trait LogSoftmaxV1 extends Operator {
    def LogSoftmaxV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "LogSoftmax", allInputs, map))
    }
  }
*/
  trait LogV6 extends Operator {
    def LogV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Log", allInputs, map))
    }
  }

  trait LogV1 extends Operator {
    def LogV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(input)
      (callOp(name, "Log", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait LoopV11 extends Operator {
    def LoopV11[
        @sp I <: Long: Numeric,
        @sp B <: Boolean,
        @sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        body: (Graph),
        M: Option[Tensor[I, _]] = None,
        cond: Option[Tensor[B, _]] = None,
        v_initial: Seq[Tensor[V, _]]
    ): Tensor[V, _] = {
      val map: Map[String, Any] = Map("body" -> body)
      val allInputs = 
        Seq(M, cond) ++ (Tuple.fromArray(v_initial.toArray).asInstanceOf[Tuple])

      (callOp(name, "Loop", allInputs, map))
    }
  }

  trait LoopV1 extends Operator {
    def LoopV1[
        @sp I <: Long: Numeric,
        @sp B <: Boolean,
        @sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        body: (Graph),
        M: Option[Tensor[I, _]] = None,
        cond: Option[Tensor[B, _]] = None,
        v_initial: Seq[Tensor[V, _]]
    ): Tensor[V, _] = {
      val map: Map[String, Any] = Map("body" -> body)
      val allInputs =
        Seq(M, cond) ++ (Tuple.fromArray(v_initial.toArray).asInstanceOf[Tuple])

      (callOp(name, "Loop", allInputs, map))
    }
  }

  trait LpNormalizationV1 extends Operator {
    def LpNormalizationV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        p: Option[(Int)] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "p" -> p)
      val allInputs             = Seq(input)
      (callOp(name, "LpNormalization", allInputs, map))
    }
  }

  trait LpPoolV11 extends Operator {
    def LpPoolV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: (Array[Int]),
        p: Option[(Int)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "p"            -> p,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "LpPool", allInputs, map))
    }
  }

  trait LpPoolV2 extends Operator {
    def LpPoolV2[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: (Array[Int]),
        p: Option[(Int)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "p"            -> p,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "LpPool", allInputs, map))
    }
  }

  trait LpPoolV1 extends Operator {
    def LpPoolV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        p: Option[(Float)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "p"            -> p,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "LpPool", allInputs, map))
    }
  }

  trait MatMulIntegerV10 extends Operator {
    def MatMulIntegerV10[
        @sp T1 <: Byte | UByte: Numeric,
        @sp T2 <: Byte | UByte: Numeric,
        @sp T3 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        A: Tensor[T1,_],
        B: Tensor[T2, _],
        a_zero_point: Option[Tensor[T1,_]] = None,
        b_zero_point: Option[Tensor[T2, _]] = None
    ): Tensor[T3, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A, B, a_zero_point, b_zero_point)
      (callOp(name, "MatMulInteger", allInputs, map))
    }
  }
*/
  //TODO: Constraint
  trait MatMulV9 extends Operator {
    def MatMulV9[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Bx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "MatMul", allInputs, map))
    }
  }

  trait MatMulV1 extends Operator {
    def MatMulV1[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Bx]
    ): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "MatMul", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait MaxPoolV12 extends Operator {
    def MaxPoolV12[
        @sp T <: Float16 | Float | Double | Byte | UByte: Numeric,
        @sp I <: Long: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        dilations: Option[(Array[Int])] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"      -> auto_pad,
        "ceil_mode"     -> ceil_mode,
        "dilations"     -> dilations,
        "kernel_shape"  -> kernel_shape,
        "pads"          -> pads,
        "storage_order" -> storage_order,
        "strides"       -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "MaxPool", allInputs, map))
    }
  }

  trait MaxPoolV11 extends Operator {
    def MaxPoolV11[
        @sp T <: Float16 | Float | Double | Byte | UByte: Numeric,
        @sp I <: Long: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        dilations: Option[(Array[Int])] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"      -> auto_pad,
        "ceil_mode"     -> ceil_mode,
        "dilations"     -> dilations,
        "kernel_shape"  -> kernel_shape,
        "pads"          -> pads,
        "storage_order" -> storage_order,
        "strides"       -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "MaxPool", allInputs, map))
    }
  }
*/
  //TODO: constraints
  trait MaxPoolV10 extends Operator {
    def MaxPoolV10[
        @sp T <: Float16 | Float | Double | Byte | UByte: Numeric,
        @sp I <: Long: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        dilations: Option[(Array[Int])] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"      -> auto_pad,
        "ceil_mode"     -> ceil_mode,
        "dilations"     -> dilations,
        "kernel_shape"  -> kernel_shape,
        "pads"          -> pads,
        "storage_order" -> storage_order,
        "strides"       -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "MaxPool", allInputs, map))
    }
  }

  trait MaxPoolV8 extends Operator {
    def MaxPoolV8[
        @sp T <: Float16 | Float | Double | Byte | UByte: Numeric,
        @sp I <: Long: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"      -> auto_pad,
        "kernel_shape"  -> kernel_shape,
        "pads"          -> pads,
        "storage_order" -> storage_order,
        "strides"       -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "MaxPool", allInputs, map))
    }
  }

  trait MaxPoolV1 extends Operator {
    def MaxPoolV1[@sp T <: Float16 | Float | Double | Byte | UByte: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "MaxPool", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait MaxRoiPoolV1 extends Operator {
    def MaxRoiPoolV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        pooled_shape: (Array[Int]),
        spatial_scaleAttr: Option[(Float)] = None,
        X: Tensor[T, _],
        rois: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] =
        Map("pooled_shape" -> pooled_shape, "spatial_scaleAttr" -> spatial_scaleAttr)
      val allInputs = Seq(X, rois)
      (callOp(name, "MaxRoiPool", allInputs, map))
    }
  }

  trait MaxUnpoolV11 extends Operator {
    def MaxUnpoolV11[
        @sp T1 <: Float16 | Float | Double: Numeric,
        @sp T2 <: Long: Numeric
    , Ax <: Axes](
        name: String,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T1,_],
        I: Tensor[T2, _],
        output_shapeInput: Option[Tensor[T2, _]] = None
    ): Tensor[T1,_] = {
      val map: Map[String, Any] =
        Map("kernel_shape" -> kernel_shape, "pads" -> pads, "strides" -> strides)
      val allInputs = Seq(X, I, output_shapeInput)
      (callOp(name, "MaxUnpool", allInputs, map))
    }
  }

  trait MaxUnpoolV9 extends Operator {
    def MaxUnpoolV9[
        @sp T1 <: Float16 | Float | Double: Numeric,
        @sp T2 <: Long: Numeric
    , Ax <: Axes](
        name: String,
        kernel_shape: (Array[Int]),
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T1,_],
        I: Tensor[T2, _],
        output_shapeInput: Option[Tensor[T2, _]] = None
    ): Tensor[T1,_] = {
      val map: Map[String, Any] =
        Map("kernel_shape" -> kernel_shape, "pads" -> pads, "strides" -> strides)
      val allInputs = Seq(X, I, output_shapeInput)
      (callOp(name, "MaxUnpool", allInputs, map))
    }
  }
*/
  trait MaxV12 extends Operator {
    def MaxV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, data_0: Seq[Tensor[T, Ax]]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = data_0 
      (callOp(name, "Max", allInputs, map))
    }
  }

  trait MaxV8 extends Operator {
    def MaxV8[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, data_0: Seq[Tensor[T, Ax]]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = data_0 
      (callOp(name, "Max", allInputs, map))
    }
  }

  trait MaxV6 extends Operator {
    def MaxV6[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, data_0: Seq[Tensor[T, Ax]]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = data_0 
      (callOp(name, "Max", allInputs, map))
    }
  }

  trait MaxV1 extends Operator {
    def MaxV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        data_0: Seq[Tensor[T, Ax]]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = data_0 
      (callOp(name, "Max", allInputs, map))
    }
  }

  //Not supported, missing in ONNXJS
  /*
  trait MeanSquaredDistanceV12 extends Operator {
    def MeanSquaredDistanceV12[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        reduction: Option[(String)] = None,
        scores: Tensor[T, _],
        labels: Tensor[T, _],
        weights: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("reduction" -> reduction)
      val allInputs             = Seq(scores, labels, weights)
      (callOp(name, "MeanSquaredDistance", allInputs, map))
    }
  }

  trait MeanV8 extends Operator {
    def MeanV8[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        data_0: Seq[Tensor[T, _]]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple.fromArray(data_0.toArray).asInstanceOf[Tuple]
      (callOp(name, "Mean", allInputs, map))
    }
  }

  trait MeanV6 extends Operator {
    def MeanV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        data_0: Seq[Tensor[T, Ax]]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple.fromArray(data_0.toArray).asInstanceOf[Tuple]
      (callOp(name, "Mean", allInputs, map))
    }
  }

  trait MeanV1 extends Operator {
    def MeanV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        data_0: Seq[Tensor[T, Ax]]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Tuple.fromArray(data_0.toArray).asInstanceOf[Tuple]
      (callOp(name, "Mean", allInputs, map))
    }
  }

  trait MeanVarianceNormalizationV9 extends Operator {
    def MeanVarianceNormalizationV9[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs             = Seq(X)
      (callOp(name, "MeanVarianceNormalization", allInputs, map))
    }
  }
*/
  trait MinV12 extends Operator {
    def MinV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, data_0: Seq[Tensor[T, Ax]]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = data_0 
      (callOp(name, "Min", allInputs, map))
    }
  }

  trait MinV8 extends Operator {
    def MinV8[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, data_0: Seq[Tensor[T, Ax]]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = data_0
      (callOp(name, "Min", allInputs, map))
    }
  }

  trait MinV6 extends Operator {
    def MinV6[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, data_0: Seq[Tensor[T, Ax]]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = data_0 
      (callOp(name, "Min", allInputs, map))
    }
  }

  trait MinV1 extends Operator {
    def MinV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        data_0: Seq[Tensor[T, Ax]]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = data_0 
      (callOp(name, "Min", allInputs, map))
    }
  }

  trait ModV10 extends Operator {
    def ModV10[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, fmod: Option[(Int)] = None, A: Tensor[T, Ax], B: Tensor[T, Ax]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("fmod" -> fmod)
      val allInputs             = Seq(A,B)
      (callOp(name, "Mod", allInputs, map))
    }
  }

  //Not supported, training not yet GA
  /*
  trait MomentumV1 extends Operator {
    def MomentumV1[
        @sp T1 <: Float | Double: Numeric,
        @sp T2 <: Long: Numeric,
        @sp T3 <: Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        alpha: (Float),
        beta: (Float),
        mode: (String),
        norm_coefficient: (Float),
        R: Tensor[T1,_],
        T: Tensor[T2, _],
        inputs: Seq[Tensor[T3, _]]
    ): Tensor[T3, _] = {
      val map: Map[String, Any] = Map(
        "alpha"            -> alpha,
        "beta"             -> beta,
        "mode"             -> mode,
        "norm_coefficient" -> norm_coefficient
      )
      val allInputs = 
        Seq(R, T) ++ (Tuple.fromArray(inputs.toArray).asInstanceOf[Tuple])
      (callOp(name, "Momentum", allInputs, map))
    }
  }
*/
  trait MulV7 extends Operator {
    def MulV7[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Mul", allInputs, map))
    }
  }

  trait MulV6 extends Operator {
    def MulV6[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Mul", allInputs, map))
    }
  }

  trait MulV1 extends Operator {
    def MulV1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "broadcast" -> broadcast, "consumed_inputs" -> consumed_inputs)
      val allInputs = Seq(A,B)
      (callOp(name, "Mul", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait MultinomialV7 extends Operator {
    def MultinomialV7[
        @sp T1 <: Float16 | Float | Double: Numeric,
        @sp T2 <: Int | Long: Numeric
    , Ax <: Axes](
        name: String,
        dtype: Option[(Int)] = None,
        sample_size: Option[(Int)] = None,
        seed: Option[(Float)] = None,
        input: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "sample_size" -> sample_size, "seed" -> seed)
      val allInputs = Seq(input)
      (callOp(name, "Multinomial", allInputs, map))
    }
  }
*/

  trait NegV6 extends Operator {
    def NegV6[@sp T <: Float | Int | Byte | Short | Long | Float16 | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Neg", allInputs, map))
    }
  }

  trait NegV1 extends Operator {
    def NegV1[@sp T <: Float | Int | Byte | Short | Long | Float16 | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Neg", allInputs, map))
    }
  }
  //Not supported, missing from ONNXJS
  /*
  trait NegativeLogLikelihoodLossV12 extends Operator {
    def NegativeLogLikelihoodLossV12[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes](
        name: String,
        reduction: Option[(String)] = None,
        input: Tensor[T, _],
        target: Tensor[Tind, _],
        weight: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("reduction" -> reduction)
      val allInputs             = Seq(input, target, weight)
      (callOp(name, "NegativeLogLikelihoodLoss", allInputs, map))
    }
  }

  trait NonMaxSuppressionV11 extends Operator {
    def NonMaxSuppressionV11(
        name: String,
        center_point_box: Option[(Int)] = None,
        boxes: Tensor[Float,_],
        scores: Tensor[Float,_],
        max_output_boxes_per_class: Option[Tensor[Long, _]] = None,
        iou_threshold: Option[Tensor[Float,_]] = None,
        score_threshold: Option[Tensor[Float,_]] = None
    ): Tensor[Long, _] = {
      val map: Map[String, Any] = Map("center_point_box" -> center_point_box)
      val allInputs = 
        Seq(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
      (callOp(name, "NonMaxSuppression", allInputs, map))
    }
  }

  trait NonMaxSuppressionV10 extends Operator {
    def NonMaxSuppressionV10(
        name: String,
        center_point_box: Option[(Int)] = None,
        boxes: Tensor[Float,_],
        scores: Tensor[Float,_],
        max_output_boxes_per_class: Option[Tensor[Long, _]] = None,
        iou_threshold: Option[Tensor[Float,_]] = None,
        score_threshold: Option[Tensor[Float,_]] = None
    ): Tensor[Long, _] = {
      val map: Map[String, Any] = Map("center_point_box" -> center_point_box)
      val allInputs = 
        Seq(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
      (callOp(name, "NonMaxSuppression", allInputs, map))
    }
  }

  trait NonZeroV9 extends Operator {
    def NonZeroV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, X: Tensor[T, _]): Tensor[Long, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "NonZero", allInputs, map))
    }
  }
*/
  //Not supported, ONNX ML
  /*
  trait NormalizerV1 extends Operator {
    def NormalizerV1[@sp T <: Float | Double | Long | Int: Numeric, Ax <: Axes](
        name: String,
        norm: Option[(String)] = None,
        X: Tensor[T, _]
    ): Tensor[Float,_] = {
      val map: Map[String, Any] = Map("norm" -> norm)
      val allInputs             = Seq(X)
      (callOp(name, "Normalizer", allInputs, map))
    }
  }
*/
  trait NotV1 extends Operator {
    def NotV1[@sp T <: Boolean, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Not", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait OneHotEncoderV1 extends Operator {
    def OneHotEncoderV1[@sp T <: String | Long | Int | Float | Double: Numeric, Ax <: Axes](
        name: String,
        cats_int64s: Option[(Array[Int])] = None,
        cats_strings: Option[(Array[String])] = None,
        zeros: Option[(Int)] = None,
        X: Tensor[T, _]
    ): Tensor[Float,_] = {
      val map: Map[String, Any] =
        Map("cats_int64s" -> cats_int64s, "cats_strings" -> cats_strings, "zeros" -> zeros)
      val allInputs = Seq(X)
      (callOp(name, "OneHotEncoder", allInputs, map))
    }
  }

  trait OneHotV11 extends Operator {
    def OneHotV11[
        @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T2 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T3 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        indices: Tensor[T1,_],
        depth: Tensor[T2, _],
        values: Tensor[T3, _]
    ): Tensor[T3, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(indices, depth, values)
      (callOp(name, "OneHot", allInputs, map))
    }
  }

  trait OneHotV9 extends Operator {
    def OneHotV9[
        @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T2 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp T3 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        indices: Tensor[T1,_],
        depth: Tensor[T2, _],
        values: Tensor[T3, _]
    ): Tensor[T3, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(indices, depth, values)
      (callOp(name, "OneHot", allInputs, map))
    }
  }
*/

  trait OrV7 extends Operator {
    def OrV7[@sp T <: Boolean, @sp T1 <: Boolean, Ax <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Or", allInputs, map))
    }
  }

  trait OrV1 extends Operator {
    def OrV1[@sp T <: Boolean, @sp T1 <: Boolean, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Or", allInputs, map))
    }
  }

  trait PReluV9 extends Operator {
    def PReluV9[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax],
        slope: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X, slope)
      (callOp(name, "PRelu", allInputs, map))
    }
  }

  trait PReluV7 extends Operator {
    def PReluV7[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax],
        slope: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X, slope)
      (callOp(name, "PRelu", allInputs, map))
    }
  }

  trait PReluV6 extends Operator {
    def PReluV6[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax],
        slope: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X, slope)
      (callOp(name, "PRelu", allInputs, map))
    }
  }

  trait PReluV1 extends Operator {
    def PReluV1[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax],
        slope: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X, slope)
      (callOp(name, "PRelu", allInputs, map))
    }
  }

  trait PadV11 extends Operator {
    def PadV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes](
        name: String,
        mode: Option[(String)] = None,
        data: Tensor[T, Ax],
        pads: Tensor[Long, Bx],
        constant_value: Option[Tensor[T, Cx]] = None
    ): Tensor[T, Dx] = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs             = Seq(data, pads, constant_value)
      (callOp(name, "Pad", allInputs, map))
    }
  }

  trait PadV2 extends Operator {
    def PadV2[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        mode: Option[(String)] = None,
        pads: (Array[Int]),
        value: Option[(Float)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("mode" -> mode, "pads" -> pads, "value" -> value)
      val allInputs             = Seq(data)
      (callOp(name, "Pad", allInputs, map))
    }
  }

  trait PadV1 extends Operator {
    def PadV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        mode: Option[(String)] = None,
        paddings: (Array[Int]),
        value: Option[(Float)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("mode" -> mode, "paddings" -> paddings, "value" -> value)
      val allInputs             = Seq(data)
      (callOp(name, "Pad", allInputs, map))
    }
  }

  trait PowV12 extends Operator {
    def PowV12[
        @sp T <: Int | Long | Float16 | Float | Double: Numeric,
        @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, X: Tensor[T, Ax], Y: Tensor[T1, Ax]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X,Y)
      (callOp(name, "Pow", allInputs, map))
    }
  }

  trait PowV7 extends Operator {
    def PowV7[@sp T <: Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax],
        Y: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X,Y)
      (callOp(name, "Pow", allInputs, map))
    }
  }

  trait PowV1 extends Operator {
    def PowV1[@sp T <: Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        X: Tensor[T, Ax],
        Y: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(X,Y)
      (callOp(name, "Pow", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait QLinearConvV10 extends Operator {
    def QLinearConvV10[
        @sp T1 <: Byte | UByte: Numeric,
        @sp T2 <: Byte | UByte: Numeric,
        @sp T3 <: Byte | UByte: Numeric,
        @sp T4 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        x: Tensor[T1,_],
        x_scale: Tensor[Float,_],
        x_zero_point: Tensor[T1,_],
        w: Tensor[T2, _],
        w_scale: Tensor[Float,_],
        w_zero_point: Tensor[T2, _],
        y_scale: Tensor[Float,_],
        y_zero_point: Tensor[T3, _],
        B: Option[Tensor[T4, _]] = None
    ): Tensor[T3, _] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = 
        Seq(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B)
      (callOp(name, "QLinearConv", allInputs, map))
    }
  }

  trait QLinearMatMulV10 extends Operator {
    def QLinearMatMulV10[
        @sp T1 <: Byte | UByte: Numeric,
        @sp T2 <: Byte | UByte: Numeric,
        @sp T3 <: Byte | UByte: Numeric
    , Ax <: Axes](
        name: String,
        a: Tensor[T1,_],
        a_scale: Tensor[Float,_],
        a_zero_point: Tensor[T1,_],
        b: Tensor[T2, _],
        b_scale: Tensor[Float,_],
        b_zero_point: Tensor[T2, _],
        y_scale: Tensor[Float,_],
        y_zero_point: Tensor[T3, _]
    ): Tensor[T3, _] = {
      val map: Map[String, Any] = Map()
      val allInputs = 
        Seq(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point)
      (callOp(name, "QLinearMatMul", allInputs, map))
    }
  }

  trait QuantizeLinearV10 extends Operator {
    def QuantizeLinearV10[
        @sp T1 <: Float | Int: Numeric,
        @sp T2 <: Byte | UByte: Numeric
    , Ax <: Axes](
        name: String,
        x: Tensor[T1,_],
        y_scale: Tensor[Float,_],
        y_zero_point: Option[Tensor[T2, _]] = None
    ): Tensor[T2, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(x, y_scale, y_zero_point)
      (callOp(name, "QuantizeLinear", allInputs, map))
    }
  }

  trait RNNV7 extends Operator {
    def RNNV7[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        X: Tensor[T, _],
        W: Tensor[T, _],
        R: Tensor[T, _],
        B: Option[Tensor[T, _]] = None,
        sequence_lens: Option[Tensor[T1,_]] = None,
        initial_h: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size
      )
      val allInputs = Seq(X, W, R, B, sequence_lens, initial_h)
      (callOp(name, "RNN", allInputs, map))
    }
  }

  trait RNNV1 extends Operator {
    def RNNV1[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp T1 <: Int: Numeric
    , Ax <: Axes](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        output_sequence: Option[(Int)] = None,
        X: Tensor[T, _],
        W: Tensor[T, _],
        R: Tensor[T, _],
        B: Option[Tensor[T, _]] = None,
        sequence_lens: Option[Tensor[T1,_]] = None,
        initial_h: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size,
        "output_sequence"  -> output_sequence
      )
      val allInputs = Seq(X, W, R, B, sequence_lens, initial_h)
      (callOp(name, "RNN", allInputs, map))
    }
  }

  trait RandomNormalLikeV1 extends Operator {
    def RandomNormalLikeV1[
        @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T2 <: Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        dtype: Option[(Int)] = None,
        mean: Option[(Float)] = None,
        scaleAttr: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        input: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "mean" -> mean, "scaleAttr" -> scaleAttr, "seed" -> seed)
      val allInputs = Seq(input)
      (callOp(name, "RandomNormalLike", allInputs, map))
    }
  }

  trait RandomNormalV1 extends Operator {
    def RandomNormalV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        dtype: Option[(Int)] = None,
        mean: Option[(Float)] = None,
        scaleAttr: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        shape: (Array[Int])
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "dtype"     -> dtype,
        "mean"      -> mean,
        "scaleAttr" -> scaleAttr,
        "seed"      -> seed,
        "shape"     -> shape
      )
      val allInputs = Seq()
      (callOp(name, "RandomNormal", allInputs, map))
    }
  }

  trait RandomUniformLikeV1 extends Operator {
    def RandomUniformLikeV1[
        @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T2 <: Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        dtype: Option[(Int)] = None,
        high: Option[(Float)] = None,
        low: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        input: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "high" -> high, "low" -> low, "seed" -> seed)
      val allInputs = Seq(input)
      (callOp(name, "RandomUniformLike", allInputs, map))
    }
  }

  trait RandomUniformV1 extends Operator {
    def RandomUniformV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        dtype: Option[(Int)] = None,
        high: Option[(Float)] = None,
        low: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        shape: (Array[Int])
    ): Tensor[T, _] = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "high" -> high, "low" -> low, "seed" -> seed, "shape" -> shape)
      val allInputs = Seq()
      (callOp(name, "RandomUniform", allInputs, map))
    }
  }
*/
  trait RangeV11 extends Operator {
    def RangeV11[@sp T <: Float | Double | Short | Int | Long: Numeric, Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes](
        name: String,
        start: Tensor[T, Ax],
        limit: Tensor[T, Bx],
        delta: Tensor[T, Cx]
    ): Tensor[T, Dx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(start, limit, delta)
      (callOp(name, "Range", allInputs, map))
    }
  }

  trait ReciprocalV6 extends Operator {
    def ReciprocalV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Reciprocal", allInputs, map))
    }
  }

  trait ReciprocalV1 extends Operator {
    def ReciprocalV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Reciprocal", allInputs, map))
    }
  }

  //Not supported, missing in ONNXJS
  /*
  trait ReduceL1V11 extends Operator {
    def ReduceL1V11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceL1", allInputs, map))
    }
  }

  trait ReduceL1V1 extends Operator {
    def ReduceL1V1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceL1", allInputs, map))
    }
  }

  trait ReduceL2V11 extends Operator {
    def ReduceL2V11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceL2", allInputs, map))
    }
  }

  trait ReduceL2V1 extends Operator {
    def ReduceL2V1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceL2", allInputs, map))
    }
  }

  trait ReduceLogSumExpV11 extends Operator {
    def ReduceLogSumExpV11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceLogSumExp", allInputs, map))
    }
  }

  trait ReduceLogSumExpV1 extends Operator {
    def ReduceLogSumExpV1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceLogSumExp", allInputs, map))
    }
  }
*/
  //tf-dotty reduce eligible
  trait ReduceLogSumV11 extends Operator {
    def ReduceLogSumV11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
        , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceLogSum", allInputs, map))
    }
  }

  trait ReduceLogSumV1 extends Operator {
    def ReduceLogSumV1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceLogSum", allInputs, map))
    }
  }

  //tf-dotty reduce eligible
  trait ReduceMaxV12 extends Operator {
    def ReduceMaxV12[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceMax", allInputs, map))
    }
  }

  trait ReduceMaxV11 extends Operator {
    def ReduceMaxV11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceMax", allInputs, map))
    }
  }

  trait ReduceMaxV1 extends Operator {
    def ReduceMaxV1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceMax", allInputs, map))
    }
  }

  //tf-dotty reduce eligible
  trait ReduceMeanV11 extends Operator {
    def ReduceMeanV11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceMean", allInputs, map))
    }
  }

  trait ReduceMeanV1 extends Operator {
    def ReduceMeanV1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceMean", allInputs, map))
    }
  }

  //tf-dotty reduce eligible
  trait ReduceMinV12 extends Operator {
    def ReduceMinV12[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceMin", allInputs, map))
    }
  }

  trait ReduceMinV11 extends Operator {
    def ReduceMinV11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceMin", allInputs, map))
    }
  }

  trait ReduceMinV1 extends Operator {
    def ReduceMinV1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceMin", allInputs, map))
    }
  }

  //tf-dotty reduce eligible
  trait ReduceProdV11 extends Operator {
    def ReduceProdV11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceProd", allInputs, map))
    }
  }

  trait ReduceProdV1 extends Operator {
    def ReduceProdV1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceProd", allInputs, map))
    }
  }

  //tf-dotty reduce eligible
  trait ReduceSumSquareV11 extends Operator {
    def ReduceSumSquareV11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceSumSquare", allInputs, map))
    }
  }

  trait ReduceSumSquareV1 extends Operator {
    def ReduceSumSquareV1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceSumSquare", allInputs, map))
    }
  }

  //tf-dotty reduce eligible
  trait ReduceSumV11 extends Operator {
    def ReduceSumV11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceSum", allInputs, map))
    }
  }

  trait ReduceSumV1 extends Operator {
    def ReduceSumV1[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Seq(data)
      (callOp(name, "ReduceSum", allInputs, map))
    }
  }

  trait ReluV6 extends Operator {
    def ReluV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Relu", allInputs, map))
    }
  }

  trait ReluV1 extends Operator {
    def ReluV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Relu", allInputs, map))
    }
  }

  //TODO: Constraint, //tf-dotty reshape eligible
  //(given NumElements[Old] =:= NumElements[New]
  //+ match types on axes
  trait ReshapeV5 extends Operator {
    def ReshapeV5[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](name: String, data: Tensor[T, Ax], shapeInput: Tensor[Long, Bx]): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(data, shapeInput)
      (callOp(name, "Reshape", allInputs, map))
    }
  }

  trait ReshapeV1 extends Operator {
    def ReshapeV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        shape: Option[(Array[Int])] = None,
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs, "shape" -> shape)
      val allInputs             = Seq(data)
      (callOp(name, "Reshape", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait ResizeV11 extends Operator {
    def ResizeV11[
        @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T2 <: Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        coordinate_transformation_mode: Option[(String)] = None,
        cubic_coeff_a: Option[(Float)] = None,
        exclude_outside: Option[(Int)] = None,
        extrapolation_value: Option[(Float)] = None,
        mode: Option[(String)] = None,
        nearest_mode: Option[(String)] = None,
        X: Tensor[T1,_],
        roi: Tensor[T2, _],
        scales: Tensor[Float,_],
        sizes: Option[Tensor[Long, _]] = None
    ): Tensor[T1,_] = {
      val map: Map[String, Any] = Map(
        "coordinate_transformation_mode" -> coordinate_transformation_mode,
        "cubic_coeff_a"                  -> cubic_coeff_a,
        "exclude_outside"                -> exclude_outside,
        "extrapolation_value"            -> extrapolation_value,
        "mode"                           -> mode,
        "nearest_mode"                   -> nearest_mode
      )
      val allInputs = Seq(X, roi, scales, sizes)
      (callOp(name, "Resize", allInputs, map))
    }
  }

  trait ResizeV10 extends Operator {
    def ResizeV10[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        mode: Option[(String)] = None,
        X: Tensor[T, _],
        scales: Tensor[Float,_]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs             = Seq(X, scales)
      (callOp(name, "Resize", allInputs, map))
    }
  }
*/
  //Not supported, sequence op
  /*
  trait ReverseSequenceV10 extends Operator {
    def ReverseSequenceV10[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        batch_axis: Option[(Int)] = None,
        time_axis: Option[(Int)] = None,
        input: Tensor[T, _],
        sequence_lens: Tensor[Long, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("batch_axis" -> batch_axis, "time_axis" -> time_axis)
      val allInputs             = Seq(input, sequence_lens)
      (callOp(name, "ReverseSequence", allInputs, map))
    }
  }
*/
  //Not supported, missing from ONNXJS
  /*
  trait RoiAlignV10 extends Operator {
    def RoiAlignV10[
        @sp T1 <: Float16 | Float | Double: Numeric,
        @sp T2 <: Long: Numeric
    , Ax <: Axes](
        name: String,
        mode: Option[(String)] = None,
        output_height: Option[(Int)] = None,
        output_width: Option[(Int)] = None,
        sampling_ratio: Option[(Int)] = None,
        spatial_scaleAttr: Option[(Float)] = None,
        X: Tensor[T1,_],
        rois: Tensor[T1,_],
        batch_indices: Tensor[T2, _]
    ): Tensor[T1,_] = {
      val map: Map[String, Any] = Map(
        "mode"              -> mode,
        "output_height"     -> output_height,
        "output_width"      -> output_width,
        "sampling_ratio"    -> sampling_ratio,
        "spatial_scaleAttr" -> spatial_scaleAttr
      )
      val allInputs = Seq(X, rois, batch_indices)
      (callOp(name, "RoiAlign", allInputs, map))
    }
  }
*/
  trait RoundV11 extends Operator {
    def RoundV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Round", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait SVMClassifierV1 extends Operator {
    def SVMClassifierV1[
        @sp T1 <: Float | Double | Long | Int: Numeric,
        @sp T2 <: String | Long: Numeric
    , Ax <: Axes](
        name: String,
        classlabels_ints: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        coefficients: Option[(Array[Float])] = None,
        kernel_params: Option[(Array[Float])] = None,
        kernel_type: Option[(String)] = None,
        post_transform: Option[(String)] = None,
        prob_a: Option[(Array[Float])] = None,
        prob_b: Option[(Array[Float])] = None,
        rho: Option[(Array[Float])] = None,
        support_vectors: Option[(Array[Float])] = None,
        vectors_per_class: Option[(Array[Int])] = None,
        X: Tensor[T1, _]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] = Map(
        "classlabels_ints"    -> classlabels_ints,
        "classlabels_strings" -> classlabels_strings,
        "coefficients"        -> coefficients,
        "kernel_params"       -> kernel_params,
        "kernel_type"         -> kernel_type,
        "post_transform"      -> post_transform,
        "prob_a"              -> prob_a,
        "prob_b"              -> prob_b,
        "rho"                 -> rho,
        "support_vectors"     -> support_vectors,
        "vectors_per_class"   -> vectors_per_class
      )
      val allInputs = Seq(X)
      (callOp(name, "SVMClassifier", allInputs, map))
    }
  }

  trait SVMRegressorV1 extends Operator {
    def SVMRegressorV1[@sp T <: Float | Double | Long | Int: Numeric, Ax <: Axes](
        name: String,
        coefficients: Option[(Array[Float])] = None,
        kernel_params: Option[(Array[Float])] = None,
        kernel_type: Option[(String)] = None,
        n_supports: Option[(Int)] = None,
        one_class: Option[(Int)] = None,
        post_transform: Option[(String)] = None,
        rho: Option[(Array[Float])] = None,
        support_vectors: Option[(Array[Float])] = None,
        X: Tensor[T, _]
    ): Tensor[Float,_] = {
      val map: Map[String, Any] = Map(
        "coefficients"    -> coefficients,
        "kernel_params"   -> kernel_params,
        "kernel_type"     -> kernel_type,
        "n_supports"      -> n_supports,
        "one_class"       -> one_class,
        "post_transform"  -> post_transform,
        "rho"             -> rho,
        "support_vectors" -> support_vectors
      )
      val allInputs = Seq(X)
      (callOp(name, "SVMRegressor", allInputs, map))
    }
  }

  trait ScalerV1 extends Operator {
    def ScalerV1[@sp T <: Float | Double | Long | Int: Numeric, Ax <: Axes](
        name: String,
        offset: Option[(Array[Float])] = None,
        scaleAttr: Option[(Array[Float])] = None,
        X: Tensor[T, _]
    ): Tensor[Float,_] = {
      val map: Map[String, Any] = Map("offset" -> offset, "scaleAttr" -> scaleAttr)
      val allInputs             = Seq(X)
      (callOp(name, "Scaler", allInputs, map))
    }
  }
*/
  //Not supported, missing from ONNXJS
  /*
  trait ScanV11 extends Operator {
    def ScanV11[
        @sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        body: (Graph),
        num_scan_inputs: (Int),
        scan_input_axes: Option[(Array[Int])] = None,
        scan_input_directions: Option[(Array[Int])] = None,
        scan_output_axes: Option[(Array[Int])] = None,
        scan_output_directions: Option[(Array[Int])] = None,
        initial_state_and_scan_inputs: Seq[Tensor[V, _]]
    ): Tensor[V, _] = {
      val map: Map[String, Any] = Map(
        "body"                   -> body,
        "num_scan_inputs"        -> num_scan_inputs,
        "scan_input_axes"        -> scan_input_axes,
        "scan_input_directions"  -> scan_input_directions,
        "scan_output_axes"       -> scan_output_axes,
        "scan_output_directions" -> scan_output_directions
      )
      val allInputs = 
        Tuple.fromArray(initial_state_and_scan_inputs.toArray).asInstanceOf[Tuple]
      (callOp(name, "Scan", allInputs, map))
    }
  }

  trait ScanV9 extends Operator {
    def ScanV9[
        @sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        body: (Graph),
        num_scan_inputs: (Int),
        scan_input_axes: Option[(Array[Int])] = None,
        scan_input_directions: Option[(Array[Int])] = None,
        scan_output_axes: Option[(Array[Int])] = None,
        scan_output_directions: Option[(Array[Int])] = None,
        initial_state_and_scan_inputs: Seq[Tensor[V, _]]
    ): Tensor[V, _] = {
      val map: Map[String, Any] = Map(
        "body"                   -> body,
        "num_scan_inputs"        -> num_scan_inputs,
        "scan_input_axes"        -> scan_input_axes,
        "scan_input_directions"  -> scan_input_directions,
        "scan_output_axes"       -> scan_output_axes,
        "scan_output_directions" -> scan_output_directions
      )
      val allInputs = 
        Tuple.fromArray(initial_state_and_scan_inputs.toArray).asInstanceOf[Tuple]
      (callOp(name, "Scan", allInputs, map))
    }
  }

  trait ScanV8 extends Operator {
    def ScanV8[
        @sp I <: Long: Numeric,
        @sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        body: (Graph),
        directions: Option[(Array[Int])] = None,
        num_scan_inputs: (Int),
        sequence_lens: Option[Tensor[I, _]] = None,
        initial_state_and_scan_inputs: Seq[Tensor[V, _]]
    ): Tensor[V, _] = {
      val map: Map[String, Any] =
        Map("body" -> body, "directions" -> directions, "num_scan_inputs" -> num_scan_inputs)
      val allInputs = 
        Seq(sequence_lens) ++ (Tuple
          .fromArray(initial_state_and_scan_inputs.toArray)
          .asInstanceOf[Tuple])
      (callOp(name, "Scan", allInputs, map))
    }
  }

  trait ScatterElementsV11 extends Operator {
    def ScatterElementsV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        data: Tensor[T, _],
        indices: Tensor[Tind, _],
        updates: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(data, indices, updates)
      (callOp(name, "ScatterElements", allInputs, map))
    }
  }

  trait ScatterNDV11 extends Operator {
    def ScatterNDV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        data: Tensor[T, _],
        indices: Tensor[Long, _],
        updates: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(data, indices, updates)
      (callOp(name, "ScatterND", allInputs, map))
    }
  }

  //Deprecated
  trait ScatterV11 extends Operator {
    def ScatterV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        data: Tensor[T, _],
        indices: Tensor[Tind, _],
        updates: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(data, indices, updates)
      (callOp(name, "Scatter", allInputs, map))
    }
  }

  trait ScatterV9 extends Operator {
    def ScatterV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        data: Tensor[T, _],
        indices: Tensor[Tind, _],
        updates: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(data, indices, updates)
      (callOp(name, "Scatter", allInputs, map))
    }
  }

  trait SeluV6 extends Operator {
    def SeluV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        gamma: Option[(Float)] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "gamma" -> gamma)
      val allInputs             = Seq(X)
      (callOp(name, "Selu", allInputs, map))
    }
  }

  trait SeluV1 extends Operator {
    def SeluV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        gamma: Option[(Float)] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "consumed_inputs" -> consumed_inputs, "gamma" -> gamma)
      val allInputs = Seq(X)
      (callOp(name, "Selu", allInputs, map))
    }
  }
*/
  //Not supported, sequence op
  /*
  trait SequenceAtV11 extends Operator {
    def SequenceAtV11[@sp S <: Seq[Tensor[UByte, _]] | Seq[Tensor[UShort, _]] | Seq[Tensor[UInt, _]] | Seq[
      Tensor[ULong, _]
    ] | Seq[Tensor[Byte, _]] | Seq[Tensor[Short, _]] | Seq[Tensor[Int, _]] | Seq[Tensor[Long, _]] | Seq[
      Tensor[Float16, _]
    ] | Seq[Tensor[Float,_]] | Seq[Tensor[Double, _]] | Seq[Tensor[String, _]] | Seq[Tensor[Boolean, _]] | Seq[
      Tensor[Complex[Float], _]
    ] | Seq[
      Tensor[Complex[Double], _]
    ]: Numeric, @sp I <: Int | Long: Numeric, @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
      Float
    ] | Complex[Double]: Numeric, Ax <: Axes](
        name: String,
        input_sequence: S,
        position: Tensor[I, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input_sequence, position)
      (callOp(name, "SequenceAt", allInputs, map))
    }
  }

  trait SequenceConstructV11 extends Operator {
    def SequenceConstructV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp S <: Seq[Tensor[UByte, _]] | Seq[Tensor[UShort, _]] | Seq[Tensor[UInt, _]] | Seq[
          Tensor[ULong, _]
        ] | Seq[
          Tensor[Byte, _]
        ] | Seq[Tensor[Short, _]] | Seq[Tensor[Int, _]] | Seq[Tensor[Long, _]] | Seq[Tensor[Float16, _]] | Seq[
          Tensor[Float,_]
        ] | Seq[Tensor[Double, _]] | Seq[Tensor[String, _]] | Seq[Tensor[Boolean, _]] | Seq[
          Tensor[Complex[Float]]
        ] | Seq[Tensor[Complex[Double]]]: Numeric
    , Ax <: Axes](name: String, inputs: Seq[Tensor[T, _]]): S = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple.fromArray(inputs.toArray).asInstanceOf[Tuple]
      (callOp(name, "SequenceConstruct", allInputs, map))
    }
  }

  trait SequenceEmptyV11 extends Operator {
    def SequenceEmptyV11[@sp S <: Seq[Tensor[UByte, _]] | Seq[Tensor[UShort, _]] | Seq[
      Tensor[UInt, _]
    ] | Seq[
      Tensor[ULong, _]
    ] | Seq[Tensor[Byte, _]] | Seq[Tensor[Short, _]] | Seq[Tensor[Int, _]] | Seq[Tensor[Long, _]] | Seq[
      Tensor[Float16, _]
    ] | Seq[Tensor[Float,_]] | Seq[Tensor[Double, _]] | Seq[Tensor[String, _]] | Seq[Tensor[Boolean, _]] | Seq[
      Tensor[Complex[Float]]
    ] | Seq[Tensor[Complex[Double]]]: Numeric, Ax <: Axes](
        name: String,
        dtype: Option[(Int)] = None
    ): S = {
      val map: Map[String, Any] = Map("dtype" -> dtype)
      val allInputs             = Seq()
      (callOp(name, "SequenceEmpty", allInputs, map))
    }
  }

  trait SequenceEraseV11 extends Operator {
    def SequenceEraseV11[@sp S <: Seq[Tensor[UByte, _]] | Seq[Tensor[UShort, _]] | Seq[
      Tensor[UInt, _]
    ] | Seq[
      Tensor[ULong, _]
    ] | Seq[Tensor[Byte, _]] | Seq[Tensor[Short, _]] | Seq[Tensor[Int, _]] | Seq[Tensor[Long, _]] | Seq[
      Tensor[Float16, _]
    ] | Seq[Tensor[Float,_]] | Seq[Tensor[Double, _]] | Seq[Tensor[String, _]] | Seq[Tensor[Boolean, _]] | Seq[
      Tensor[Complex[Float]]
    ] | Seq[Tensor[Complex[Double]]]: Numeric, @sp I <: Int | Long: Numeric, Ax <: Axes](
        name: String,
        input_sequence: S,
        position: Option[Tensor[I, _]] = None
    ): S = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input_sequence, position)
      (callOp(name, "SequenceErase", allInputs, map))
    }
  }

  trait SequenceInsertV11 extends Operator {
    def SequenceInsertV11[@sp S <: Seq[Tensor[UByte, _]] | Seq[Tensor[UShort, _]] | Seq[
      Tensor[UInt, _]
    ] | Seq[
      Tensor[ULong, _]
    ] | Seq[Tensor[Byte, _]] | Seq[Tensor[Short, _]] | Seq[Tensor[Int, _]] | Seq[Tensor[Long, _]] | Seq[
      Tensor[Float16, _]
    ] | Seq[Tensor[Float,_]] | Seq[Tensor[Double, _]] | Seq[Tensor[String, _]] | Seq[Tensor[Boolean, _]] | Seq[
      Tensor[Complex[Float]]
    ] | Seq[
      Tensor[Complex[Double]]
    ]: Numeric, @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
      Float
    ] | Complex[Double]: Numeric, @sp I <: Int | Long: Numeric, Ax <: Axes](
        name: String,
        input_sequence: S,
        tensor: Tensor[T, _],
        position: Option[Tensor[I, _]] = None
    ): S = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input_sequence, tensor, position)
      (callOp(name, "SequenceInsert", allInputs, map))
    }
  }

  trait SequenceLengthV11 extends Operator {
    def SequenceLengthV11[@sp S <: Seq[Tensor[UByte, _]] | Seq[Tensor[UShort, _]] | Seq[
      Tensor[UInt, _]
    ] | Seq[
      Tensor[ULong, _]
    ] | Seq[Tensor[Byte, _]] | Seq[Tensor[Short, _]] | Seq[Tensor[Int, _]] | Seq[Tensor[Long, _]] | Seq[
      Tensor[Float16, _]
    ] | Seq[Tensor[Float,_]] | Seq[Tensor[Double, _]] | Seq[Tensor[String, _]] | Seq[Tensor[Boolean, _]] | Seq[
      Tensor[Complex[Float], _]
    ] | Seq[Tensor[Complex[Double], _]]: Numeric, @sp I <: Long: Numeric, Ax <: Axes](
        name: String,
        input_sequence: S
    ): Tensor[I, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input_sequence)
      (callOp(name, "SequenceLength", allInputs, map))
    }
  }
*/
  trait ShapeV1 extends Operator {
    def ShapeV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T1 <: Long: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, data: Tensor[T, Ax]): Tensor[T1, Bx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(data)
      (callOp(name, "Shape", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait ShrinkV9 extends Operator {
    def ShrinkV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        bias: Option[(Float)] = None,
        lambd: Option[(Float)] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("bias" -> bias, "lambd" -> lambd)
      val allInputs             = Seq(input)
      (callOp(name, "Shrink", allInputs, map))
    }
  }
*/
  trait SigmoidV6 extends Operator {
    def SigmoidV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Sigmoid", allInputs, map))
    }
  }

  trait SigmoidV1 extends Operator {
    def SigmoidV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Sigmoid", allInputs, map))
    }
  }

  trait SignV9 extends Operator {
    def SignV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, input: Tensor[T, Ax]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Sign", allInputs, map))
    }
  }

  trait SinV7 extends Operator {
    def SinV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Sin", allInputs, map))
    }
  }

  trait SinhV9 extends Operator {
    def SinhV9[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Sinh", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait SizeV1 extends Operator {
    def SizeV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T1 <: Long: Numeric
    , Ax <: Axes](name: String, data: Tensor[T, _]): Tensor[T1,_] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(data)
      (callOp(name, "Size", allInputs, map))
    }
  }
*/
  //TODO: Constraint
  trait SliceV11 extends Operator {
    def SliceV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double],
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes, Ex <: Axes, Fx <: Axes](
        name: String,
        data: Tensor[T, Ax],
        starts: Tensor[Tind, Bx],
        ends: Tensor[Tind, Cx],
        axes: Option[Tensor[Tind, Dx]] = None,
        steps: Option[Tensor[Tind, Ex]] = None
    ): Tensor[T, Fx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(data, starts, ends, axes, steps)
      (callOp(name, "Slice", allInputs, map))
    }
  }

  trait SliceV10 extends Operator {
    def SliceV10[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes, Ex <: Axes, Fx <: Axes](
        name: String,
        data: Tensor[T, Ax],
        starts: Tensor[Tind, Bx],
        ends: Tensor[Tind, Cx],
        axes: Option[Tensor[Tind, Dx]] = None,
        steps: Option[Tensor[Tind, Ex]] = None
    ): Tensor[T, Fx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(data, starts, ends, axes, steps)
      (callOp(name, "Slice", allInputs, map))
    }
  }

  trait SliceV1 extends Operator {
    def SliceV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        ends: (Array[Int]),
        starts: (Array[Int]),
        data: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes, "ends" -> ends, "starts" -> starts)
      val allInputs             = Seq(data)
      (callOp(name, "Slice", allInputs, map))
    }
  }

  //Not supported, missing in ONNXJS
  /*
   //To consider restoring, need a loss function
  trait SoftmaxCrossEntropyLossV12 extends Operator {
    def SoftmaxCrossEntropyLossV12[
        @sp T <: Float16 | Float | Double: Numeric,
        @sp Tind <: Int | Long: Numeric
    , Ax <: Axes](
        name: String,
        reduction: Option[(String)] = None,
        scores: Tensor[T, _],
        labels: Tensor[Tind, _],
        weights: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("reduction" -> reduction)
      val allInputs             = Seq(scores, labels, weights)
      (callOp(name, "SoftmaxCrossEntropyLoss", allInputs, map))
    }
  }
*/
  trait SoftmaxV11 extends Operator {
    def SoftmaxV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "Softmax", allInputs, map))
    }
  }

  trait SoftmaxV1 extends Operator {
    def SoftmaxV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes, Bx <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, Ax]
    ): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(input)
      (callOp(name, "Softmax", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait SoftplusV1 extends Operator {
    def SoftplusV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Softplus", allInputs, map))
    }
  }

  trait SoftsignV1 extends Operator {
    def SoftsignV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Softsign", allInputs, map))
    }
  }

  trait SpaceToDepthV1 extends Operator {
    def SpaceToDepthV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, blocksize: (Int), input: Tensor[T, _]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("blocksize" -> blocksize)
      val allInputs             = Seq(input)
      (callOp(name, "SpaceToDepth", allInputs, map))
    }
  }


  //Sequence op, disabled
  trait SplitToSequenceV11 extends Operator {
    def SplitToSequenceV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp I <: Int | Long: Numeric,
        @sp S <: Seq[Tensor[UByte, _]] | Seq[Tensor[UShort, _]] | Seq[Tensor[UInt, _]] | Seq[
          Tensor[ULong, _]
        ] | Seq[
          Tensor[Byte, _]
        ] | Seq[Tensor[Short, _]] | Seq[Tensor[Int, _]] | Seq[Tensor[Long, _]] | Seq[Tensor[Float16, _]] | Seq[
          Tensor[Float,_]
        ] | Seq[Tensor[Double, _]] | Seq[Tensor[String, _]] | Seq[Tensor[Boolean, _]] | Seq[
          Tensor[Complex[Float]]
        ] | Seq[Tensor[Complex[Double]]]: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        input: Tensor[T, _],
        split: Option[Tensor[I, _]] = None
    ): S = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs             = Seq(input, split)
      (callOp(name, "SplitToSequence", allInputs, map))
    }
  }


  trait SplitV11 extends Operator {
    def SplitV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "splitAttr" -> splitAttr)
      val allInputs             = Seq(input)
      (callOp(name, "Split", allInputs, map))
    }
  }

  trait SplitV2 extends Operator {
    def SplitV2[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "splitAttr" -> splitAttr)
      val allInputs             = Seq(input)
      (callOp(name, "Split", allInputs, map))
    }
  }

  trait SplitV1 extends Operator {
    def SplitV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Tensor[T, _],
        split: Option[Tensor[T, _]] = None
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "splitAttr" -> splitAttr)
      val allInputs             = Seq(input, split)
      (callOp(name, "Split", allInputs, map))
    }
  }
*/
  trait SqrtV6 extends Operator {
    def SqrtV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(X)
      (callOp(name, "Sqrt", allInputs, map))
    }
  }

  trait SqrtV1 extends Operator {
    def SqrtV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(X)
      (callOp(name, "Sqrt", allInputs, map))
    }
  }

  //TODO: Constraint
  trait SqueezeV11 extends Operator {
    def SqueezeV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]
    , Ax <: Axes, Bx <: Axes](name: String, axes: Option[(Array[Int])] = None, data: Tensor[T, Ax]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs             = Seq(data)
      (callOp(name, "Squeeze", allInputs, map))
    }
  }

  trait SqueezeV1 extends Operator {
    def SqueezeV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, axes: Option[(Array[Int])] = None, data: Tensor[T, Ax]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs             = Seq(data)
      (callOp(name, "Squeeze", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait StringNormalizerV10 extends Operator {
    def StringNormalizerV10(
        name: String,
        case_change_action: Option[(String)] = None,
        is_case_sensitive: Option[(Int)] = None,
        locale: Option[(String)] = None,
        stopwords: Option[(Array[String])] = None,
        X: Tensor[String, _]
    ): Tensor[String, _] = {
      val map: Map[String, Any] = Map(
        "case_change_action" -> case_change_action,
        "is_case_sensitive"  -> is_case_sensitive,
        "locale"             -> locale,
        "stopwords"          -> stopwords
      )
      val allInputs = Seq(X)
      (callOp(name, "StringNormalizer", allInputs, map))
    }
  }
*/
  trait SubV7 extends Operator {
    def SubV7[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Sub", allInputs, map))
    }
  }

  trait SubV6 extends Operator {
    def SubV6[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Sub", allInputs, map))
    }
  }

  trait SubV1 extends Operator {
    def SubV1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "broadcast" -> broadcast, "consumed_inputs" -> consumed_inputs)
      val allInputs = Seq(A,B)
      (callOp(name, "Sub", allInputs, map))
    }
  }

  trait SumV8 extends Operator {
    def SumV8[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        data_0: Seq[Tensor[T, Ax]]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = data_0 
      (callOp(name, "Sum", allInputs, map))
    }
  }

  trait SumV6 extends Operator {
    def SumV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        data_0: Seq[Tensor[T, Ax]]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = data_0 
      (callOp(name, "Sum", allInputs, map))
    }
  }

  trait SumV1 extends Operator {
    def SumV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        data_0: Seq[Tensor[T, Ax]]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = data_0 
      (callOp(name, "Sum", allInputs, map))
    }
  }

  trait TanV7 extends Operator {
    def TanV7[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Tan", allInputs, map))
    }
  }

  trait TanhV6 extends Operator {
    def TanhV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input)
      (callOp(name, "Tanh", allInputs, map))
    }
  }

  trait TanhV1 extends Operator {
    def TanhV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs             = Seq(input)
      (callOp(name, "Tanh", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait TfIdfVectorizerV9 extends Operator {
    def TfIdfVectorizerV9[
        @sp T <: String | Int | Long: Numeric,
        @sp T1 <: Float: Numeric
    , Ax <: Axes](
        name: String,
        max_gram_length: (Int),
        max_skip_count: (Int),
        min_gram_length: (Int),
        mode: (String),
        ngram_counts: (Array[Int]),
        ngram_indexes: (Array[Int]),
        pool_int64s: Option[(Array[Int])] = None,
        pool_strings: Option[(Array[String])] = None,
        weights: Option[(Array[Float])] = None,
        X: Tensor[T, _]
    ): Tensor[T1,_] = {
      val map: Map[String, Any] = Map(
        "max_gram_length" -> max_gram_length,
        "max_skip_count"  -> max_skip_count,
        "min_gram_length" -> min_gram_length,
        "mode"            -> mode,
        "ngram_counts"    -> ngram_counts,
        "ngram_indexes"   -> ngram_indexes,
        "pool_int64s"     -> pool_int64s,
        "pool_strings"    -> pool_strings,
        "weights"         -> weights
      )
      val allInputs = Seq(X)
      (callOp(name, "TfIdfVectorizer", allInputs, map))
    }
  }
*/
  //Not supported, missing from ONNXJS
  /*
  trait ThresholdedReluV10 extends Operator {
    def ThresholdedReluV10[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        X: Tensor[T, Ax]
    ): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("alpha" -> alpha)
      val allInputs             = Seq(X)
      (callOp(name, "ThresholdedRelu", allInputs, map))
    }
  }
*/
  trait TileV6 extends Operator {
    def TileV6[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T1 <: Long: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes](name: String, input: Tensor[T, Ax], repeats: Tensor[T1, Bx]): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input, repeats)
      (callOp(name, "Tile", allInputs, map))
    }
  }

  trait TileV1 extends Operator {
    def TileV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes, Cx <: Axes, Dx <: Axes](name: String, input: Tensor[T, Ax], tiles: Tensor[T, Bx], axis: Tensor[T, Cx]): Tensor[T, Dx] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(input, tiles, axis)
      (callOp(name, "Tile", allInputs, map))
    }
  }

  //Not supported, missing from ONNXJS
  /*
  trait TopKV11 extends Operator {
    def TopKV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp I <: Long: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        largest: Option[(Int)] = None,
        sorted: Option[(Int)] = None,
        X: Tensor[T, _],
        K: Tensor[Long, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "largest" -> largest, "sorted" -> sorted)
      val allInputs             = Seq(X, K)
      (callOp(name, "TopK", allInputs, map))
    }
  }

  trait TopKV10 extends Operator {
    def TopKV10[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp I <: Long: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        X: Tensor[T, _],
        K: Tensor[Long, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Seq(X, K)
      (callOp(name, "TopK", allInputs, map))
    }
  }

  trait TopKV1 extends Operator {
    def TopKV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric,
        @sp I <: Long: Numeric
    , Ax <: Axes](name: String, axis: Option[(Int)] = None, k: (Int), X: Tensor[T, _]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "k" -> k)
      val allInputs             = Seq(X)
      (callOp(name, "TopK", allInputs, map))
    }
  }
*/
  //TODO: Constraint
  trait TransposeV1 extends Operator {
    def TransposeV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]
    , Ax <: Axes, Bx <: Axes](name: String, perm: Option[(Array[Int])] = None, data: Tensor[T, Ax]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("perm" -> perm)
      val allInputs             = Seq(data)
      (callOp(name, "Transpose", allInputs, map))
    }
  }

  //Not supported, ONNX ML
  /*
  trait TreeEnsembleClassifierV1 extends Operator {
    def TreeEnsembleClassifierV1[
        @sp T1 <: Float | Double | Long | Int: Numeric,
        @sp T2 <: String | Long: Numeric
    , Ax <: Axes](
        name: String,
        base_values: Option[(Array[Float])] = None,
        class_ids: Option[(Array[Int])] = None,
        class_nodeids: Option[(Array[Int])] = None,
        class_treeids: Option[(Array[Int])] = None,
        class_weights: Option[(Array[Float])] = None,
        classlabels_int64s: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        nodes_falsenodeids: Option[(Array[Int])] = None,
        nodes_featureids: Option[(Array[Int])] = None,
        nodes_hitrates: Option[(Array[Float])] = None,
        nodes_missing_value_tracks_true: Option[(Array[Int])] = None,
        nodes_modes: Option[(Array[String])] = None,
        nodes_nodeids: Option[(Array[Int])] = None,
        nodes_treeids: Option[(Array[Int])] = None,
        nodes_truenodeids: Option[(Array[Int])] = None,
        nodes_values: Option[(Array[Float])] = None,
        post_transform: Option[(String)] = None,
        X: Tensor[T1,_]
    ): Tensor[T2, _] = {
      val map: Map[String, Any] = Map(
        "base_values"                     -> base_values,
        "class_ids"                       -> class_ids,
        "class_nodeids"                   -> class_nodeids,
        "class_treeids"                   -> class_treeids,
        "class_weights"                   -> class_weights,
        "classlabels_int64s"              -> classlabels_int64s,
        "classlabels_strings"             -> classlabels_strings,
        "nodes_falsenodeids"              -> nodes_falsenodeids,
        "nodes_featureids"                -> nodes_featureids,
        "nodes_hitrates"                  -> nodes_hitrates,
        "nodes_missing_value_tracks_true" -> nodes_missing_value_tracks_true,
        "nodes_modes"                     -> nodes_modes,
        "nodes_nodeids"                   -> nodes_nodeids,
        "nodes_treeids"                   -> nodes_treeids,
        "nodes_truenodeids"               -> nodes_truenodeids,
        "nodes_values"                    -> nodes_values,
        "post_transform"                  -> post_transform
      )
      val allInputs = Seq(X)
      (callOp(name, "TreeEnsembleClassifier", allInputs, map))
    }
  }

  trait TreeEnsembleRegressorV1 extends Operator {
    def TreeEnsembleRegressorV1[@sp T <: Float | Double | Long | Int: Numeric, Ax <: Axes](
        name: String,
        aggregate_function: Option[(String)] = None,
        base_values: Option[(Array[Float])] = None,
        n_targets: Option[(Int)] = None,
        nodes_falsenodeids: Option[(Array[Int])] = None,
        nodes_featureids: Option[(Array[Int])] = None,
        nodes_hitrates: Option[(Array[Float])] = None,
        nodes_missing_value_tracks_true: Option[(Array[Int])] = None,
        nodes_modes: Option[(Array[String])] = None,
        nodes_nodeids: Option[(Array[Int])] = None,
        nodes_treeids: Option[(Array[Int])] = None,
        nodes_truenodeids: Option[(Array[Int])] = None,
        nodes_values: Option[(Array[Float])] = None,
        post_transform: Option[(String)] = None,
        target_ids: Option[(Array[Int])] = None,
        target_nodeids: Option[(Array[Int])] = None,
        target_treeids: Option[(Array[Int])] = None,
        target_weights: Option[(Array[Float])] = None,
        X: Tensor[T, _]
    ): Tensor[Float,_] = {
      val map: Map[String, Any] = Map(
        "aggregate_function"              -> aggregate_function,
        "base_values"                     -> base_values,
        "n_targets"                       -> n_targets,
        "nodes_falsenodeids"              -> nodes_falsenodeids,
        "nodes_featureids"                -> nodes_featureids,
        "nodes_hitrates"                  -> nodes_hitrates,
        "nodes_missing_value_tracks_true" -> nodes_missing_value_tracks_true,
        "nodes_modes"                     -> nodes_modes,
        "nodes_nodeids"                   -> nodes_nodeids,
        "nodes_treeids"                   -> nodes_treeids,
        "nodes_truenodeids"               -> nodes_truenodeids,
        "nodes_values"                    -> nodes_values,
        "post_transform"                  -> post_transform,
        "target_ids"                      -> target_ids,
        "target_nodeids"                  -> target_nodeids,
        "target_treeids"                  -> target_treeids,
        "target_weights"                  -> target_weights
      )
      val allInputs = Seq(X)
      (callOp(name, "TreeEnsembleRegressor", allInputs, map))
    }
  }
*/

  //Not supported - ?
  /*
  trait UnfoldToDepthV12 extends Operator {
    def UnfoldToDepthV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        block_size: (Array[Int]),
        dilations: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "block_size" -> block_size,
        "dilations"  -> dilations,
        "pads"       -> pads,
        "strides"    -> strides
      )
      val allInputs = Seq(X)
      (callOp(name, "UnfoldToDepth", allInputs, map))
    }
  }
*/
  //Not supported, missing in ONNXJS
  /*
  trait UniqueV11 extends Operator {
    def UniqueV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        sorted: Option[(Int)] = None,
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "sorted" -> sorted)
      val allInputs             = Seq(X)
      (callOp(name, "Unique", allInputs, map))
    }
  }
*/
  trait UnsqueezeV11 extends Operator {
    def UnsqueezeV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, axes: (Array[Int]), data: Tensor[T, Ax]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs             = Seq(data)
      (callOp(name, "Unsqueeze", allInputs, map))
    }
  }

  trait UnsqueezeV1 extends Operator {
    def UnsqueezeV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes, Bx <: Axes](name: String, axes: (Array[Int]), data: Tensor[T, Ax]): Tensor[T, Bx] = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs             = Seq(data)
      (callOp(name, "Unsqueeze", allInputs, map))
    }
  }

  //Not supported, Deprecated
  /*
  trait UpsampleV10 extends Operator {
    def UpsampleV10[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        mode: Option[(String)] = None,
        X: Tensor[T, _],
        scales: Tensor[Float,_]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs             = Seq(X, scales)
      (callOp(name, "Upsample", allInputs, map))
    }
  }

  trait UpsampleV9 extends Operator {
    def UpsampleV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        mode: Option[(String)] = None,
        X: Tensor[T, _],
        scales: Tensor[Float,_]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs             = Seq(X, scales)
      (callOp(name, "Upsample", allInputs, map))
    }
  }

  trait UpsampleV7 extends Operator {
    def UpsampleV7[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        mode: Option[(String)] = None,
        scaleAttrs: (Array[Float]),
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map("mode" -> mode, "scaleAttrs" -> scaleAttrs)
      val allInputs             = Seq(X)
      (callOp(name, "Upsample", allInputs, map))
    }
  }

  trait UpsampleV1 extends Operator {
    def UpsampleV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](
        name: String,
        height_scaleAttr: (Float),
        mode: Option[(String)] = None,
        width_scaleAttr: (Float),
        X: Tensor[T, _]
    ): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "height_scaleAttr" -> height_scaleAttr,
        "mode"             -> mode,
        "width_scaleAttr"  -> width_scaleAttr
      )
      val allInputs = Seq(X)
      (callOp(name, "Upsample", allInputs, map))
    }
  }
*/
  //Not supported, missing from ONNXJS
  /*
  trait WhereV9 extends Operator {
    def WhereV9[
        @sp B <: Boolean,
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, condition: Tensor[B, _], X: Tensor[T, _], Y: Tensor[T, _]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(condition, X, Y)
      (callOp(name, "Where", allInputs, map))
    }
  }
*/

  trait XorV7 extends Operator {
    def XorV7[@sp T <: Boolean, @sp T1 <: Boolean, Ax <: Axes](
        name: String,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Seq(A,B)
      (callOp(name, "Xor", allInputs, map))
    }
  }

  trait XorV1 extends Operator {
    def XorV1[@sp T <: Boolean, @sp T1 <: Boolean, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Tensor[T, Ax],
        B: Tensor[T, Ax]
    ): Tensor[T1, Ax] = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs             = Seq(A,B)
      (callOp(name, "Xor", allInputs, map))
    }
  }
  //Not supported, ONNX ML
  /*
  trait ZipMapV1 extends Operator {
    def ZipMapV1[@sp T <: Seq[Map[String, Float]] | Seq[Map[Long, Float]]: Numeric, Ax <: Axes](
        name: String,
        classlabels_int64s: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        X: Tensor[Float,_]
    ): T = {
      val map: Map[String, Any] = Map(
        "classlabels_int64s"  -> classlabels_int64s,
        "classlabels_strings" -> classlabels_strings
      )
      val allInputs = Seq(X)
      (callOp(name, "ZipMap", allInputs, map))
    }
  }
*/
}
