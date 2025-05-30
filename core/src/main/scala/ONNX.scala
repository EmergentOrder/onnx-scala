package org.emergentorder

import org.emergentorder.compiletime.TensorShapeDenotation.Reverse
import org.emergentorder.compiletime._
import org.emergentorder.onnx.Tensors._
import spire.math.Complex
import spire.math.Numeric
import spire.math.UByte
import spire.math.UInt
import spire.math.ULong
import spire.math.UShort

import scala.collection.immutable.ArraySeq
import scala.language.higherKinds
import scala.{specialized => sp}

import io.kjaer.compiletime._
import io.kjaer.compiletime.Shape.NumElements

//TODO: Add new Trilu operator from V14, other "function" operators, as need be

//ONNX domain: ai.onnx(default)
//Only the ops which are supported in both ONNX Runtime and ONNX.js
//See: https://github.com/onnx/onnx/blob/v1.8.1/docs/Operators.md#aionnx-default
//Also: https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md
//Also: https://github.com/microsoft/onnxjs/blob/master/docs/operators.md
//Tests currently live one level up at: https://github.com/SciScala/NDScala/blob/master/ONNXScala/src/test/scala/ndscala/ONNXScalaNDArraySpec.scala
package object onnx {
   // TODO P2: Symbolic shape values
   // TODO P2: Support bfloat16 type (new in ONNX 1.8.0)
   // TODO P3: Encode node names as types

   // Note: Indices should at least optionally be longs, but currently compiletime ops in Scala 3 are only defined for ints
   // Note: Broadcasting is not supported, by design
   sealed trait Operator {
      def callOp[
          T <: Supported,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String, opName: String, inputs: Tuple, attrs: Map[String, Any])(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]]
   }

   // TODO: restore onnxbytes here
   abstract class Model() extends Operator {
      def fullModel[
          T <: Supported,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          inputs: Tuple
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]]
   }

   // Not in the spec, allows access to params from within the loaded model
   trait DataSource {
      def getParams[
          T <: Supported,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String)(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]]
   }

   trait AbsV13 extends Operator {
      def AbsV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String, X: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Abs", allInputs, map))
      }
   }

   trait AcosV7 extends Operator {
      def AcosV7[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Acos", allInputs, map))
      }
   }

   trait AcoshV9 extends Operator {
      def AcoshV9[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Acosh", allInputs, map))
      }
   }

   trait AddV14 extends Operator {
      def AddV14[
          @sp T <: UByte | Byte | UShort | Short | UInt | ULong | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Add", allInputs, map))
      }
   }

   trait AndV7 extends Operator {
      def AndV7[
          @sp T <: Boolean,
          @sp T1 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T1, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "And", allInputs, map))
      }
   }

   trait AsinV7 extends Operator {
      def AsinV7[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Asin", allInputs, map))
      }
   }

   trait AsinhV9 extends Operator {
      def AsinhV9[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Asinh", allInputs, map))
      }
   }

   trait AtanV7 extends Operator {
      def AtanV7[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Atan", allInputs, map))
      }
   }

   trait AtanhV9 extends Operator {
      def AtanhV9[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Atanh", allInputs, map))
      }
   }

   // Contrained to 2d image, means 4d tensor.
   // output shape: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
   // TODO: handle pads, strides
   trait AveragePoolV11 extends Operator {
      def AveragePoolV11[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: Dimension #: SNil,
          PadsBefore <: None.type | (Dimension #: Dimension #: SNil),
          PadsAfter <: None.type | (Dimension #: Dimension #: SNil)
      ](
          name: String,
          auto_pad: String = "NOTSET",
          ceil_mode: Int = 0,
          count_include_pad: Int = 0,
          kernel_shape: S1,
          padsBefore: PadsBefore = None,
          padsAfter: PadsAfter = None,
          strides: Option[(Array[Int])] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[PaddedShape[PoolShape[S, S1], PadsBefore, PadsAfter]],
          s1: ShapeOf[S1]
      ): Tensor[T, Tuple3[Tt1, Td1, PaddedShape[PoolShape[S, S1], PadsBefore, PadsAfter]]] = {
         val padsB: Array[Int] = padsBefore match {
            case x: Shape => x.toSeq.toArray
            case None     => Array.fill(shapeOf[S1].toSeq.size)(0)
         }

         val padsA: Array[Int] = padsAfter match {
            case x: Shape => x.toSeq.toArray
            case None     => Array.fill(shapeOf[S1].toSeq.size)(0)
         }

         val map: Map[String, Any] = Map(
           "auto_pad"          -> auto_pad,
           "ceil_mode"         -> ceil_mode,
           "count_include_pad" -> count_include_pad,
           "kernel_shape"      -> kernel_shape.toSeq.toArray,
           "pads"              -> (padsB ++ padsA),
           "strides"           -> strides
         )
         val allInputs = Tuple1(X)
         (callOp(name, "AveragePool", allInputs, map))
      }
   }

   // Missing optional outputs, only needed for training mode
   trait BatchNormalizationV15 extends Operator {
      def BatchNormalizationV15[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          N <: Dimension,
          C <: Dimension,
          H <: Dimension,
          W <: Dimension,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: N #: C #: H #: W #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: C #: SNil,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation
      ](
          name: String,
          epsilon: Float = 1e-05,
          momentum: Float = 0.9,
          training_mode: Int = 0,
          X: Tensor[T, Tuple3[Tt, Td, S]],
          scale: Tensor[T, Tuple3[Tt1, Td1, S1]],
          B: Tensor[T, Tuple3[Tt1, Td1, S1]],
          input_mean: Tensor[T, Tuple3[Tt1, Td1, S1]],
          input_var: Tensor[T, Tuple3[Tt1, Td1, S1]]
      )(using
          tt: ValueOf[Tt2],
          td: TensorShapeDenotationOf[Td2],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt2, Td2, S]] = {
         val map: Map[String, Any] = Map("epsilon" -> epsilon, "momentum" -> momentum)
         val allInputs             = Tuple5(X, scale, B, input_mean, input_var)
         (callOp(name, "BatchNormalization", allInputs, map))
      }
   }

   // TODO:
   // New op: Bernoulli - Since opset 15
   // New ops: Bitwise (And, Not, Or, Xor) - since opset 18
   // New Op: Blackman window - since opset 17

   /* - Not supported - cast on the JVM side
  //Missing in NDScala P2 - needs match type from data type to int
  trait CastV13 extends Operator {
    def CastV9[
        @sp T1 <: BFloat16 | Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String: Numeric,
        @sp T2 <: BFloat16 | Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String: Numeric
    , Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](name: String, input: Tensor[T1,Tuple3[Tt, Td, S]])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, Tuple3[Tt, Td, S]] = {
      val map: Map[String, Any] = Map("to" -> to)
      val allInputs             = Tuple1(input)
      (callOp(name, "Cast", allInputs, map))
    }
  }
    */

   // Missing (unsupported) op: CastLike - since opset 15

   trait CeilV13 extends Operator {
      def CeilV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Ceil", allInputs, map))
      }
   }

   trait CeluV12 extends Operator {
      def CeluV12[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          alpha: Float = 1.0,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map("alpha" -> alpha)
         val allInputs             = Tuple1(X)
         (callOp(name, "Celu", allInputs, map))
      }
   }

   // TODO:
   // New op:  CenterCropPad - since opset 18
   // New op: Col2lm - since opset 18
   //
   // FIXME: "All input tensors must have the same shape, except for the dimension size of the axis to concatenate on". Currently assumes tensors to be concated are same shape except the leading dimension
   // TODO P1: Arbitrary arity inputs
   trait ConcatV13 extends Operator {
      def ConcatV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double],
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          SSuffix <: Shape,
          S <: Dimension #: SSuffix,
          S1 <: Dimension #: SSuffix,
          Axis <: Index ::: INil
      ](
          name: String,
          axis: Axis,
          inputs: Tuple2[Tensor[T, Tuple3[Tt, Td, S]], Tensor[T, Tuple3[Tt, Td, S1]]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[AddGivenAxisSize[S, S1, Axis]],
          i: IndicesOf[Axis]
      ): Tensor[T, Tuple3[Tt, Td, AddGivenAxisSize[S, S1, Axis]]] = {
         val map: Map[String, Any] = Map("axis" -> indicesOf[Axis].indices.toArray.head)
         val allInputs             = Tuple.fromArray(inputs.toArray)
         (callOp(name, "Concat", allInputs, map))
      }
   }

   // TODO: remove the need to pass kernel_shape, it can be inferred
   // Limited to 1 feature map, 1 group, stride 1
   trait ConvV11 extends Operator {
      def ConvV11[
          @sp T <: Float16 | Float | Double: Numeric,
          N <: Dimension,
          C <: Dimension,
          H <: Dimension,
          W <: Dimension,
          KH <: Dimension,
          KW <: Dimension,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: N #: C #: H #: W #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: 1 #: C #: KH #: KW #: SNil,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation,
          S2 <: 1 #: SNil,
          Tt3 <: TensorTypeDenotation,
          Td3 <: TensorShapeDenotation,
          S3 <: KH #: KW #: SNil,
          PadsBefore <: None.type | (Dimension #: Dimension #: SNil),
          PadsAfter <: None.type | (Dimension #: Dimension #: SNil)
      ](
          name: String,
          auto_pad: String = "NOTSET",
          dilations: Option[(Array[Int])] = None,
          group: Int = 1,
          kernel_shape: S3,
          padsBefore: PadsBefore = None,
          padsAfter: PadsAfter = None,
          strides: Option[(Array[Int])] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]],
          W: Tensor[T, Tuple3[Tt1, Td1, S1]],
          B: Option[Tensor[T, Tuple3[Tt2, Td2, S2]]] = None
      )(using
          tt: ValueOf[Tt3],
          td: TensorShapeDenotationOf[Td3],
          s: ShapeOf[PaddedShape[PoolShape[S, S3], PadsBefore, PadsAfter]],
          s3: ShapeOf[S3]
      ): Tensor[T, Tuple3[Tt3, Td3, PaddedShape[PoolShape[S, S3], PadsBefore, PadsAfter]]] = {
         val padsB: Array[Int] = padsBefore match {
            case x: Shape => x.toSeq.toArray
            case None     => Array.fill(shapeOf[S3].toSeq.size)(0)
         }

         val padsA: Array[Int] = padsAfter match {
            case x: Shape => x.toSeq.toArray
            case None     => Array.fill(shapeOf[S3].toSeq.size)(0)
         }

         val map: Map[String, Any] = Map(
           "auto_pad"     -> auto_pad,
           "dilations"    -> dilations,
           "group"        -> group,
           "kernel_shape" -> kernel_shape.toSeq.toArray,
           "pads"         -> (padsB ++ padsA),
           "strides"      -> strides
         )
         val allInputs = Tuple3(X, W, B)
         (callOp(name, "Conv", allInputs, map))
      }
   }

   trait CosV7 extends Operator {
      def CosV7[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Cos", allInputs, map))
      }
   }

   trait CoshV9 extends Operator {
      def CoshV9[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Cosh", allInputs, map))
      }
   }

   // TODO:
   // New op: DFT - since opset 17
   trait DivV14 extends Operator {
      def DivV14[
          @sp T <: UByte | Byte | UShort | Short | UInt | ULong | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Div", allInputs, map))
      }
   }

   // Missing optional second output
   trait DropoutV13 extends Operator {
      def DropoutV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          @sp T1 <: Float16 | Float | Double | Boolean,
          @sp T2 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Shape,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation,
          S2 <: Shape
      ](
          name: String,
          seed: Int = 42,
          data: Tensor[T, Tuple3[Tt, Td, S]],
          ratio: Tensor[T1, Tuple3[Tt1, Td1, S1]] = Tensor(Array(0.5f), SNil),
          training_mode: Tensor[T2, Tuple3[Tt2, Td2, S2]] = Tensor(Array(false), SNil)
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map("seed" -> seed)
         val allInputs             = Tuple3(data, ratio, training_mode)
         (callOp(name, "Dropout", allInputs, map))
      }
   }

   trait EluV6 extends Operator {
      def EluV6[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          alpha: Float = 1.0,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map("alpha" -> alpha)
         val allInputs             = Tuple1(X)
         (callOp(name, "Elu", allInputs, map))
      }
   }

   trait EqualV13 extends Operator {
      def EqualV13[
          @sp T <: Boolean | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 |
             Float16 | Float | Double,
          @sp T1 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation
      ](name: String, A: Tensor[T, Tuple3[Tt, Td, S]], B: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S]
      ): Tensor[T1, Tuple3[Tt1, Td1, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Equal", allInputs, map))
      }
   }

   trait ExpV13 extends Operator {
      def ExpV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Exp", allInputs, map))
      }
   }

   // Missing constraint - need an equivalent of the size equality constraint on Squeeze, but that asserts the shapes are broadcastable
   // Explicit broadcasting - can fail
   trait ExpandV13 extends Operator {
      def ExpandV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double],
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Shape,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation,
          S2 <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]],
          shapeInput: Tensor[Long, Tuple3[Tt1, Td1, S1]]
      )(using
          tt: ValueOf[Tt2],
          td: TensorShapeDenotationOf[Td2],
          s: ShapeOf[S2]
      ): Tensor[T, Tuple3[Tt2, Td2, S2]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(input, shapeInput)
         (callOp(name, "Expand", allInputs, map))
      }
   }

   trait FlattenV13 extends Operator {
      def FlattenV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double]: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axis <: Index ::: INil
      ](name: String, axis: Axis, input: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[FlattenedShape[S, Axis]]
      ): Tensor[T, Tuple3[Tt1, Td, FlattenedShape[S, Axis]]] = {
         val map: Map[String, Any] = Map("axis" -> axis.indices.toArray.head)
         val allInputs             = Tuple1(input)
         (callOp(name, "Flatten", allInputs, map))
      }
   }

   trait FloorV13 extends Operator {
      def FloorV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Floor", allInputs, map))
      }
   }
   // Missing in NDScala - P3
   // need a match type
   trait GatherV13 extends Operator {
      def GatherV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double],
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation,
          AxisIndex <: Index ::: INil,
          AxisIndices <: Indices,
          IndicesSize <: Index
      ](
          name: String,
          axis: AxisIndex = 0 ::: INil,
          data: Tensor[T, Tuple3[Tt, Td, S]],
          indices: AxisIndices
//          indicesSize: IndicesSize[AxisIndices]
      )(using
          tt: ValueOf[Tt2],
          td: TensorShapeDenotationOf[Td2],
          s: ShapeOf[GatheredShape[S, AxisIndex, AxisIndices, IndicesSize]],
          i: IndicesOf[AxisIndex],
          i2: IndicesOf[AxisIndices]
      ): Tensor[T, Tuple3[Tt2, Td2, GatheredShape[S, AxisIndex, AxisIndices, IndicesSize]]] = {
         val map: Map[String, Any] = Map("axis" -> indicesOf[AxisIndex].indices.toArray.head)
         val allInputs = Tuple2(
           data,
           Tensor(
             indicesOf[AxisIndices].indices.toArray,
             indicesOf[AxisIndices].indices.toArray.size
                .asInstanceOf[io.kjaer.compiletime.Dimension] #: SNil
           )
         )
         (callOp(name, "Gather", allInputs, map))
      }
   }

   // Bug in ORT where the bias tensor C should be optional, but is in fact required
   // See: https://github.com/microsoft/onnxruntime/issues/6423
   trait GemmV13 extends Operator {
      def GemmV13[
          @sp T <: BFloat16 | Float16 | Float | Double | UInt | ULong | Int | Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          M <: Dimension,
          K <: Dimension,
          S <: M #: K #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          N <: Dimension,
          S1 <: K #: N #: SNil
      ](
          name: String,
          alpha: Float = 1.0,
          beta: Float = 1.0,
          transA: Int = 0,
          transB: Int = 0,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt1, Td1, S1]],
          C: Option[Tensor[T, Tuple3[Tt, Td, M #: N #: SNil]]] = None
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[M #: N #: SNil]
      ): Tensor[T, Tuple3[Tt, Td, M #: N #: SNil]] = {
         val map: Map[String, Any] =
            Map("alpha" -> alpha, "beta" -> beta, "transA" -> transA, "transB" -> transB)
         val allInputs = Tuple3(A, B, C)
         (callOp(name, "Gemm", allInputs, map))
      }
   }

   trait GlobalAveragePoolV1 extends Operator {
      def GlobalAveragePoolV1[
          @sp T <: Float16 | Float | Double: Numeric,
          N <: Dimension,
          C <: Dimension,
          H <: Dimension,
          W <: Dimension,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: N #: C #: H #: W #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: N #: C #: 1 #: 1 #: SNil
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[T, Tuple3[Tt1, Td1, S1]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "GlobalAveragePool", allInputs, map))
      }
   }

   trait GlobalMaxPoolV1 extends Operator {
      def GlobalMaxPoolV1[
          @sp T <: Float16 | Float | Double: Numeric,
          N <: Dimension,
          C <: Dimension,
          H <: Dimension,
          W <: Dimension,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: N #: C #: H #: W #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: N #: C #: 1 #: 1 #: SNil
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[T, Tuple3[Tt1, Td1, S1]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "GlobalMaxPool", allInputs, map))
      }
   }

   // TODO P2: Contrained to 2d image, means 4d tensor.
   // Consider enforcing denotations - NCHW
   trait InstanceNormalizationV6 extends Operator {
      def InstanceNormalizationV6[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: SNil,
          Tt2 <: TensorTypeDenotation
      ](
          name: String,
          epsilon: Float = 1e-5,
          input: Tensor[T, Tuple3[Tt, Td, S]],
          scale: Tensor[T, Tuple3[Tt1, Td1, S1]],
          B: Tensor[T, Tuple3[Tt1, Td1, S1]]
      )(using
          tt: ValueOf[Tt2],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt2, Td, S]] = {
         val map: Map[String, Any] = Map("epsilon" -> epsilon)
         val allInputs             = Tuple3(input, scale, B)
         (callOp(name, "InstanceNormalization", allInputs, map))
      }
   }

   trait IsNaNV13 extends Operator {
      def IsNaNV13[
          @sp T1 <: BFloat16 | Float16 | Float | Double: Numeric,
          @sp T2 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String, X: Tensor[T1, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T2, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "IsNaN", allInputs, map))
      }
   }

   // TODO P2: Contrained to 2d image, means 4d tensor.
   // Consider enforcing denotations - NCHW
   trait LRNV13 extends Operator {
      def LRNV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation
      ](
          name: String,
          alpha: Float = 0.0001,
          beta: Float = 0.75,
          bias: Float = 1.0,
          size: Int,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt1, Td, S]] = {
         val map: Map[String, Any] =
            Map("alpha" -> alpha, "beta" -> beta, "bias" -> bias, "size" -> size)
         val allInputs = Tuple1(X)
         (callOp(name, "LRN", allInputs, map))
      }
   }

   trait LeakyReluV6 extends Operator {
      def LeakyReluV6[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          alpha: Float = 0.01,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map("alpha" -> alpha)
         val allInputs             = Tuple1(X)
         (callOp(name, "LeakyRelu", allInputs, map))
      }
   }

   trait LessV13 extends Operator {
      def LessV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          @sp T1 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation
      ](name: String, A: Tensor[T, Tuple3[Tt, Td, S]], B: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S]
      ): Tensor[T1, Tuple3[Tt1, Td1, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Less", allInputs, map))
      }
   }

   // TODO:
   // New Op: LayerNormalization - since opset 17
   trait LogV13 extends Operator {
      def LogV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Log", allInputs, map))
      }
   }

   trait MatMulV13 extends Operator {
      def MatMulV13[
          @sp T <: BFloat16 | Float16 | Float | Double | UInt | ULong | Int | Long: Numeric,
          Dim0 <: Dimension,
          Dim1 <: Dimension,
          Dim2 <: Dimension,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dim0 #: Dim1 #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dim1 #: Dim2 #: SNil
      ](
          name: String,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt1, Td1, S1]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[Dim0 #: Dim2 #: SNil],
          vd0: ValueOf[scala.compiletime.ops.int.S[Dim0]],
          vd1: ValueOf[scala.compiletime.ops.int.S[Dim1]],
          vd2: ValueOf[scala.compiletime.ops.int.S[Dim2]]
      )
      // ,vd0:ValueOf[scala.compiletime.S[Dim0]],vd1:ValueOf[scala.compiletime.S[Dim1]], vd2: ValueOf[scala.compiletime.S[Dim2]])
          : Tensor[T, Tuple3[Tt, Td, Dim0 #: Dim2 #: SNil]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "MatMul", allInputs, map))
      }
   }

   // TODO : bring up to date, missing V11, V12 changes
   // TODO P2: Contrained to 2d image, means 4d tensor.
   // output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
   // TODO: pads, strides, dilations
   trait MaxPoolV12 extends Operator {
      def MaxPoolV12[
          @sp T <: Float16 | Float | Double | Byte | UByte: Numeric,
          @sp I <: Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: Dimension #: SNil,
          PadsBefore <: None.type | (Dimension #: Dimension #: SNil),
          PadsAfter <: None.type | (Dimension #: Dimension #: SNil)
      ](
          name: String,
          auto_pad: String = "NOTSET",
          ceil_mode: Int = 0,
          dilations: Option[(Array[Int])] = None,
          kernel_shape: S1,
          padsBefore: PadsBefore = None,
          padsAfter: PadsAfter = None,
          storage_order: Int = 0,
          strides: Option[(Array[Int])] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[PaddedShape[PoolShape[S, S1], PadsBefore, PadsAfter]],
          s1: ShapeOf[S1]
      ): Tensor[T, Tuple3[Tt1, Td1, PaddedShape[PoolShape[S, S1], PadsBefore, PadsAfter]]] = {
         val padsB: Array[Int] = padsBefore match {
            case x: Shape => x.toSeq.toArray
            case None     => Array.fill(shapeOf[S1].toSeq.size)(0)
         }

         val padsA: Array[Int] = padsAfter match {
            case x: Shape => x.toSeq.toArray
            case None     => Array.fill(shapeOf[S1].toSeq.size)(0)
         }

         val map: Map[String, Any] = Map(
           "auto_pad"      -> auto_pad,
           "ceil_mode"     -> ceil_mode,
           "dilations"     -> dilations,
           "kernel_shape"  -> kernel_shape.toSeq.toArray,
           "pads"          -> (padsB ++ padsA),
           "storage_order" -> storage_order,
           "strides"       -> strides
         )
         val allInputs = Tuple1(X)
         (callOp(name, "MaxPool", allInputs, map))
      }
   }

   trait MulV14 extends Operator {
      def MulV14[
          @sp T <: UByte | Byte | UShort | Short | UInt | ULong | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Mul", allInputs, map))
      }
   }

   trait NegV13 extends Operator {
      def NegV13[
          @sp T <: Float | Int | Byte | Short | Long | BFloat16 | Float16 | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Neg", allInputs, map))
      }
   }

   trait NotV1 extends Operator {
      def NotV1[
          @sp T <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Not", allInputs, map))
      }
   }

   // TODO:
   // New ops: Optional, OptionalGetElement, OptionalHasElement from opset 15
   // Map to Scala Option
   //
   trait OrV7 extends Operator {
      def OrV7[
          @sp T <: Boolean,
          @sp T1 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T1, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Or", allInputs, map))
      }
   }

   trait PReluV16 extends Operator {
      def PReluV16[
          @sp T <: BFloat16 | Float16 | Float | Double | UInt | ULong | Int | Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]],
          slope: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(X, slope)
         (callOp(name, "PRelu", allInputs, map))
      }
   }

   trait PadV13 extends Operator {
      def PadV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation,
          S2 <: Shape,
          Tt3 <: TensorTypeDenotation,
          AxesBefore <: Shape,
          AxesAfter <: Shape
      ](
          name: String,
          mode: String = "constant",
          data: Tensor[T, Tuple3[Tt, Td, S]],
          padsBefore: AxesBefore,
          padsAfter: AxesAfter, // Tensor[Long, Tuple3[Tt1,Td1,S1]], //`pads` should be a 1D tensor of shape [2 * input_rank].
          constant_value: Option[Tensor[T, Tuple3[Tt2, Td2, S2]]] = None
      )(using
          tt: ValueOf[Tt3],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[PaddedShape[S, AxesBefore, AxesAfter]]
      ): Tensor[T, Tuple3[Tt3, Td, PaddedShape[S, AxesBefore, AxesAfter]]] = {
         val map: Map[String, Any] = Map("mode" -> mode)
         val beforeArr             = padsBefore.toSeq.toArray
         val afterArr              = padsAfter.toSeq.toArray
         val padsArr               = (beforeArr ++ afterArr).map(_.toLong)
         val pads =
            Tensor(padsArr, padsArr.size.asInstanceOf[io.kjaer.compiletime.Dimension] #: SNil)

         val allInputs = Tuple3(data, pads, constant_value)
         (callOp(name, "Pad", allInputs, map))
      }
   }

   trait PowV15 extends Operator {
      def PowV15[
          @sp T <: Int | Long | Float16 | Float | Double: Numeric,
          @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String, X: Tensor[T, Tuple3[Tt, Td, S]], Y: Tensor[T1, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(X, Y)
         (callOp(name, "Pow", allInputs, map))
      }
   }

   trait ReciprocalV13 extends Operator {
      def ReciprocalV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Reciprocal", allInputs, map))
      }
   }

   // TODO P2: make axes param optional at the type level
   trait ReduceLogSumV13 extends Operator {
      def ReduceLogSumV13[
          @sp T <: UInt | ULong | Int | Long | BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axes: Option[(Axes)] = None,
          keepdims: KeepDims = true,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axes, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axes, KeepDims]],
          i: IndicesOf[Axes],
          k: ValueOf[KeepDims]
      ): Tensor[T, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axes, KeepDims],
        KeepOrReduceDims[S, Axes, KeepDims]
      ]] = {
         val map: Map[String, Any] = Map(
           "axes"     -> indicesOf[Axes].indices.toArray,
           "keepdims" -> (if valueOf[KeepDims] then 1 else 0)
         )
         val allInputs = Tuple1(data)
         (callOp(name, "ReduceLogSum", allInputs, map))
      }
   }

   trait ReduceMaxV13 extends Operator {
      def ReduceMaxV13[
          @sp T <: UInt | ULong | Int | Long | BFloat16 | Float16 | Float | Double | UByte |
             Byte: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axes: Option[(Axes)] = None,
          keepdims: KeepDims = true,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axes, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axes, KeepDims]],
          i: IndicesOf[Axes],
          k: ValueOf[KeepDims]
      ): Tensor[T, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axes, KeepDims],
        KeepOrReduceDims[S, Axes, KeepDims]
      ]] = {
         val map: Map[String, Any] = Map(
           "axes"     -> indicesOf[Axes].indices.toArray,
           "keepdims" -> (if valueOf[KeepDims] then 1 else 0)
         )
         val allInputs = Tuple1(data)
         (callOp(name, "ReduceMax", allInputs, map))
      }
   }

   trait ReduceMeanV13 extends Operator {
      def ReduceMeanV13[
          @sp T <: UInt | ULong | Int | Long | BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axes: Option[(Axes)] = None,
          keepdims: KeepDims = true,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axes, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axes, KeepDims]],
          i: IndicesOf[Axes],
          k: ValueOf[KeepDims]
      ): Tensor[T, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axes, KeepDims],
        KeepOrReduceDims[S, Axes, KeepDims]
      ]] = {
         val map: Map[String, Any] = Map(
           "axes"     -> indicesOf[Axes].indices.toArray,
           "keepdims" -> (if valueOf[KeepDims] then 1 else 0)
         )
         val allInputs = Tuple1(data)
         (callOp(name, "ReduceMean", allInputs, map))
      }
   }

   trait ReduceMinV13 extends Operator {
      def ReduceMinV13[
          @sp T <: UInt | ULong | Int | Long | BFloat16 | Float16 | Float | Double | UByte |
             Byte: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axes: Option[(Axes)] = None,
          keepdims: KeepDims = true,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axes, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axes, KeepDims]],
          i: IndicesOf[Axes],
          k: ValueOf[KeepDims]
      ): Tensor[T, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axes, KeepDims],
        KeepOrReduceDims[S, Axes, KeepDims]
      ]] = {
         val map: Map[String, Any] = Map(
           "axes"     -> indicesOf[Axes].indices.toArray,
           "keepdims" -> (if valueOf[KeepDims] then 1 else 0)
         )
         val allInputs = Tuple1(data)
         (callOp(name, "ReduceMin", allInputs, map))
      }
   }

   trait ReduceProdV13 extends Operator {
      def ReduceProdV13[
          @sp T <: UInt | ULong | Int | Long | BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axes: Option[(Axes)] = None,
          keepdims: KeepDims = true,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axes, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axes, KeepDims]],
          i: IndicesOf[Axes],
          k: ValueOf[KeepDims]
      ): Tensor[T, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axes, KeepDims],
        KeepOrReduceDims[S, Axes, KeepDims]
      ]] = {
         val map: Map[String, Any] = Map(
           "axes"     -> indicesOf[Axes].indices.toArray,
           "keepdims" -> (if valueOf[KeepDims] then 1 else 0)
         )
         val allInputs = Tuple1(data)
         (callOp(name, "ReduceProd", allInputs, map))
      }
   }

   trait ReduceSumSquareV13 extends Operator {
      def ReduceSumSquareV13[
          @sp T <: UInt | ULong | Int | Long | BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axes: Option[(Axes)] = None,
          keepdims: KeepDims = true,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axes, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axes, KeepDims]],
          i: IndicesOf[Axes],
          k: ValueOf[KeepDims]
      ): Tensor[T, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axes, KeepDims],
        KeepOrReduceDims[S, Axes, KeepDims]
      ]] = {
         val map: Map[String, Any] = Map(
           "axes"     -> indicesOf[Axes].indices.toArray,
           "keepdims" -> (if valueOf[KeepDims] then 1 else 0)
         )
         val allInputs = Tuple1(data)
         (callOp(name, "ReduceSumSquare", allInputs, map))
      }
   }

   // TODO: move "axes" from attributes to inputs on the rest of the Reduce ops,
   // as done below with ReduceSum (when updating to opset 18)
   //
   // TODO: new attr : noop_with_empty_axes
   trait ReduceSumV13 extends Operator {
      def ReduceSumV13[
          @sp T <: UInt | ULong | Int | Long | BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axes: Option[(Axes)] = None,
          keepdims: KeepDims = true,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axes, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axes, KeepDims]],
          i: IndicesOf[Axes],
          k: ValueOf[KeepDims]
      ): Tensor[T, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axes, KeepDims],
        KeepOrReduceDims[S, Axes, KeepDims]
      ]] = {
         val axes                  = indicesOf[Axes].indices.toArray
         val map: Map[String, Any] = Map("keepdims" -> (if valueOf[KeepDims] then 1 else 0))
         val allInputs = Tuple2(
           data,
           Tensor(axes.map(_.toLong), Shape.fromSeq(ArraySeq.unsafeWrapArray(Array(axes.size))))
         )
         (callOp(name, "ReduceSum", allInputs, map))
      }
   }

   trait ReluV14 extends Operator {
      def ReluV14[
          @sp T <: Int | Byte | Short | Long | BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Relu", allInputs, map))
      }
   }

   trait ReshapeV14 extends Operator {
      def ReshapeV14[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double],
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Shape,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation,
          S2 <: Shape
      ](
          name: String,
          allowzero: Option[Int] = None,
          data: Tensor[T, Tuple3[Tt, Td, S]],
          shapeInput: Tensor[Long, Tuple3[Tt1, Td1, S1]]
      )(using
          tt: ValueOf[Tt2],
          td: TensorShapeDenotationOf[Td2],
          s: ShapeOf[S2],
          sizeSeq: NumElements[S] =:= NumElements[S2]
      ): Tensor[T, Tuple3[Tt2, Td2, S2]] = {
         val map: Map[String, Any] = allowzero match {
            case None    => Map()
            case Some(x) => Map("allowzero" -> x)
         }
         val allInputs = Tuple2(data, shapeInput)
         (callOp(name, "Reshape", allInputs, map))
      }
   }

   // TODO P2: Add, was added to ONNXJS recently, WebGl only
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T1,_] = {
      val map: Map[String, Any] = Map(
        "coordinate_transformation_mode" -> coordinate_transformation_mode,
        "cubic_coeff_a"                  -> cubic_coeff_a,
        "exclude_outside"                -> exclude_outside,
        "extrapolation_value"            -> extrapolation_value,
        "mode"                           -> mode,
        "nearest_mode"                   -> nearest_mode
      )
      val allInputs = Tuple4(X, roi, scales, sizes)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs             = Tuple2(X, scales)
      (callOp(name, "Resize", allInputs, map))
    }
  }
    */

   trait RoundV11 extends Operator {
      def RoundV11[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Round", allInputs, map))
      }
   }

   // TODO:
   // New Op: STFT - since opset 17
   trait SeluV6 extends Operator {
      def SeluV6[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          alpha: Float = 1.67326,
          gamma: Float = 1.0507,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map("alpha" -> alpha, "gamma" -> gamma)
         val allInputs             = Tuple1(X)
         (callOp(name, "Selu", allInputs, map))
      }
   }

   trait ShapeV15 extends Operator {
      def ShapeV15[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | Boolean | String | Complex[Float] | Complex[Double],
          @sp T1 <: Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation
      ](
          name: String,
          end: Option[Int] = None,
          start: Option[Int] = None,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[io.kjaer.compiletime.Shape.Rank[S] & (Dimension #: SNil)]
      ): Tensor[T1, Tuple3[Tt1, Td1, io.kjaer.compiletime.Shape.Rank[S] & (Dimension #: SNil)]] = {
         val map: Map[String, Any] = Map("end" -> end, "start" -> start)
         val allInputs             = Tuple1(data)
         (callOp(name, "Shape", allInputs, map))
      }
   }

   trait SigmoidV13 extends Operator {
      def SigmoidV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Sigmoid", allInputs, map))
      }
   }

   trait SignV13 extends Operator {
      def SignV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String, input: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Sign", allInputs, map))
      }
   }

   trait SinV7 extends Operator {
      def SinV7[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Sin", allInputs, map))
      }
   }

   trait SinhV9 extends Operator {
      def SinhV9[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Sinh", allInputs, map))
      }
   }

   // TODO P1: All 4 params must be 1D vectors of same size - to enforce
   // TODO P2: Constraints on axes / steps params
   trait SliceV13 extends Operator {
      def SliceV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double],
//        @sp Tind <: Int | Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          AxesStart <: Indices,
          AxesEnd <: Indices,
          AxisIndices <: None.type | Indices,
          StepIndices <: None.type | Indices
      ](
          name: String,
          data: Tensor[T, Tuple3[Tt, Td, S]],
          starts: AxesStart,
          ends: AxesEnd,
          axes: AxisIndices = None,
          steps: StepIndices = None
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s5: ShapeOf[SlicedShape[AxesStart, AxesEnd]]
      ): Tensor[T, Tuple3[Tt1, Td, SlicedShape[AxesStart, AxesEnd]]] = {
         val map: Map[String, Any] = Map()
         val startsArr             = starts.indices.toArray
         val newStarts =
            Tensor(startsArr, startsArr.size.asInstanceOf[io.kjaer.compiletime.Dimension] #: SNil)
         val endsArr = ends.indices.toArray
         val newEnds =
            Tensor(endsArr, endsArr.size.asInstanceOf[io.kjaer.compiletime.Dimension] #: SNil)

         val newAxes = axes match {
            case None => None
            case x: Indices => {
               val axesArr = x.indices.toArray
               Tensor(axesArr, axesArr.size.asInstanceOf[io.kjaer.compiletime.Dimension] #: SNil)
            }
         }

         val newSteps = steps match {
            case None => None
            case x: Indices => {
               val stepsArr = x.indices.toArray
               Tensor(stepsArr, stepsArr.size.asInstanceOf[io.kjaer.compiletime.Dimension] #: SNil)
            }
         }

         val allInputs = Tuple5(data, newStarts, newEnds, newAxes, newSteps)
         (callOp(name, "Slice", allInputs, map))
      }
   }

   // From the spec : "The input does not need to explicitly be a 2D vector; rather, it will be coerced into one"
   // Here we require that it is 2D
   trait SoftmaxV13 extends Operator {
      def SoftmaxV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation
      ](
          name: String,
          axis: Int = 1,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt1, Td, S]] = {
         val map: Map[String, Any] = Map("axis" -> axis)
         val allInputs             = Tuple1(input)
         (callOp(name, "Softmax", allInputs, map))
      }
   }

   trait SqrtV13 extends Operator {
      def SqrtV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(X)
         (callOp(name, "Sqrt", allInputs, map))
      }
   }

   // "If axes is not provided, all the single dimensions will be removed from the shape"
   trait SqueezeV13 extends Operator {
      def SqueezeV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double],
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices
      ](name: String, axes: Option[(Axes)] = None, data: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axes, false]],
          s: ShapeOf[KeepOrReduceDims[S, Axes, false]],
          i: IndicesOf[Axes]
      ): Tensor[T, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axes, false],
        KeepOrReduceDims[S, Axes, false]
      ]] = {
         val axes                  = indicesOf[Axes].indices.toArray
         val map: Map[String, Any] = Map()
         val allInputs = Tuple2(
           data,
           Tensor(axes.map(_.toLong), Shape.fromSeq(ArraySeq.unsafeWrapArray(Array(axes.size))))
         )
         (callOp(name, "Squeeze", allInputs, map))
      }
   }

   trait SubV14 extends Operator {
      def SubV14[
          @sp T <: UByte | Byte | UShort | Short | UInt | ULong | Int | Long | BFloat16 | Float16 |
             Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Sub", allInputs, map))
      }
   }

   trait SumV13 extends Operator {
      def SumV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          data_0: Seq[Tensor[T, Tuple3[Tt, Td, S]]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple.fromArray(data_0.toArray)
         (callOp(name, "Sum", allInputs, map))
      }
   }

   trait TanV7 extends Operator {
      def TanV7[
          @sp T <: Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Tan", allInputs, map))
      }
   }

   trait TanhV13 extends Operator {
      def TanhV13[
          @sp T <: BFloat16 | Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Tanh", allInputs, map))
      }
   }

   trait TileV13 extends Operator {
      def TileV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean | Complex[Float] | Complex[Double],
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt2 <: TensorTypeDenotation,
          AxisRepeats <: Indices
      ](name: String, input: Tensor[T, Tuple3[Tt, Td, S]], repeats: AxisRepeats)(using
          tt: ValueOf[Tt2],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[TiledShape[S, AxisRepeats]]
      ): Tensor[T, Tuple3[Tt2, Td, TiledShape[S, AxisRepeats]]] = {
         val map: Map[String, Any] = Map()
         val repeatsArr            = repeats.indices.toArray.map(_.toLong)
         val repeatsTens =
            Tensor(repeatsArr, repeatsArr.size.asInstanceOf[io.kjaer.compiletime.Dimension] #: SNil)

         val allInputs = Tuple2(input, repeatsTens)
         (callOp(name, "Tile", allInputs, map))
      }
   }

   trait TransposeV13 extends Operator {
      def TransposeV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double],
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String, perm: Option[(Array[Int])] = None, data: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Reverse[Td]],
          s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]
      ): Tensor[T, Tuple3[Tt, Reverse[Td], io.kjaer.compiletime.Shape.Reverse[S]]] = {
         val map: Map[String, Any] = Map("perm" -> perm)
         val allInputs             = Tuple1(data)
         (callOp(name, "Transpose", allInputs, map))
      }
   }

   trait UnsqueezeV13 extends Operator {
      def UnsqueezeV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | BFloat16 | Float16 |
             Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double],
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axes <: Indices
      ](name: String, axes: Option[Axes] = None, data: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[UnsqueezeShape[S, Axes]],
          i: IndicesOf[Axes]
      ): Tensor[T, Tuple3[Tt1, Td, UnsqueezeShape[S, Axes]]] = {
         val axes                  = indicesOf[Axes].indices.toArray
         val map: Map[String, Any] = Map()
         val allInputs = Tuple2(
           data,
           Tensor(axes.map(_.toLong), Shape.fromSeq(ArraySeq.unsafeWrapArray(Array(axes.size))))
         )
         (callOp(name, "Unsqueeze", allInputs, map))
      }
   }

   trait XorV7 extends Operator {
      def XorV7[
          @sp T <: Boolean,
          @sp T1 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T1, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Xor", allInputs, map))
      }
   }

}

//Ops from the default ai.onnx domain which are only supported in ORT
package object onnxruntime {

   trait ArgMaxV13 extends onnx.Operator {
      def ArgMaxV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | onnx.BFloat16 |
             onnx.Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axis <: Index ::: INil,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axis: Axis = 0 ::: INil,
          keepdims: KeepDims = true,
          selectLastIndex: Int = 0,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axis, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axis, KeepDims]],
          i: IndicesOf[Axis],
          k: ValueOf[KeepDims]
      ): Tensor[Long, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axis, KeepDims],
        KeepOrReduceDims[S, Axis, KeepDims]
      ]] = {
         val map: Map[String, Any] = Map(
           "axis"              -> indicesOf[Axis].indices.toArray.head,
           "select_last_index" -> selectLastIndex,
           "keepdims"          -> (if valueOf[KeepDims] then 1 else 0)
         )
         val allInputs = Tuple1(data)
         (callOp(name, "ArgMax", allInputs, map))
      }
   }

   trait ArgMinV13 extends onnx.Operator {
      def ArgMinV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | onnx.BFloat16 |
             onnx.Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Axis <: Index ::: INil,
          KeepDims <: (Boolean & Singleton)
      ](
          name: String,
          axis: Axis = 0 ::: INil,
          keepdims: KeepDims = true,
          selectLastIndex: Int = 0,
          data: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td, Axis, KeepDims]],
          s: ShapeOf[KeepOrReduceDims[S, Axis, KeepDims]],
          i: IndicesOf[Axis],
          k: ValueOf[KeepDims]
      ): Tensor[Long, Tuple3[
        Tt1,
        KeepOrReduceDimDenotations[Td, Axis, KeepDims],
        KeepOrReduceDims[S, Axis, KeepDims]
      ]] = {
         val map: Map[String, Any] = Map(
           "axis"              -> indicesOf[Axis].indices.toArray.head,
           "select_last_index" -> selectLastIndex,
           "keepdims"          -> (if valueOf[KeepDims] then 1 else 0)
         )
         val allInputs = Tuple1(data)
         (callOp(name, "ArgMin", allInputs, map))
      }
   }

   trait BitShiftV11 extends onnx.Operator {
      def BitShiftV11[
          @sp T <: UByte | UShort | UInt | ULong: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          direction: (String),
          X: Tensor[T, Tuple3[Tt, Td, S]],
          Y: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map("direction" -> direction)
         val allInputs             = Tuple2(X, Y)
         (callOp(name, "BitShift", allInputs, map))
      }
   }

   // Diverging from the spec, min and max are optional there, but otherwise it's a no-op
   trait ClipV13 extends onnx.Operator {
      def ClipV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | onnx.BFloat16 |
             onnx.Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]],
          min: Tensor[T, Tuple3[Tt, Td, SNil]],
          max: Tensor[T, Tuple3[Tt, Td, SNil]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple3(input, min, max)
         (callOp(name, "Clip", allInputs, map))
      }
   }
   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, Cx] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Tuple2(input, condition)
      (callOp(name, "Compress", allInputs, map))
    }
  }
    */

   // Not supported, sequence op
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "new_axis" -> new_axis)
      val allInputs             = Tuple1(input_sequence)
      (callOp(name, "ConcatFromSequence", allInputs, map))
    }
  }
    */
   // TODO
   /*
   trait ConstantOfShapeV9 extends Operator {
    def ConstantOfShapeV9[
        @sp T1 <: Long: Numeric,
        @sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean
    ,Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](name: String,
             value: Option[(Tensor[T2, Tuple3[Tt,Td,S]])] = None,
             input: Tensor[T1, Bx])
    (using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, Tuple3[Tt,Td,S]] = {
      val map: Map[String, Any] = Map("value" -> value)
      val allInputs             = Tuple1(input)
      (callOp(name, "ConstantOfShape", allInputs, map))
    }
  }
    */
   // Bug in ORT here, it forces us to set shape as an input even though in the spec there are 0 inputs, it uses ConstantOfShape op instead
   trait ConstantV13 extends onnx.Operator {
      def ConstantV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | onnx.BFloat16 |
             onnx.Float16 | Float | Double | String | Boolean |
             Complex[
               Float
             ] | Complex[Double]: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          sparse_value: Option[(SparseTensor[T, Tuple3[Tt, Td, S]])] = None,
          value: Option[(Tensor[T, Tuple3[Tt, Td, S]])] = None,
          value_float: Option[(Float)] = None,
          value_floats: Option[(Array[Float])] = None,
          value_int: Option[(Int)] = None,
          value_ints: Option[(Array[Int])] = None,
          value_string: Option[(String)] = None,
          value_strings: Option[(Array[String])] = None
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
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
         val shapeArr             = s.value.toSeq.toArray
         val shapeSize: Dimension = shapeArr.size.asInstanceOf[Dimension]
         val allInputs            = Tuple(Tensor(shapeArr.map(_.toLong), shapeSize #: SNil))
         (callOp(name, "Constant", allInputs, map))
      }
   }

   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T3, Ex] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Tuple4(x, w, x_zero_point, w_zero_point)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, Dx] = {
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
      val allInputs = Tuple3(X, W, B)
      (callOp(name, "ConvTranspose", allInputs, map))
    }
  }
    */
   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("exclusive" -> exclusive, "reverse" -> reverse)
      val allInputs             = Tuple2(x, axis)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("blocksize" -> blocksize, "mode" -> mode)
      val allInputs             = Tuple1(input)
      (callOp(name, "DepthToSpace", allInputs, map))
    }
  }

  //Missing V13
  trait DequantizeLinearV10 extends Operator {
    def DequantizeLinearV10[@sp T <: Byte | UByte | Int: Numeric, Ax <: Axes](
        name: String,
        x: Tensor[T, _],
        x_scale: Tensor[Float,_],
        x_zero_point: Option[Tensor[T, _]] = None
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Float,_] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple3(x, x_scale, x_zero_point)
      (callOp(name, "DequantizeLinear", allInputs, map))
    }
  }
  trait DetV11 extends Operator {
    def DetV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(X)
      (callOp(name, "Det", allInputs, map))
    }
  }
    */
   // TODO
   /*
  trait DynamicQuantizeLinearV11 extends Operator {
    def DynamicQuantizeLinearV11[
        @sp T1 <: Float: Numeric,
        @sp T2 <: UByte: Numeric
    , Ax <: Axes](name: String, x: Tensor[T1,_])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(x)
      (callOp(name, "DynamicQuantizeLinear", allInputs, map))
    }
  }

  trait EinsumV12 extends Operator {
    def EinsumV12[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, equation: (String), Inputs: Seq[Tensor[T, _]])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("equation" -> equation)
      val allInputs             = Tuple.fromArray(Inputs.toArray).asInstanceOf[Tuple]
      (callOp(name, "Einsum", allInputs, map))
    }
  }
    */

   // TODO
   /*
  trait ErfV9 extends Operator {
    def ErfV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](name: String, input: Tensor[T, _])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(input)
      (callOp(name, "Erf", allInputs, map))
    }
  }
    */

   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, _] = {
      val map: Map[String, Any] = Map("dtype" -> dtype, "k" -> k)
      val allInputs             = Tuple1(input)
      (callOp(name, "EyeLike", allInputs, map))
    }
  }
    */
   // Not supported, ORT fails in backend scoreboard
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha"    -> activation_alpha,
        "activation_beta"     -> activation_beta,
        "activations"         -> activations,
        "clip"                -> clip,
        "direction"           -> direction,
        "hidden_size"         -> hidden_size,
        "linear_before_reset" -> linear_before_reset
      )
      val allInputs = Tuple6(X, W, R, B, sequence_lens, initial_h)
      (callOp(name, "GRU", allInputs, map))
    }
  }
    */

   // TODO
   /*
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Tuple2(data, indices)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("batch_dims" -> batch_dims)
      val allInputs             = Tuple2(data, indices)
      (callOp(name, "GatherND", allInputs, map))
    }
  }

  trait GatherNDV11 extends Operator {
    def GatherNDV11[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, data: Tensor[T, _], indices: Tensor[Long, _])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple2(data, indices)
      (callOp(name, "GatherND", allInputs, map))
    }
  }
    */
   // Not supported, ORT fails in backend scoreboard
   /*
  trait GlobalLpPoolV2 extends Operator {
    def GlobalLpPoolV2[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        p: Option[(Int)] = None,
        X: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("p" -> p)
      val allInputs             = Tuple1(X)
      (callOp(name, "GlobalLpPool", allInputs, map))
    }
  }

    */

   trait GreaterOrEqualV16 extends onnx.Operator {
      def GreaterOrEqualV16[
          @sp T <: onnx.BFloat16 | UByte | UShort | UInt | ULong | Byte | Short | Int | Long |
             onnx.Float16 | Float | Double: Numeric,
          @sp T1 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation
      ](name: String, A: Tensor[T, Tuple3[Tt, Td, S]], B: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S]
      ): Tensor[T1, Tuple3[Tt1, Td1, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "GreaterOrEqual", allInputs, map))
      }
   }

   trait GreaterV13 extends onnx.Operator {
      def GreaterV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | onnx.BFloat16 |
             onnx.Float16 | Float | Double: Numeric,
          @sp T1 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation
      ](name: String, A: Tensor[T, Tuple3[Tt, Td, S]], B: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S]
      ): Tensor[T1, Tuple3[Tt1, Td1, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Greater", allInputs, map))
      }
   }

   // TODO:
   // New op: GridSample - since opset 16
   // New op: GroupNormalization - since opset 18
   // New op: HammingWindow - since opset 17
   // New op: HannWindow - since opset 17
   // TODO
   /*
  trait HardSigmoidV6 extends Operator {
    def HardSigmoidV6[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        X: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "beta" -> beta)
      val allInputs             = Tuple1(X)
      (callOp(name, "HardSigmoid", allInputs, map))
    }
  }

  //Missing V13
  trait HardmaxV11 extends Operator {
    def HardmaxV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Tuple1(input)
      (callOp(name, "Hardmax", allInputs, map))
    }
  }
  //TODO
  //New op: HardSwish - since opset 14

  //Doesn't make sense in this context, this makes a copy, which in this case means scala -> backend -> scala round trip, essentially no-op
  //And it's eliminated in graph optimization anyway
  trait IdentityV1 extends Operator {
    def IdentityV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, input: Tensor[T, _])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(input)
      (callOp(name, "Identity", allInputs, map))
    }
  }

  //Missing V13
  //
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[V, _] = {
      val map: Map[String, Any] = Map("else_branch" -> else_branch, "then_branch" -> then_branch)
      val allInputs             = Tuple1(cond)
      (callOp(name, "If", allInputs, map))
    }
  }
    */
   // TODO
   /*
  trait IsInfV10 extends Operator {
    def IsInfV10[@sp T1 <: Float | Double: Numeric, @sp T2 <: Boolean, Ax <: Axes](
        name: String,
        detect_negative: Option[(Int)] = None,
        detect_positive: Option[(Int)] = None,
        X: Tensor[T1,_]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("detect_negative" -> detect_negative, "detect_positive" -> detect_positive)
      val allInputs = Tuple1(X)
      (callOp(name, "IsInf", allInputs, map))
    }
  }
    */
   // Not supported, ORT fails in backend scoreboard
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size,
        "input_forget"     -> input_forget
      )
      val allInputs = Tuple8(X, W, R, B, sequence_lens, initial_h, initial_c, P)
      (callOp(name, "LSTM", allInputs, map))
    }
  }
    */

   trait LessOrEqualV16 extends onnx.Operator {
      def LessOrEqualV16[
          @sp T <: onnx.BFloat16 | UByte | UShort | UInt | ULong | Byte | Short | Int | Long |
             onnx.Float16 | Float | Double: Numeric,
          @sp T1 <: Boolean,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation
      ](name: String, A: Tensor[T, Tuple3[Tt, Td, S]], B: Tensor[T, Tuple3[Tt, Td, S]])(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S]
      ): Tensor[T1, Tuple3[Tt1, Td1, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(A, B)
         (callOp(name, "LessOrEqual", allInputs, map))
      }
   }
   // TODO, missing V13
   /*
  trait LogSoftmaxV11 extends Operator {
    def LogSoftmaxV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        input: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Tuple1(input)
      (callOp(name, "LogSoftmax", allInputs, map))
    }
  }
    */
//TODO, Missing V13
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[V, _] = {
      val map: Map[String, Any] = Map("body" -> body)
      val allInputs =
        Tuple2(M, cond) ++ (Tuple.fromArray(v_initial.toArray).asInstanceOf[Tuple])

      (callOp(name, "Loop", allInputs, map))
    }
  }

  //Not supported, ORT fails in backend scoreboard
  trait LpNormalizationV1 extends Operator {
    def LpNormalizationV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axis: Option[(Int)] = None,
        p: Option[(Int)] = None,
        input: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "p" -> p)
      val allInputs             = Tuple1(input)
      (callOp(name, "LpNormalization", allInputs, map))
    }
  }

  //Not supported, ORT fails in backend scoreboard
  trait LpPoolV11 extends Operator {
    def LpPoolV11[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: (Array[Int]),
        p: Option[(Int)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "p"            -> p,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = Tuple1(X)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T3, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple4(A, B, a_zero_point, b_zero_point)
      (callOp(name, "MatMulInteger", allInputs, map))
    }
  }
    */
   // Not supported, ORT fails in backend scoreboard
   /*
  trait MaxRoiPoolV1 extends Operator {
    def MaxRoiPoolV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        pooled_shape: (Array[Int]),
        spatial_scaleAttr: Option[(Float)] = None,
        X: Tensor[T, _],
        rois: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] =
        Map("pooled_shape" -> pooled_shape, "spatial_scaleAttr" -> spatial_scaleAttr)
      val allInputs = Tuple2(X, rois)
      (callOp(name, "MaxRoiPool", allInputs, map))
    }
  }
    */

   // TODO
   /*
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T1,_] = {
      val map: Map[String, Any] =
        Map("kernel_shape" -> kernel_shape, "pads" -> pads, "strides" -> strides)
      val allInputs = Tuple3(X, I, output_shapeInput)
      (callOp(name, "MaxUnpool", allInputs, map))
    }
  }

    */

   trait MaxV13 extends onnx.Operator {
      def MaxV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | onnx.BFloat16 |
             onnx.Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String, data_0: Seq[Tensor[T, Tuple3[Tt, Td, S]]])(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple.fromArray(data_0.toArray)
         (callOp(name, "Max", allInputs, map))
      }
   }

   trait MeanV13 extends onnx.Operator {
      def MeanV13[
          @sp T <: onnx.BFloat16 | onnx.Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          data_0: Seq[Tensor[T, (Tt, Td, S)]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, (Tt, Td, S)] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple.fromArray(data_0.toArray)
         (callOp(name, "Mean", allInputs, map))
      }
   }

   // TODO:
   // New op: MelWeightMatrix - since opset 17

   // TODO
   /*

  trait MeanVarianceNormalizationV9 extends Operator {
    def MeanVarianceNormalizationV9[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        X: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs             = Tuple1(X)
      (callOp(name, "MeanVarianceNormalization", allInputs, map))
    }
  }
    */
   trait MinV13 extends onnx.Operator {
      def MinV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | onnx.BFloat16 |
             onnx.Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](name: String, data_0: Seq[Tensor[T, Tuple3[Tt, Td, S]]])(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple.fromArray(data_0.toArray)
         (callOp(name, "Min", allInputs, map))
      }
   }

   // TODO:
   // New Op: Mish - since opset 18
   trait ModV13 extends onnx.Operator {
      def ModV13[
          @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | onnx.BFloat16 |
             onnx.Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          fmod: Int = 0,
          A: Tensor[T, Tuple3[Tt, Td, S]],
          B: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map("fmod" -> fmod)
         val allInputs             = Tuple2(A, B)
         (callOp(name, "Mod", allInputs, map))
      }
   }
   // Not supported, ORT fails in backend scoreboard
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "sample_size" -> sample_size, "seed" -> seed)
      val allInputs = Tuple1(input)
      (callOp(name, "Multinomial", allInputs, map))
    }
  }
    */

   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("reduction" -> reduction)
      val allInputs             = Tuple3(input, target, weight)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Long, _] = {
      val map: Map[String, Any] = Map("center_point_box" -> center_point_box)
      val allInputs =
        Tuple5(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
      (callOp(name, "NonMaxSuppression", allInputs, map))
    }
  }

  trait NonZeroV9 extends Operator {
    def NonZeroV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, X: Tensor[T, _])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Long, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(X)
      (callOp(name, "NonZero", allInputs, map))
    }
  }
    */
   // TODO
   /*
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T3, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Tuple3(indices, depth, values)
      (callOp(name, "OneHot", allInputs, map))
    }
  }
    */

   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T3, _] = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs =
        Tuple9(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T3, _] = {
      val map: Map[String, Any] = Map()
      val allInputs =
        Tuple8(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point)
      (callOp(name, "QLinearMatMul", allInputs, map))
    }
  }

  //Missing V13
  trait QuantizeLinearV10 extends Operator {
    def QuantizeLinearV10[
        @sp T1 <: Float | Int: Numeric,
        @sp T2 <: Byte | UByte: Numeric
    , Ax <: Axes](
        name: String,
        x: Tensor[T1,_],
        y_scale: Tensor[Float,_],
        y_zero_point: Option[Tensor[T2, _]] = None
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple3(x, y_scale, y_zero_point)
      (callOp(name, "QuantizeLinear", allInputs, map))
    }
  }
    */

//Not supported, ORT fails in backend scoreboard
   /*
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size
      )
      val allInputs = Tuple6(X, W, R, B, sequence_lens, initial_h)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "mean" -> mean, "scaleAttr" -> scaleAttr, "seed" -> seed)
      val allInputs = Tuple1(input)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map(
        "dtype"     -> dtype,
        "mean"      -> mean,
        "scaleAttr" -> scaleAttr,
        "seed"      -> seed,
        "shape"     -> shape
      )
      val allInputs = EmptyTuple
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "high" -> high, "low" -> low, "seed" -> seed)
      val allInputs = Tuple1(input)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "high" -> high, "low" -> low, "seed" -> seed, "shape" -> shape)
      val allInputs = EmptyTuple
      (callOp(name, "RandomUniform", allInputs, map))
    }
  }
    */

   trait RangeV11 extends onnx.Operator {
      def RangeV11[
          @sp T <: Float | Double | Short | Int | Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Shape,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation,
          S2 <: Shape,
          Tt3 <: TensorTypeDenotation,
          Td3 <: TensorShapeDenotation,
          S3 <: Shape
      ](
          name: String,
          start: Tensor[T, Tuple3[Tt, Td, S]],
          limit: Tensor[T, Tuple3[Tt1, Td1, S1]],
          delta: Tensor[T, Tuple3[Tt2, Td2, S2]]
      )(using
          tt: ValueOf[Tt3],
          td: TensorShapeDenotationOf[Td3],
          s: ShapeOf[S3]
      ): Tensor[T, Tuple3[Tt3, Td3, S3]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple3(start, limit, delta)
         (callOp(name, "Range", allInputs, map))
      }
   }
   // TODO
   /*
  trait ReduceL1V11 extends Operator {
    def ReduceL1V11[
        @sp T <: UInt | ULong | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Tuple1(data)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Tuple1(data)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs             = Tuple1(data)
      (callOp(name, "ReduceLogSumExp", allInputs, map))
    }
  }

    */
   // Not supported, sequence op
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("batch_axis" -> batch_axis, "time_axis" -> time_axis)
      val allInputs             = Tuple2(input, sequence_lens)
      (callOp(name, "ReverseSequence", allInputs, map))
    }
  }
    */
   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T1,_] = {
      val map: Map[String, Any] = Map(
        "mode"              -> mode,
        "output_height"     -> output_height,
        "output_width"      -> output_width,
        "sampling_ratio"    -> sampling_ratio,
        "spatial_scaleAttr" -> spatial_scaleAttr
      )
      val allInputs = Tuple3(X, rois, batch_indices)
      (callOp(name, "RoiAlign", allInputs, map))
    }
  }
    */
   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[V, _] = {
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs             = Tuple3(data, indices, updates)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple3(data, indices, updates)
      (callOp(name, "ScatterND", allInputs, map))
    }
  }
    */
   // Not supported, sequence op
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple2(input_sequence, position)
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
    , Ax <: Axes](name: String, inputs: Seq[Tensor[T, _]])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): S = {
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): S = {
      val map: Map[String, Any] = Map("dtype" -> dtype)
      val allInputs             = EmptyTuple
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): S = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple2(input_sequence, position)
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): S = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple3(input_sequence, tensor, position)
      (callOp(name, "SequenceInsert", allInputs, map))
    }
  }

  //TODO:
  //New Op: SequenceMap - since opset 17
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[I, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(input_sequence)
      (callOp(name, "SequenceLength", allInputs, map))
    }
  }
    */
   // TODO
   /*
  trait ShrinkV9 extends Operator {
    def ShrinkV9[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double: Numeric
    , Ax <: Axes](
        name: String,
        bias: Option[(Float)] = None,
        lambd: Option[(Float)] = None,
        input: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("bias" -> bias, "lambd" -> lambd)
      val allInputs             = Tuple1(input)
      (callOp(name, "Shrink", allInputs, map))
    }
  }
    */
   // TODO
   /*
  trait SizeV1 extends Operator {
    def SizeV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T1 <: Long: Numeric
    , Ax <: Axes](name: String, data: Tensor[T, _])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T1,_] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(data)
      (callOp(name, "Size", allInputs, map))
    }
  }
    */
   // TODO, missing V13
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("reduction" -> reduction)
      val allInputs             = Tuple3(scores, labels, weights)
      (callOp(name, "SoftmaxCrossEntropyLoss", allInputs, map))
    }
  }
    */
//TODO
   /*
  trait SoftplusV1 extends Operator {
    def SoftplusV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        X: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(X)
      (callOp(name, "Softplus", allInputs, map))
    }
  }

  trait SoftsignV1 extends Operator {
    def SoftsignV1[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        input: Tensor[T, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(input)
      (callOp(name, "Softsign", allInputs, map))
    }
  }
  //Not supported, ORT fails in backend scoreboard
  trait SpaceToDepthV1 extends Operator {
    def SpaceToDepthV1[
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, blocksize: (Int), input: Tensor[T, _])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("blocksize" -> blocksize)
      val allInputs             = Tuple1(input)
      (callOp(name, "SpaceToDepth", allInputs, map))
    }
  }

  //Not supported, sequence op
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): S = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs             = Tuple2(input, split)
      (callOp(name, "SplitToSequence", allInputs, map))
    }
  }


  //Missing V13
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "splitAttr" -> splitAttr)
      val allInputs             = Tuple1(input)
      (callOp(name, "Split", allInputs, map))
    }
  }

    */
   /*
  trait StringNormalizerV10 extends Operator {
    def StringNormalizerV10(
        name: String,
        case_change_action: Option[(String)] = None,
        is_case_sensitive: Option[(Int)] = None,
        locale: Option[(String)] = None,
        stopwords: Option[(Array[String])] = None,
        X: Tensor[String, _]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[String, _] = {
      val map: Map[String, Any] = Map(
        "case_change_action" -> case_change_action,
        "is_case_sensitive"  -> is_case_sensitive,
        "locale"             -> locale,
        "stopwords"          -> stopwords
      )
      val allInputs = Tuple1(X)
      (callOp(name, "StringNormalizer", allInputs, map))
    }
  }
    */
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T1,_] = {
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
      val allInputs = Tuple1(X)
      (callOp(name, "TfIdfVectorizer", allInputs, map))
    }
  }
    */
   // TODO
   /*
  trait ThresholdedReluV10 extends Operator {
    def ThresholdedReluV10[@sp T <: Float16 | Float | Double: Numeric, Ax <: Axes](
        name: String,
        alpha: Option[(Float)] = None,
        X: Tensor[T, Ax]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, Ax] = {
      val map: Map[String, Any] = Map("alpha" -> alpha)
      val allInputs             = Tuple1(X)
      (callOp(name, "ThresholdedRelu", allInputs, map))
    }
  }
    */
   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "largest" -> largest, "sorted" -> sorted)
      val allInputs             = Tuple2(X, K)
      (callOp(name, "TopK", allInputs, map))
    }
  }

    */

   // TODO
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map("axis" -> axis, "sorted" -> sorted)
      val allInputs             = Tuple1(X)
      (callOp(name, "Unique", allInputs, map))
    }
  }
    */

   // TODO
   /*
  trait WhereV9 extends Operator {
    def WhereV9[
        @sp B <: Boolean,
        @sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric
    , Ax <: Axes](name: String, condition: Tensor[B, _], X: Tensor[T, _], Y: Tensor[T, _])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T, _] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple3(condition, X, Y)
      (callOp(name, "Where", allInputs, map))
    }
  }
    */

}
//ORT contrib ops
//ONNX domain: com.microsoft
//See: https://github.com/microsoft/onnxruntime/blob/v1.7.1/docs/ContribOperators.md
package object onnxruntimecontrib {
   trait InverseV1 extends onnx.Operator {
      def InverseV1[
          @sp T <: onnx.Float16 | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](
          name: String,
          input: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple1(input)
         (callOp(name, "Inverse", allInputs, map))
      }
   }
}

//ONNX domain: ai.onnx.ml
//See: https://github.com/onnx/onnx/blob/v1.8.1/docs/Operators-ml.md
package object onnxml {
   // TODO: P3 shape constraints
   trait ArrayFeatureExtractorV1 extends onnx.Operator {
      def ArrayFeatureExtractorV1[
          @sp T <: Float | Double | Long | Int | String: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Shape,
          Tt2 <: TensorTypeDenotation,
          Td2 <: TensorShapeDenotation,
          S2 <: Shape
      ](
          name: String,
          X: Tensor[T, Tuple3[Tt, Td, S]],
          Y: Tensor[Long, Tuple3[Tt1, Td1, S1]]
      )(using
          tt: ValueOf[Tt2],
          td: TensorShapeDenotationOf[Td2],
          s: ShapeOf[S2]
      ): Tensor[T, Tuple3[Tt2, Td2, S2]] = {
         val map: Map[String, Any] = Map()
         val allInputs             = Tuple2(X, Y)
         (callOp(name, "ArrayFeatureExtractor", allInputs, map))
      }
   }

   trait BinarizerV1 extends onnx.Operator {
      def BinarizerV1[
          @sp T <: Float | Double | Long | Int: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation
      ](
          name: String,
          threshold: Float = 0.0f,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T, Tuple3[Tt1, Td, S]] = {
         val map: Map[String, Any] = Map("threshold" -> threshold)
         val allInputs             = Tuple1(X)
         (callOp(name, "Binarizer", allInputs, map))
      }
   }

   // Not supported, ONNX ML - using ONNX Map
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, Ax] = {
      val map: Map[String, Any] =
        Map("cast_to" -> cast_to, "map_form" -> map_form, "max_map" -> max_map)
      val allInputs = Tuple1(X)
      (callOp(name, "CastMap", allInputs, map))
    }
  }
    */

   // TODO: P3 constraints -
   // split out to Int -> String and String -> Int
   trait CategoryMapperV1 extends onnx.Operator {
      def CategoryMapperV1[
          @sp T1 <: String | Long: Numeric,
          @sp T2 <: String | Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation
      ](
          name: String,
          cats_int64s: Array[Int],
          cats_strings: Array[String],
          default_int64: Int = -1,
          default_string: String = "_Unused",
          X: Tensor[T1, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T2, Tuple3[Tt1, Td, S]] = {
         val map: Map[String, Any] = Map(
           "cats_int64s"    -> cats_int64s,
           "cats_strings"   -> cats_strings,
           "default_int64"  -> default_int64,
           "default_string" -> default_string
         )
         val allInputs = Tuple1(X)
         (callOp(name, "CategoryMapper", allInputs, map))
      }
   }

//Not supported - ONNX ML - uses ONNX Map
   /*
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T2, _] = {
      val map: Map[String, Any] =
        Map("int64_vocabulary" -> int64_vocabulary, "string_vocabulary" -> string_vocabulary)
      val allInputs = Tuple1(X)
      (callOp(name, "DictVectorizer", allInputs, map))
    }
  }
    */

   // All input shapes are 2-D and are concatenated along the second dimension
   // TODO: P3 output shape constraint - match type summing over second dim, similar to concat op
   trait FeatureVectorizerV1 extends onnx.Operator {
      def FeatureVectorizerV1[
          @sp T1 <: Int | Long | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: Dimension #: SNil
      ](
          name: String,
          inputdimensions: Option[(Array[Int])] = None,
          X: Seq[Tensor[T1, Tuple3[Tt, Td, S]]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[Float, Tuple3[Tt1, Td1, S1]] = {
         val map: Map[String, Any] = Map("inputdimensions" -> inputdimensions)
         val allInputs             = Tuple.fromArray(X.toArray).asInstanceOf[Tuple]
         (callOp(name, "FeatureVectorizer", allInputs, map))
      }
   }

   // TODO: P3 constraints
   // split out to floats / ints
   trait ImputerV1 extends onnx.Operator {
      def ImputerV1[
          @sp T <: Float | Double | Long | Int: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: Dimension #: SNil
      ](
          name: String,
          imputed_value_floats: Option[(Array[Float])] = None,
          imputed_value_int64s: Option[(Array[Int])] = None,
          replaced_value_float: Option[(Float)] = None,
          replaced_value_int64: Option[(Int)] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[T, Tuple3[Tt1, Td1, S1]] = {
         val map: Map[String, Any] = Map(
           "imputed_value_floats" -> imputed_value_floats,
           "imputed_value_int64s" -> imputed_value_int64s,
           "replaced_value_float" -> replaced_value_float,
           "replaced_value_int64" -> replaced_value_int64
         )
         val allInputs = Tuple1(X)
         (callOp(name, "Imputer", allInputs, map))
      }
   }

   // TODO: P3 constraints
   // split out to floats / ints / strings ?
   trait LabelEncoderV2 extends onnx.Operator {
      def LabelEncoderV2[
          @sp T1 <: String | Long | Float: Numeric,
          @sp T2 <: String | Long | Float: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation
      ](
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
          X: Tensor[T1, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[T2, Tuple3[Tt1, Td, S]] = {
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
         val allInputs = Tuple1(X)
         (callOp(name, "LabelEncoder", allInputs, map))
      }
   }

   // TODO: P3 constraints
   // split out to strings / ints
   trait LinearClassifierV1 extends onnx.Operator {
      def LinearClassifierV1[
          @sp T1 <: Float | Double | Long | Int: Numeric,
          @sp T2 <: String | Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: Dimension #: SNil
      ](
          name: String,
          classlabels_ints: Option[(Array[Int])] = None,
          classlabels_strings: Option[(Array[String])] = None,
          coefficients: (Array[Float]),
          intercepts: Option[(Array[Float])] = None,
          multi_class: Option[(Int)] = None,
          post_transform: Option[(String)] = None,
          X: Tensor[T1, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[T2, Tuple3[Tt1, Td1, S1]] = {
         val map: Map[String, Any] = Map(
           "classlabels_ints"    -> classlabels_ints,
           "classlabels_strings" -> classlabels_strings,
           "coefficients"        -> coefficients,
           "intercepts"          -> intercepts,
           "multi_class"         -> multi_class,
           "post_transform"      -> post_transform
         )
         val allInputs = Tuple1(X)
         (callOp(name, "LinearClassifier", allInputs, map))
      }
   }
   // TODO: P3 constraints
   trait LinearRegressorV1 extends onnx.Operator {
      def LinearRegressorV1[
          @sp T <: Float | Double | Long | Int: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: Dimension #: SNil
      ](
          name: String,
          coefficients: Option[(Array[Float])] = None,
          intercepts: Option[(Array[Float])] = None,
          post_transform: Option[(String)] = None,
          targets: Option[(Int)] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[Float, Tuple3[Tt1, Td1, S1]] = {
         val map: Map[String, Any] = Map(
           "coefficients"   -> coefficients,
           "intercepts"     -> intercepts,
           "post_transform" -> post_transform,
           "targets"        -> targets
         )
         val allInputs = Tuple1(X)
         (callOp(name, "LinearRegressor", allInputs, map))
      }
   }

   trait NormalizerV1 extends onnx.Operator {
      def NormalizerV1[
          @sp T <: Float | Double | Long | Int: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: SNil
      ](
          name: String,
          norm: Option[(String)] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[Float, Tuple3[Tt, Td, S]] = {
         val map: Map[String, Any] = Map("norm" -> norm)
         val allInputs             = Tuple1(X)
         (callOp(name, "Normalizer", allInputs, map))
      }
   }

   // TODO: P3 constraints
   // split out to ints / strings
   trait OneHotEncoderV1 extends onnx.Operator {
      def OneHotEncoderV1[
          @sp T <: String | Long | Int | Float | Double: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Shape
      ](
          name: String,
          cats_int64s: Option[(Array[Int])] = None,
          cats_strings: Option[(Array[String])] = None,
          zeros: Option[(Int)] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[Float, Tuple3[Tt1, Td1, S1]] = {
         val map: Map[String, Any] =
            Map("cats_int64s" -> cats_int64s, "cats_strings" -> cats_strings, "zeros" -> zeros)
         val allInputs = Tuple1(X)
         (callOp(name, "OneHotEncoder", allInputs, map))
      }
   }

   // TODO: P3 constraints
   // split out to strings / ints
   trait SVMClassifierV1 extends onnx.Operator {
      def SVMClassifierV1[
          @sp T1 <: Float | Double | Long | Int: Numeric,
          @sp T2 <: String | Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: Dimension #: SNil
      ](
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
          X: Tensor[T1, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[T2, Tuple3[Tt1, Td1, S1]] = {
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
         val allInputs = Tuple1(X)
         (callOp(name, "SVMClassifier", allInputs, map))
      }
   }
   // TODO: P3 constraints
   trait SVMRegressorV1 extends onnx.Operator {
      def SVMRegressorV1[
          @sp T <: Float | Double | Long | Int: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Dimension #: Dimension #: SNil,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Dimension #: Dimension #: SNil
      ](
          name: String,
          coefficients: Option[(Array[Float])] = None,
          kernel_params: Option[(Array[Float])] = None,
          kernel_type: Option[(String)] = None,
          n_supports: Option[(Int)] = None,
          one_class: Option[(Int)] = None,
          post_transform: Option[(String)] = None,
          rho: Option[(Array[Float])] = None,
          support_vectors: Option[(Array[Float])] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[Float, Tuple3[Tt1, Td1, S1]] = {
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
         val allInputs = Tuple1(X)
         (callOp(name, "SVMRegressor", allInputs, map))
      }
   }

   // TODO: P3 constraints on offset / scaleAttr
   trait ScalerV1 extends onnx.Operator {
      def ScalerV1[
          @sp T <: Float | Double | Long | Int: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation
      ](
          name: String,
          offset: Option[(Array[Float])] = None,
          scaleAttr: Option[(Array[Float])] = None,
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td],
          s: ShapeOf[S]
      ): Tensor[Float, Tuple3[Tt1, Td, S]] = {
         val map: Map[String, Any] = Map("offset" -> offset, "scaleAttr" -> scaleAttr)
         val allInputs             = Tuple1(X)
         (callOp(name, "Scaler", allInputs, map))
      }
   }

   // TODO: P3 constraints
   // split out to strings / ints
   trait TreeEnsembleClassifierV1 extends onnx.Operator {
      def TreeEnsembleClassifierV1[
          @sp T1 <: Float | Double | Long | Int: Numeric,
          @sp T2 <: String | Long: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Shape
      ](
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
          X: Tensor[T1, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[T2, Tuple3[Tt1, Td1, S1]] = {
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
         val allInputs = Tuple1(X)
         (callOp(name, "TreeEnsembleClassifier", allInputs, map))
      }
   }

   // TODO: P3 constraints
   trait TreeEnsembleRegressorV1 extends onnx.Operator {
      def TreeEnsembleRegressorV1[
          @sp T <: Float | Double | Long | Int: Numeric,
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape,
          Tt1 <: TensorTypeDenotation,
          Td1 <: TensorShapeDenotation,
          S1 <: Shape
      ](
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
          X: Tensor[T, Tuple3[Tt, Td, S]]
      )(using
          tt: ValueOf[Tt1],
          td: TensorShapeDenotationOf[Td1],
          s: ShapeOf[S1]
      ): Tensor[Float, Tuple3[Tt1, Td1, S1]] = {
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
         val allInputs = Tuple1(X)
         (callOp(name, "TreeEnsembleRegressor", allInputs, map))
      }
   }

   // Not supported, ONNX ML - uses ONNX Map
   /*
  trait ZipMapV1 extends Operator {
    def ZipMapV1[@sp T <: Seq[Map[String, Float]] | Seq[Map[Long, Float]]: Numeric, Ax <: Axes](
        name: String,
        classlabels_int64s: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        X: Tensor[Float,_]
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): T = {
      val map: Map[String, Any] = Map(
        "classlabels_int64s"  -> classlabels_int64s,
        "classlabels_strings" -> classlabels_strings
      )
      val allInputs = Tuple1(X)
      (callOp(name, "ZipMap", allInputs, map))
    }
  }
    */
}

//Not supported, training is not yet GA
//ONNX domain: ai.onnx.preview.training
//See: https://github.com/onnx/onnx/blob/v1.8.1/docs/Operators.md#aionnxpreviewtraining
//Missing: Adam
package object onnxtraining {

   // Not yet supported, training has yet to GA
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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T3, Dx] = {
      val map: Map[String, Any] = Map(
        "decay_factor"     -> decay_factor,
        "epsilon"          -> epsilon,
        "norm_coefficient" -> norm_coefficient
      )
      val allInputs =
        Tuple2(R, T) ++ (Tuple.fromArray(inputs.toArray).asInstanceOf[Tuple])
      (callOp(name, "Adagrad", allInputs, map))
    }
  }

  trait GradientV1 extends Operator {
    def GradientV1[
        @sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[
          Float
        ] | Complex[Double]: Numeric,
        @sp T2 <: Float16 | Float | Double: Numeric
    , Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape](
        name: String,
        xs: (Array[String]),
        y: (String),
        zs: Option[(Array[String])] = None,
        Inputs: Seq[Tensor[T1,Tuple3[Tt, Td, S]]]
    )(using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[Td1], s: ShapeOf[S1]): Tensor[T2, Tuple3[Tt1,Td1,S1]] = {
      val map: Map[String, Any] = Map("xs" -> xs, "y" -> y, "zs" -> zs)
      val allInputs             = Tuple.fromArray(Inputs.toArray)
      (callOp(name, "Gradient", allInputs, map))
    }
  }

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
    )(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[T3, _] = {
      val map: Map[String, Any] = Map(
        "alpha"            -> alpha,
        "beta"             -> beta,
        "mode"             -> mode,
        "norm_coefficient" -> norm_coefficient
      )
      val allInputs =
        Tuple2(R, T) ++ (Tuple.fromArray(inputs.toArray).asInstanceOf[Tuple])
      (callOp(name, "Momentum", allInputs, map))
    }
  }
    */
}
