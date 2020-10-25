package org.emergentorder

import scala.language.higherKinds
import scala.{specialized => sp}
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import spire.implicits._
import spire.algebra.Field
import scala.reflect.ClassTag
import org.bytedeco.onnx.ModelProto
import org.emergentorder.onnx.Tensors._

package object onnx {

    //TODO: Use  monadless(except dead, find followup) / scala-async (with -Xasync?) / dotty-cps-async to replace for comprehensions
  //TODO: Remove requirement to be Numeric for ops with non-numeric outputs / inputs
  //TODO: Encode node names as types
  //TODO: fix encoding of type constraints, use Tensor as part of definition of types

  //Warning: Some data types not supported by ORT: eg Cos, Double
  type ![A]  = A => Nothing
  type !![A] = ![![A]]

  trait Disjunction[T] {
    type or[S]  = Disjunction[T with ![S]]
    type create = ![T]
  }

  type Union[T] = {
    type or[S] = Disjunction[![T]]#or[S]
  }

  type Contains[S, T] = !![S] <:< T

  type UNil

  trait Operator {
    def callOp[
        T: ClassTag,
        T1: ClassTag,
        T2: ClassTag,
        T3: ClassTag,
        T4: ClassTag,
        T5: ClassTag,
        T6: ClassTag,
        T7: ClassTag,
        T8: ClassTag,
        T9: ClassTag,
        T10: ClassTag,
        T11: ClassTag,
        T12: ClassTag,
        T13: ClassTag,
        T14: ClassTag,
        T15: ClassTag,
        T16: ClassTag,
        T17: ClassTag
    ](
        name: String,
        opName: String,
        inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
        //    outName: String,
        attrs: Map[String, Any]
    ): (T9)
  }

  abstract class Model(onnxBytes: Array[Byte]) extends Operator {
    def fullModel[
        T: ClassTag,
        T1: ClassTag,
        T2: ClassTag,
        T3: ClassTag,
        T4: ClassTag,
        T5: ClassTag,
        T6: ClassTag,
        T7: ClassTag,
        T8: ClassTag,
        T9: ClassTag,
        T10: ClassTag,
        T11: ClassTag,
        T12: ClassTag,
        T13: ClassTag,
        T14: ClassTag,
        T15: ClassTag,
        T16: ClassTag,
        T17: ClassTag
    ](
        inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8]
    ): (T9)
  }

  trait Graph
  trait DataSource {
    def getParams[T: Numeric: ClassTag](name: String): Tensor[T]
  }
  trait Abs extends Operator {

    def Abs1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Abs", allInputs, map))
    }

    def Abs6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Abs", allInputs, map))
    }
  }
  trait Acos extends Operator {

    def Acos7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Acos", allInputs, map))
    }
  }
  trait Acosh extends Operator {

    def Acosh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Acosh", allInputs, map))
    }
  }
  trait Add extends Operator {

    def Add1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "broadcast" -> broadcast, "consumed_inputs" -> consumed_inputs)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Add", allInputs, map))
    }

    def Add6[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Add", allInputs, map))
    }

    def Add7[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Add", allInputs, map))
    }
  }
  trait And extends Operator {

    def And1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "And", allInputs, map))
    }

    def And7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "And", allInputs, map))
    }
  }
  trait ArgMax extends Operator {

    def ArgMax1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[Long]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Long],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ArgMax", allInputs, map))
    }

    def ArgMax11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[Long]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Long],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ArgMax", allInputs, map))
    }
  }
  trait ArgMin extends Operator {

    def ArgMin1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[Long]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Long],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ArgMin", allInputs, map))
    }

    def ArgMin11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[Long]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Long],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ArgMin", allInputs, map))
    }
  }
  trait ArrayFeatureExtractor extends Operator {

    def ArrayFeatureExtractor1[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        Y: Option[Tensor[Long]]
    )(implicit
        evT: Contains[
          T,
          Union[Float]#or[Double]#or[Long]#or[Int]#or[String]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        Y,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[Option[Tensor[T]], Option[Tensor[Long]], Any, Any, Any, Any, Any, Any, Any, Tensor[
          T
        ], Any, Any, Any, Any, Any, Any, Any, Any](
          name,
          "ArrayFeatureExtractor",
          allInputs,
          map
        )
      )
    }
  }
  trait Asin extends Operator {

    def Asin7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Asin", allInputs, map))
    }
  }
  trait Asinh extends Operator {

    def Asinh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Asinh", allInputs, map))
    }
  }
  trait Atan extends Operator {

    def Atan7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Atan", allInputs, map))
    }
  }
  trait Atanh extends Operator {

    def Atanh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Atanh", allInputs, map))
    }
  }
  trait AveragePool extends Operator {

    def AveragePool1[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "AveragePool", allInputs, map))
    }

    def AveragePool7[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"          -> auto_pad,
        "count_include_pad" -> count_include_pad,
        "kernel_shape"      -> kernel_shape,
        "pads"              -> pads,
        "strides"           -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "AveragePool", allInputs, map))
    }

    def AveragePool10[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"          -> auto_pad,
        "ceil_mode"         -> ceil_mode,
        "count_include_pad" -> count_include_pad,
        "kernel_shape"      -> kernel_shape,
        "pads"              -> pads,
        "strides"           -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "AveragePool", allInputs, map))
    }

    def AveragePool11[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"          -> auto_pad,
        "ceil_mode"         -> ceil_mode,
        "count_include_pad" -> count_include_pad,
        "kernel_shape"      -> kernel_shape,
        "pads"              -> pads,
        "strides"           -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "AveragePool", allInputs, map))
    }
  }
  trait BatchNormalization extends Operator {

    def BatchNormalization1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])],
        epsilon: Option[(Float)] = None,
        is_test: Option[(Int)] = None,
        momentum: Option[(Float)] = None,
        spatial: Option[(Int)] = None,
        X: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]],
        mean: Option[Tensor[T]],
        someVar: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "consumed_inputs" -> consumed_inputs,
        "epsilon"         -> epsilon,
        "is_test"         -> is_test,
        "momentum"        -> momentum,
        "spatial"         -> spatial
      )
      val allInputs = (
        X,
        scale,
        B,
        mean,
        someVar,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "BatchNormalization",
          allInputs,
          map
        )
      )
    }

    def BatchNormalization6[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        is_test: Option[(Int)] = None,
        momentum: Option[(Float)] = None,
        spatial: Option[(Int)] = None,
        X: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]],
        mean: Option[Tensor[T]],
        someVar: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "epsilon"  -> epsilon,
        "is_test"  -> is_test,
        "momentum" -> momentum,
        "spatial"  -> spatial
      )
      val allInputs = (
        X,
        scale,
        B,
        mean,
        someVar,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "BatchNormalization",
          allInputs,
          map
        )
      )
    }

    def BatchNormalization7[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        momentum: Option[(Float)] = None,
        spatial: Option[(Int)] = None,
        X: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]],
        mean: Option[Tensor[T]],
        someVar: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("epsilon" -> epsilon, "momentum" -> momentum, "spatial" -> spatial)
      val allInputs = (
        X,
        scale,
        B,
        mean,
        someVar,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "BatchNormalization",
          allInputs,
          map
        )
      )
    }

    def BatchNormalization9[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        momentum: Option[(Float)] = None,
        X: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]],
        mean: Option[Tensor[T]],
        someVar: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("epsilon" -> epsilon, "momentum" -> momentum)
      val allInputs = (
        X,
        scale,
        B,
        mean,
        someVar,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "BatchNormalization",
          allInputs,
          map
        )
      )
    }
  }
  trait Binarizer extends Operator {

    def Binarizer1[@sp T: Numeric: ClassTag](
        name: String,
        threshold: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("threshold" -> threshold)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Binarizer", allInputs, map))
    }
  }
  trait BitShift extends Operator {

    def BitShift11[@sp T: Numeric: ClassTag](
        name: String,
        direction: Option[(String)],
        X: Option[Tensor[T]],
        Y: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("direction" -> direction)
      val allInputs = (
        X,
        Y,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "BitShift", allInputs, map))
    }
  }
  trait Cast extends Operator {

    def Cast1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(String)],
        input: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[
          T1,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[String]#or[UNil]#create
        ],
        evT2: Contains[
          T2,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[String]#or[UNil]#create
        ]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map("to" -> to)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Cast", allInputs, map))
    }

    def Cast6[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(Int)],
        input: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[
          T1,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[String]#or[UNil]#create
        ],
        evT2: Contains[
          T2,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[String]#or[UNil]#create
        ]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map("to" -> to)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Cast", allInputs, map))
    }

    def Cast9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(Int)],
        input: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[
          T1,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[String]#or[UNil]#create
        ],
        evT2: Contains[
          T2,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[String]#or[UNil]#create
        ]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map("to" -> to)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Cast", allInputs, map))
    }
  }
  trait CastMap extends Operator {

    def CastMap1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        cast_to: Option[(String)] = None,
        map_form: Option[(String)] = None,
        max_map: Option[(Int)] = None,
        X: Option[T1]
    )(implicit
        evT1: Contains[T1, Union[Map[Long, String]]#or[Map[Long, Float]]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Float]#or[Long]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] =
        Map("cast_to" -> cast_to, "map_form" -> map_form, "max_map" -> max_map)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "CastMap", allInputs, map))
    }
  }
  trait CategoryMapper extends Operator {

    def CategoryMapper1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        cats_int64s: Option[(Array[Int])] = None,
        cats_strings: Option[(Array[String])] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        X: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[String]#or[Long]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map(
        "cats_int64s"    -> cats_int64s,
        "cats_strings"   -> cats_strings,
        "default_int64"  -> default_int64,
        "default_string" -> default_string
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "CategoryMapper", allInputs, map))
    }
  }
  trait Ceil extends Operator {

    def Ceil1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Ceil", allInputs, map))
    }

    def Ceil6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Ceil", allInputs, map))
    }
  }
  trait Clip extends Operator {

    def Clip1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        max: Option[(Float)] = None,
        min: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("consumed_inputs" -> consumed_inputs, "max" -> max, "min" -> min)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Clip", allInputs, map))
    }

    def Clip6[@sp T: Numeric: ClassTag](
        name: String,
        max: Option[(Float)] = None,
        min: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("max" -> max, "min" -> min)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Clip", allInputs, map))
    }

    def Clip11[@sp T: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        min: Option[Tensor[T]] = None,
        max: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        min,
        max,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Clip",
          allInputs,
          map
        )
      )
    }
  }
  trait Compress extends Operator {

    def Compress9[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]],
        condition: Option[Tensor[T1]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        condition,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Compress", allInputs, map))
    }

    def Compress11[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]],
        condition: Option[Tensor[T1]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        condition,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Compress", allInputs, map))
    }
  }
  trait Concat extends Operator {

    def Concat1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        inputs: Seq[Option[Tensor[T]]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        inputs.lift(0).flatten,
        inputs.lift(1).flatten,
        inputs.lift(2).flatten,
        inputs.lift(3).flatten,
        inputs.lift(4).flatten,
        inputs.lift(5).flatten,
        inputs.lift(6).flatten,
        inputs.lift(7).flatten,
        inputs.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Concat",
          allInputs,
          map
        )
      )
    }

    def Concat4[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)],
        inputs: Seq[Option[Tensor[T]]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        inputs.lift(0).flatten,
        inputs.lift(1).flatten,
        inputs.lift(2).flatten,
        inputs.lift(3).flatten,
        inputs.lift(4).flatten,
        inputs.lift(5).flatten,
        inputs.lift(6).flatten,
        inputs.lift(7).flatten,
        inputs.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Concat",
          allInputs,
          map
        )
      )
    }

    def Concat11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)],
        inputs: Seq[Option[Tensor[T]]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        inputs.lift(0).flatten,
        inputs.lift(1).flatten,
        inputs.lift(2).flatten,
        inputs.lift(3).flatten,
        inputs.lift(4).flatten,
        inputs.lift(5).flatten,
        inputs.lift(6).flatten,
        inputs.lift(7).flatten,
        inputs.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Concat",
          allInputs,
          map
        )
      )
    }
  }
  trait ConcatFromSequence extends Operator {

    def ConcatFromSequence11[@sp S: Numeric: ClassTag, @sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)],
        new_axis: Option[(Int)] = None,
        input_sequence: Option[S]
    )(implicit
        evS: Contains[
          S,
          Union[Seq[Tensor[UByte]]]#or[Seq[Tensor[UShort]]]#or[Seq[Tensor[UInt]]]#or[Seq[Tensor[
            ULong
          ]]]#or[Seq[Tensor[Byte]]]#or[Seq[Tensor[Short]]]#or[Seq[Tensor[Int]]]#or[Seq[
            Tensor[Long]
          ]]#or[Seq[Tensor[Float16]]]#or[Seq[Tensor[Float]]]#or[Seq[Tensor[Double]]]#or[Seq[
            Tensor[String]
          ]]#or[Seq[Tensor[Boolean]]]#or[Seq[Tensor[Complex[Float]]]]#or[Seq[
            Tensor[Complex[Double]]
          ]]#or[UNil]#create
        ],
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "new_axis" -> new_axis)
      val allInputs = (
        input_sequence,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[S],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ConcatFromSequence", allInputs, map))
    }
  }
  trait Constant extends Operator {

    def Constant1[@sp T: Numeric: ClassTag](name: String, value: Option[(Tensor[T])])(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("value" -> value)
      val allInputs = (
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Constant", allInputs, map))
    }

    def Constant9[@sp T: Numeric: ClassTag](name: String, value: Option[(Tensor[T])])(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("value" -> value)
      val allInputs = (
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Constant", allInputs, map))
    }

    def Constant11[@sp T: Numeric: ClassTag](
        name: String,
        sparse_value: Option[(SparseTensor[T])] = None,
        value: Option[(Tensor[T])] = None
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("sparse_value" -> sparse_value, "value" -> value)
      val allInputs = (
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Constant", allInputs, map))
    }
  }
  trait ConstantOfShape extends Operator {

    def ConstantOfShape9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        value: Option[(Tensor[T2])] = None,
        input: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[Long]#or[UNil]#create],
        evT2: Contains[
          T2,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[UNil]#create
        ]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map("value" -> value)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ConstantOfShape", allInputs, map))
    }
  }
  trait Conv extends Operator {

    def Conv1[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        B: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (
        X,
        W,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Conv",
          allInputs,
          map
        )
      )
    }

    def Conv11[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        B: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (
        X,
        W,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Conv",
          allInputs,
          map
        )
      )
    }
  }
  trait ConvInteger extends Operator {

    def ConvInteger10[
        @sp T1: Numeric: ClassTag,
        @sp T2: Numeric: ClassTag,
        @sp T3: Numeric: ClassTag
    ](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        x: Option[Tensor[T1]],
        w: Option[Tensor[T2]],
        x_zero_point: Option[Tensor[T1]] = None,
        w_zero_point: Option[Tensor[T2]] = None
    )(implicit
        evT1: Contains[T1, Union[Byte]#or[UByte]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create],
        evT3: Contains[T3, Union[Int]#or[UNil]#create]
    ): (Tensor[T3]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (
        x,
        w,
        x_zero_point,
        w_zero_point,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T3],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "ConvInteger",
          allInputs,
          map
        )
      )
    }
  }
  trait ConvTranspose extends Operator {

    def ConvTranspose1[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        output_padding: Option[(Array[Int])] = None,
        output_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        B: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
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
      val allInputs = (
        X,
        W,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "ConvTranspose",
          allInputs,
          map
        )
      )
    }

    def ConvTranspose11[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        output_padding: Option[(Array[Int])] = None,
        output_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        B: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
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
      val allInputs = (
        X,
        W,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "ConvTranspose",
          allInputs,
          map
        )
      )
    }
  }
  trait Cos extends Operator {

    def Cos7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Cos", allInputs, map))
    }
  }
  trait Cosh extends Operator {

    def Cosh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Cosh", allInputs, map))
    }
  }
  trait CumSum extends Operator {

    def CumSum11[@sp T: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        exclusive: Option[(Int)] = None,
        reverse: Option[(Int)] = None,
        x: Option[Tensor[T]],
        axis: Option[Tensor[T2]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float]#or[Double]#or[UNil]#create
        ],
        evT2: Contains[T2, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("exclusive" -> exclusive, "reverse" -> reverse)
      val allInputs = (
        x,
        axis,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T2]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "CumSum", allInputs, map))
    }
  }
  trait DepthToSpace extends Operator {

    def DepthToSpace1[@sp T: Numeric: ClassTag](
        name: String,
        blocksize: Option[(Int)],
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("blocksize" -> blocksize)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "DepthToSpace", allInputs, map))
    }

    def DepthToSpace11[@sp T: Numeric: ClassTag](
        name: String,
        blocksize: Option[(Int)],
        mode: Option[(String)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("blocksize" -> blocksize, "mode" -> mode)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "DepthToSpace", allInputs, map))
    }
  }
  trait DequantizeLinear extends Operator {

    def DequantizeLinear10[@sp T: Numeric: ClassTag](
        name: String,
        x: Option[Tensor[T]],
        x_scale: Option[Tensor[Float]],
        x_zero_point: Option[Tensor[T]] = None
    )(implicit evT: Contains[T, Union[Byte]#or[UByte]#or[Int]#or[UNil]#create]): (Tensor[Float]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        x,
        x_scale,
        x_zero_point,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[Float]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[Float],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "DequantizeLinear",
          allInputs,
          map
        )
      )
    }
  }
  trait Det extends Operator {

    def Det11[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Det", allInputs, map))
    }
  }
  trait DictVectorizer extends Operator {

    def DictVectorizer1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        int64_vocabulary: Option[(Array[Int])] = None,
        string_vocabulary: Option[(Array[String])] = None,
        X: Option[T1]
    )(implicit
        evT1: Contains[T1, Union[Map[String, Long]]#or[Map[Long, String]]#or[Map[
          Long,
          Float
        ]]#or[Map[Long, Double]]#or[Map[String, Float]]#or[Map[String, Double]]#or[UNil]#create],
        evT2: Contains[T2, Union[Long]#or[Float]#or[Double]#or[String]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] =
        Map("int64_vocabulary" -> int64_vocabulary, "string_vocabulary" -> string_vocabulary)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "DictVectorizer", allInputs, map))
    }
  }
  trait Div extends Operator {

    def Div1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "broadcast" -> broadcast, "consumed_inputs" -> consumed_inputs)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Div", allInputs, map))
    }

    def Div6[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Div", allInputs, map))
    }

    def Div7[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Div", allInputs, map))
    }
  }
  trait Dropout extends Operator {

    def Dropout1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("consumed_inputs" -> consumed_inputs, "is_test" -> is_test, "ratio" -> ratio)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Dropout", allInputs, map))
    }

    def Dropout6[@sp T: Numeric: ClassTag](
        name: String,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("is_test" -> is_test, "ratio" -> ratio)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Dropout", allInputs, map))
    }

    def Dropout7[@sp T: Numeric: ClassTag](
        name: String,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("ratio" -> ratio)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Dropout", allInputs, map))
    }

    def Dropout10[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("ratio" -> ratio)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Dropout", allInputs, map))
    }
  }
  trait DynamicQuantizeLinear extends Operator {

    def DynamicQuantizeLinear11[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        x: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[Float]#or[UNil]#create],
        evT2: Contains[T2, Union[UByte]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        x,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[Option[Tensor[T1]], Any, Any, Any, Any, Any, Any, Any, Any, Tensor[T2], Tensor[
          Float
        ], Tensor[T2], Any, Any, Any, Any, Any, Any](
          name,
          "DynamicQuantizeLinear",
          allInputs,
          map
        )
      )
    }
  }
  trait Elu extends Operator {

    def Elu1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Elu", allInputs, map))
    }

    def Elu6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("alpha" -> alpha)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Elu", allInputs, map))
    }
  }
  trait Equal extends Operator {

    def Equal1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[
          ULong
        ]#or[Byte]#or[Short]#or[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Equal", allInputs, map))
    }

    def Equal7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[
          ULong
        ]#or[Byte]#or[Short]#or[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Equal", allInputs, map))
    }

    def Equal11[@sp T: Numeric: ClassTag, @sp T1: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[
          ULong
        ]#or[Byte]#or[Short]#or[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Equal", allInputs, map))
    }
  }
  trait Erf extends Operator {

    def Erf9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Erf", allInputs, map))
    }
  }
  trait Exp extends Operator {

    def Exp1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Exp", allInputs, map))
    }

    def Exp6[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Exp", allInputs, map))
    }
  }
  trait Expand extends Operator {

    def Expand8[@sp T: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        shapeInput: Option[Tensor[Long]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        shapeInput,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[Long]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Expand", allInputs, map))
    }
  }
  trait EyeLike extends Operator {

    def EyeLike9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        k: Option[(Int)] = None,
        input: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[
          T1,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[UNil]#create
        ],
        evT2: Contains[
          T2,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[UNil]#create
        ]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map("dtype" -> dtype, "k" -> k)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "EyeLike", allInputs, map))
    }
  }
  trait FeatureVectorizer extends Operator {

    def FeatureVectorizer1[@sp T1: Numeric: ClassTag](
        name: String,
        inputdimensions: Option[(Array[Int])] = None,
        X: Seq[Option[Tensor[T1]]]
    )(implicit
        evT1: Contains[T1, Union[Int]#or[Long]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[Float]) = {
      val map: Map[String, Any] = Map("inputdimensions" -> inputdimensions)
      val allInputs = (
        X.lift(0).flatten,
        X.lift(1).flatten,
        X.lift(2).flatten,
        X.lift(3).flatten,
        X.lift(4).flatten,
        X.lift(5).flatten,
        X.lift(6).flatten,
        X.lift(7).flatten,
        X.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Tensor[Float],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "FeatureVectorizer",
          allInputs,
          map
        )
      )
    }
  }
  trait Flatten extends Operator {

    def Flatten1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Flatten", allInputs, map))
    }

    def Flatten9[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Flatten", allInputs, map))
    }

    def Flatten11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Flatten", allInputs, map))
    }
  }
  trait Floor extends Operator {

    def Floor1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Floor", allInputs, map))
    }

    def Floor6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Floor", allInputs, map))
    }
  }
  trait GRU extends Operator {

    def GRU1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        output_sequence: Option[(Int)] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        R: Option[Tensor[T]],
        B: Option[Tensor[T]] = None,
        sequence_lens: Option[Tensor[T1]] = None,
        initial_h: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size,
        "output_sequence"  -> output_sequence
      )
      val allInputs = (
        X,
        W,
        R,
        B,
        sequence_lens,
        initial_h,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T1]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "GRU",
          allInputs,
          map
        )
      )
    }

    def GRU3[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        linear_before_reset: Option[(Int)] = None,
        output_sequence: Option[(Int)] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        R: Option[Tensor[T]],
        B: Option[Tensor[T]] = None,
        sequence_lens: Option[Tensor[T1]] = None,
        initial_h: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
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
      val allInputs = (
        X,
        W,
        R,
        B,
        sequence_lens,
        initial_h,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T1]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "GRU",
          allInputs,
          map
        )
      )
    }

    def GRU7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        linear_before_reset: Option[(Int)] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        R: Option[Tensor[T]],
        B: Option[Tensor[T]] = None,
        sequence_lens: Option[Tensor[T1]] = None,
        initial_h: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "activation_alpha"    -> activation_alpha,
        "activation_beta"     -> activation_beta,
        "activations"         -> activations,
        "clip"                -> clip,
        "direction"           -> direction,
        "hidden_size"         -> hidden_size,
        "linear_before_reset" -> linear_before_reset
      )
      val allInputs = (
        X,
        W,
        R,
        B,
        sequence_lens,
        initial_h,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T1]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "GRU",
          allInputs,
          map
        )
      )
    }
  }
  trait Gather extends Operator {

    def Gather1[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        data,
        indices,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[Tind]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Gather", allInputs, map))
    }

    def Gather11[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        data,
        indices,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[Tind]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Gather", allInputs, map))
    }
  }
  trait GatherElements extends Operator {

    def GatherElements11[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        data,
        indices,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[Option[Tensor[T]], Option[Tensor[Tind]], Any, Any, Any, Any, Any, Any, Any, Tensor[
          T
        ], Any, Any, Any, Any, Any, Any, Any, Any](
          name,
          "GatherElements",
          allInputs,
          map
        )
      )
    }
  }
  trait GatherND extends Operator {

    def GatherND11[@sp T: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Long]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data,
        indices,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[Long]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "GatherND", allInputs, map))
    }
  }
  trait Gemm extends Operator {

    def Gemm1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        broadcast: Option[(Int)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "alpha"     -> alpha,
        "beta"      -> beta,
        "broadcast" -> broadcast,
        "transA"    -> transA,
        "transB"    -> transB
      )
      val allInputs = (
        A,
        B,
        C,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Gemm",
          allInputs,
          map
        )
      )
    }

    def Gemm6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        broadcast: Option[(Int)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "alpha"     -> alpha,
        "beta"      -> beta,
        "broadcast" -> broadcast,
        "transA"    -> transA,
        "transB"    -> transB
      )
      val allInputs = (
        A,
        B,
        C,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Gemm",
          allInputs,
          map
        )
      )
    }

    def Gemm7[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "transA" -> transA, "transB" -> transB)
      val allInputs = (
        A,
        B,
        C,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Gemm",
          allInputs,
          map
        )
      )
    }

    def Gemm9[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "transA" -> transA, "transB" -> transB)
      val allInputs = (
        A,
        B,
        C,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Gemm",
          allInputs,
          map
        )
      )
    }

    def Gemm11[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "transA" -> transA, "transB" -> transB)
      val allInputs = (
        A,
        B,
        C,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Gemm",
          allInputs,
          map
        )
      )
    }
  }
  trait GlobalAveragePool extends Operator {

    def GlobalAveragePool1[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "GlobalAveragePool", allInputs, map))
    }
  }
  trait GlobalLpPool extends Operator {

    def GlobalLpPool1[@sp T: Numeric: ClassTag](
        name: String,
        p: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("p" -> p)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "GlobalLpPool", allInputs, map))
    }

    def GlobalLpPool2[@sp T: Numeric: ClassTag](
        name: String,
        p: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("p" -> p)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "GlobalLpPool", allInputs, map))
    }
  }
  trait GlobalMaxPool extends Operator {

    def GlobalMaxPool1[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "GlobalMaxPool", allInputs, map))
    }
  }

  trait GreaterOrEqual extends Operator {
    def GreaterOrEqual12[
        @sp T : Numeric: ClassTag,
        @sp T1 : ClassTag
    ](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]]) 
        (implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ], 
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      ) 
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "GreaterOrEqual", allInputs, map))
    }
  }



  trait Greater extends Operator {

    def Greater1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Greater", allInputs, map))
    }

    def Greater7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Greater", allInputs, map))
    }

    def Greater9[@sp T: Numeric: ClassTag, @sp T1: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Greater", allInputs, map))
    }
  }
  trait HardSigmoid extends Operator {

    def HardSigmoid1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "HardSigmoid", allInputs, map))
    }

    def HardSigmoid6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "beta" -> beta)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "HardSigmoid", allInputs, map))
    }
  }
  trait Hardmax extends Operator {

    def Hardmax1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Hardmax", allInputs, map))
    }

    def Hardmax11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Hardmax", allInputs, map))
    }
  }
  trait Identity extends Operator {

    def Identity1[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Identity", allInputs, map))
    }
  }
  trait If extends Operator {

    def If1[@sp B: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        else_branch: Option[(Graph)],
        then_branch: Option[(Graph)],
        cond: Option[Tensor[B]]
    )(implicit
        evB: Contains[B, Union[Boolean]#or[UNil]#create],
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[V]) = {
      val map: Map[String, Any] = Map("else_branch" -> else_branch, "then_branch" -> then_branch)
      val allInputs = (
        cond,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[B]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[V],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "If", allInputs, map))
    }

    def If11[@sp B: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        else_branch: Option[(Graph)],
        then_branch: Option[(Graph)],
        cond: Option[Tensor[B]]
    )(implicit
        evB: Contains[B, Union[Boolean]#or[UNil]#create],
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[V]) = {
      val map: Map[String, Any] = Map("else_branch" -> else_branch, "then_branch" -> then_branch)
      val allInputs = (
        cond,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[B]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[V],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "If", allInputs, map))
    }
  }
  trait Imputer extends Operator {

    def Imputer1[@sp T: Numeric: ClassTag](
        name: String,
        imputed_value_floats: Option[(Array[Float])] = None,
        imputed_value_int64s: Option[(Array[Int])] = None,
        replaced_value_float: Option[(Float)] = None,
        replaced_value_int64: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "imputed_value_floats" -> imputed_value_floats,
        "imputed_value_int64s" -> imputed_value_int64s,
        "replaced_value_float" -> replaced_value_float,
        "replaced_value_int64" -> replaced_value_int64
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Imputer", allInputs, map))
    }
  }
  trait InstanceNormalization extends Operator {

    def InstanceNormalization1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        epsilon: Option[(Float)] = None,
        input: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs, "epsilon" -> epsilon)
      val allInputs = (
        input,
        scale,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "InstanceNormalization",
          allInputs,
          map
        )
      )
    }

    def InstanceNormalization6[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        input: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("epsilon" -> epsilon)
      val allInputs = (
        input,
        scale,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "InstanceNormalization",
          allInputs,
          map
        )
      )
    }
  }
  trait IsInf extends Operator {

    def IsInf10[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        detect_negative: Option[(Int)] = None,
        detect_positive: Option[(Int)] = None,
        X: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] =
        Map("detect_negative" -> detect_negative, "detect_positive" -> detect_positive)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "IsInf", allInputs, map))
    }
  }
  trait IsNaN extends Operator {

    def IsNaN9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "IsNaN", allInputs, map))
    }
  }
  trait LRN extends Operator {

    def LRN1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        bias: Option[(Float)] = None,
        size: Option[(Int)],
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "beta" -> beta, "bias" -> bias, "size" -> size)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LRN", allInputs, map))
    }
  }
  trait LSTM extends Operator {

    def LSTM1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        input_forget: Option[(Int)] = None,
        output_sequence: Option[(Int)] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        R: Option[Tensor[T]],
        B: Option[Tensor[T]] = None,
        sequence_lens: Option[Tensor[T1]] = None,
        initial_h: Option[Tensor[T]] = None,
        initial_c: Option[Tensor[T]] = None,
        P: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
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
      val allInputs = (X, W, R, B, sequence_lens, initial_h, initial_c, P, None: Option[Any])
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T1]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "LSTM",
          allInputs,
          map
        )
      )
    }

    def LSTM7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        input_forget: Option[(Int)] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        R: Option[Tensor[T]],
        B: Option[Tensor[T]] = None,
        sequence_lens: Option[Tensor[T1]] = None,
        initial_h: Option[Tensor[T]] = None,
        initial_c: Option[Tensor[T]] = None,
        P: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size,
        "input_forget"     -> input_forget
      )
      val allInputs = (X, W, R, B, sequence_lens, initial_h, initial_c, P, None: Option[Any])
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T1]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Tensor[T],
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "LSTM",
          allInputs,
          map
        )
      )
    }
  }
  trait LabelEncoder extends Operator {

    def LabelEncoder1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        classes_strings: Option[(Array[String])] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        X: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[String]#or[Long]#or[Float]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[Float]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map(
        "classes_strings" -> classes_strings,
        "default_int64"   -> default_int64,
        "default_string"  -> default_string
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LabelEncoder", allInputs, map))
    }

    def LabelEncoder2[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
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
        X: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[String]#or[Long]#or[Float]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[Float]#or[UNil]#create]
    ): (Tensor[T2]) = {
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
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LabelEncoder", allInputs, map))
    }
  }
  trait LeakyRelu extends Operator {

    def LeakyRelu1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LeakyRelu", allInputs, map))
    }

    def LeakyRelu6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("alpha" -> alpha)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LeakyRelu", allInputs, map))
    }
  }

  trait LessOrEqual extends Operator {
    def LessOrEqual12[
        @sp T : Numeric: ClassTag,
        @sp T1 : ClassTag
    ](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]]) 
        (implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ], 
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      ) 
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LessOrEqual", allInputs, map))
    }
  }

  trait Less extends Operator {

    def Less1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Less", allInputs, map))
    }

    def Less7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Less", allInputs, map))
    }

    def Less9[@sp T: Numeric: ClassTag, @sp T1: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Less", allInputs, map))
    }
  }
  trait LinearClassifier extends Operator {

    def LinearClassifier1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        classlabels_ints: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        coefficients: Option[(Array[Float])],
        intercepts: Option[(Array[Float])] = None,
        multi_class: Option[(Int)] = None,
        post_transform: Option[(String)] = None,
        X: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map(
        "classlabels_ints"    -> classlabels_ints,
        "classlabels_strings" -> classlabels_strings,
        "coefficients"        -> coefficients,
        "intercepts"          -> intercepts,
        "multi_class"         -> multi_class,
        "post_transform"      -> post_transform
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[Option[Tensor[T1]], Any, Any, Any, Any, Any, Any, Any, Any, Tensor[T2], Tensor[
          Float
        ], Any, Any, Any, Any, Any, Any, Any](
          name,
          "LinearClassifier",
          allInputs,
          map
        )
      )
    }
  }
  trait LinearRegressor extends Operator {

    def LinearRegressor1[@sp T: Numeric: ClassTag](
        name: String,
        coefficients: Option[(Array[Float])] = None,
        intercepts: Option[(Array[Float])] = None,
        post_transform: Option[(String)] = None,
        targets: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): (Tensor[Float]) = {
      val map: Map[String, Any] = Map(
        "coefficients"   -> coefficients,
        "intercepts"     -> intercepts,
        "post_transform" -> post_transform,
        "targets"        -> targets
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Float],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LinearRegressor", allInputs, map))
    }
  }
  trait Log extends Operator {

    def Log1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Log", allInputs, map))
    }

    def Log6[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Log", allInputs, map))
    }
  }
  trait LogSoftmax extends Operator {

    def LogSoftmax1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LogSoftmax", allInputs, map))
    }

    def LogSoftmax11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LogSoftmax", allInputs, map))
    }
  }
  trait Loop extends Operator {

    def Loop1[@sp I: Numeric: ClassTag, @sp B: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        M: Option[Tensor[I]] = None,
        cond: Option[Tensor[B]] = None,
        v_initial: Seq[Option[Tensor[V]]]
    )(implicit
        evI: Contains[I, Union[Long]#or[UNil]#create],
        evB: Contains[B, Union[Boolean]#or[UNil]#create],
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[V]) = {
      val map: Map[String, Any] = Map("body" -> body)
      val allInputs = (
        M,
        cond,
        v_initial.lift(0).flatten,
        v_initial.lift(1).flatten,
        v_initial.lift(2).flatten,
        v_initial.lift(3).flatten,
        v_initial.lift(4).flatten,
        v_initial.lift(5).flatten,
        v_initial.lift(6).flatten
      )
      (
        callOp[
          Option[Tensor[I]],
          Option[Tensor[B]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Tensor[V],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Loop",
          allInputs,
          map
        )
      )
    }

    def Loop11[@sp I: Numeric: ClassTag, @sp B: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        M: Option[Tensor[I]] = None,
        cond: Option[Tensor[B]] = None,
        v_initial: Seq[Option[Tensor[V]]]
    )(implicit
        evI: Contains[I, Union[Long]#or[UNil]#create],
        evB: Contains[B, Union[Boolean]#or[UNil]#create],
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[V]) = {
      val map: Map[String, Any] = Map("body" -> body)
      val allInputs = (
        M,
        cond,
        v_initial.lift(0).flatten,
        v_initial.lift(1).flatten,
        v_initial.lift(2).flatten,
        v_initial.lift(3).flatten,
        v_initial.lift(4).flatten,
        v_initial.lift(5).flatten,
        v_initial.lift(6).flatten
      )
      (
        callOp[
          Option[Tensor[I]],
          Option[Tensor[B]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Tensor[V],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Loop",
          allInputs,
          map
        )
      )
    }
  }
  trait LpNormalization extends Operator {

    def LpNormalization1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        p: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "p" -> p)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LpNormalization", allInputs, map))
    }
  }
  trait LpPool extends Operator {

    def LpPool1[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        p: Option[(Float)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "p"            -> p,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LpPool", allInputs, map))
    }

    def LpPool2[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        p: Option[(Int)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "p"            -> p,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LpPool", allInputs, map))
    }

    def LpPool11[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        p: Option[(Int)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "p"            -> p,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "LpPool", allInputs, map))
    }
  }
  trait MatMul extends Operator {

    def MatMul1[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "MatMul", allInputs, map))
    }

    def MatMul9[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "MatMul", allInputs, map))
    }
  }
  trait MatMulInteger extends Operator {

    def MatMulInteger10[
        @sp T1: Numeric: ClassTag,
        @sp T2: Numeric: ClassTag,
        @sp T3: Numeric: ClassTag
    ](
        name: String,
        A: Option[Tensor[T1]],
        B: Option[Tensor[T2]],
        a_zero_point: Option[Tensor[T1]] = None,
        b_zero_point: Option[Tensor[T2]] = None
    )(implicit
        evT1: Contains[T1, Union[Byte]#or[UByte]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create],
        evT3: Contains[T3, Union[Int]#or[UNil]#create]
    ): (Tensor[T3]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        a_zero_point,
        b_zero_point,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T3],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "MatMulInteger",
          allInputs,
          map
        )
      )
    }
  }
  trait Max extends Operator {

    def Max1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        data_0: Seq[Option[Tensor[T]]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Max",
          allInputs,
          map
        )
      )
    }

    def Max6[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Max",
          allInputs,
          map
        )
      )
    }

    def Max8[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Max",
          allInputs,
          map
        )
      )
    }
  }
  trait MaxPool extends Operator {

    def MaxPool1[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "MaxPool", allInputs, map))
    }

    def MaxPool8[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"      -> auto_pad,
        "kernel_shape"  -> kernel_shape,
        "pads"          -> pads,
        "storage_order" -> storage_order,
        "strides"       -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[I],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "MaxPool", allInputs, map))
    }

    def MaxPool10[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        dilations: Option[(Array[Int])] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"      -> auto_pad,
        "ceil_mode"     -> ceil_mode,
        "dilations"     -> dilations,
        "kernel_shape"  -> kernel_shape,
        "pads"          -> pads,
        "storage_order" -> storage_order,
        "strides"       -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[I],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "MaxPool", allInputs, map))
    }

    def MaxPool11[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        dilations: Option[(Array[Int])] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"      -> auto_pad,
        "ceil_mode"     -> ceil_mode,
        "dilations"     -> dilations,
        "kernel_shape"  -> kernel_shape,
        "pads"          -> pads,
        "storage_order" -> storage_order,
        "strides"       -> strides
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[I],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "MaxPool", allInputs, map))
    }
  }
  trait MaxRoiPool extends Operator {

    def MaxRoiPool1[@sp T: Numeric: ClassTag](
        name: String,
        pooled_shape: Option[(Array[Int])],
        spatial_scaleAttr: Option[(Float)] = None,
        X: Option[Tensor[T]],
        rois: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("pooled_shape" -> pooled_shape, "spatial_scaleAttr" -> spatial_scaleAttr)
      val allInputs = (
        X,
        rois,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "MaxRoiPool", allInputs, map))
    }
  }
  trait MaxUnpool extends Operator {

    def MaxUnpool9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T1]],
        I: Option[Tensor[T2]],
        output_shapeInput: Option[Tensor[T2]] = None
    )(implicit
        evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Long]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] =
        Map("kernel_shape" -> kernel_shape, "pads" -> pads, "strides" -> strides)
      val allInputs = (
        X,
        I,
        output_shapeInput,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[T2]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T1],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "MaxUnpool",
          allInputs,
          map
        )
      )
    }

    def MaxUnpool11[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T1]],
        I: Option[Tensor[T2]],
        output_shapeInput: Option[Tensor[T2]] = None
    )(implicit
        evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Long]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] =
        Map("kernel_shape" -> kernel_shape, "pads" -> pads, "strides" -> strides)
      val allInputs = (
        X,
        I,
        output_shapeInput,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[T2]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T1],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "MaxUnpool",
          allInputs,
          map
        )
      )
    }
  }
  trait Mean extends Operator {

    def Mean1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        data_0: Seq[Option[Tensor[T]]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Mean",
          allInputs,
          map
        )
      )
    }

    def Mean6[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Mean",
          allInputs,
          map
        )
      )
    }

    def Mean8[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Mean",
          allInputs,
          map
        )
      )
    }
  }
  trait MeanVarianceNormalization extends Operator {

    def MeanVarianceNormalization9[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "MeanVarianceNormalization", allInputs, map))
    }
  }
  trait Min extends Operator {

    def Min1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        data_0: Seq[Option[Tensor[T]]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Min",
          allInputs,
          map
        )
      )
    }

    def Min6[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Min",
          allInputs,
          map
        )
      )
    }

    def Min8[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Min",
          allInputs,
          map
        )
      )
    }
  }
  trait Mod extends Operator {

    def Mod10[@sp T: Numeric: ClassTag](
        name: String,
        fmod: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("fmod" -> fmod)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Mod", allInputs, map))
    }
  }
  trait Mul extends Operator {

    def Mul1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "broadcast" -> broadcast, "consumed_inputs" -> consumed_inputs)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Mul", allInputs, map))
    }

    def Mul6[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Mul", allInputs, map))
    }

    def Mul7[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Mul", allInputs, map))
    }
  }
  trait Multinomial extends Operator {

    def Multinomial7[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        sample_size: Option[(Int)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "sample_size" -> sample_size, "seed" -> seed)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Multinomial", allInputs, map))
    }
  }
  trait Neg extends Operator {

    def Neg1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[Int]#or[Byte]#or[Short]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Neg", allInputs, map))
    }

    def Neg6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[Int]#or[Byte]#or[Short]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Neg", allInputs, map))
    }
  }
  trait NonMaxSuppression extends Operator {

    def NonMaxSuppression10(
        name: String,
        center_point_box: Option[(Int)] = None,
        boxes: Option[Tensor[Float]],
        scores: Option[Tensor[Float]],
        max_output_boxes_per_class: Option[Tensor[Long]] = None,
        iou_threshold: Option[Tensor[Float]] = None,
        score_threshold: Option[Tensor[Float]] = None
    ): (Tensor[Long]) = {
      val map: Map[String, Any] = Map("center_point_box" -> center_point_box)
      val allInputs = (
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[Float]],
          Option[Tensor[Float]],
          Option[Tensor[Long]],
          Option[Tensor[Float]],
          Option[Tensor[Float]],
          Any,
          Any,
          Any,
          Any,
          Tensor[Long],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "NonMaxSuppression",
          allInputs,
          map
        )
      )
    }

    def NonMaxSuppression11(
        name: String,
        center_point_box: Option[(Int)] = None,
        boxes: Option[Tensor[Float]],
        scores: Option[Tensor[Float]],
        max_output_boxes_per_class: Option[Tensor[Long]] = None,
        iou_threshold: Option[Tensor[Float]] = None,
        score_threshold: Option[Tensor[Float]] = None
    ): (Tensor[Long]) = {
      val map: Map[String, Any] = Map("center_point_box" -> center_point_box)
      val allInputs = (
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[Float]],
          Option[Tensor[Float]],
          Option[Tensor[Long]],
          Option[Tensor[Float]],
          Option[Tensor[Float]],
          Any,
          Any,
          Any,
          Any,
          Tensor[Long],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "NonMaxSuppression",
          allInputs,
          map
        )
      )
    }
  }
  trait NonZero extends Operator {

    def NonZero9[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[Long]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Long],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "NonZero", allInputs, map))
    }
  }
  trait Normalizer extends Operator {

    def Normalizer1[@sp T: Numeric: ClassTag](
        name: String,
        norm: Option[(String)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): (Tensor[Float]) = {
      val map: Map[String, Any] = Map("norm" -> norm)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Float],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Normalizer", allInputs, map))
    }
  }
  trait Not extends Operator {

    def Not1[@sp T: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Not", allInputs, map))
    }
  }
  trait OneHot extends Operator {

    def OneHot9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag, @sp T3: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        indices: Option[Tensor[T1]],
        depth: Option[Tensor[T2]],
        values: Option[Tensor[T3]]
    )(implicit
        evT1: Contains[
          T1,
          Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
            Int
          ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ],
        evT2: Contains[T2, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT3: Contains[T3, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[T3]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        indices,
        depth,
        values,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[T3]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T3],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "OneHot",
          allInputs,
          map
        )
      )
    }

    def OneHot11[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag, @sp T3: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        indices: Option[Tensor[T1]],
        depth: Option[Tensor[T2]],
        values: Option[Tensor[T3]]
    )(implicit
        evT1: Contains[
          T1,
          Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
            Int
          ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ],
        evT2: Contains[T2, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT3: Contains[T3, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[T3]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        indices,
        depth,
        values,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[T3]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T3],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "OneHot",
          allInputs,
          map
        )
      )
    }
  }
  trait OneHotEncoder extends Operator {

    def OneHotEncoder1[@sp T: Numeric: ClassTag](
        name: String,
        cats_int64s: Option[(Array[Int])] = None,
        cats_strings: Option[(Array[String])] = None,
        zeros: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[String]#or[Long]#or[Int]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[Float]) = {
      val map: Map[String, Any] =
        Map("cats_int64s" -> cats_int64s, "cats_strings" -> cats_strings, "zeros" -> zeros)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Float],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "OneHotEncoder", allInputs, map))
    }
  }
  trait Or extends Operator {

    def Or1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Or", allInputs, map))
    }

    def Or7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Or", allInputs, map))
    }
  }
  trait PRelu extends Operator {

    def PRelu1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        slope,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "PRelu", allInputs, map))
    }

    def PRelu6[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        slope,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "PRelu", allInputs, map))
    }

    def PRelu7[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        slope,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "PRelu", allInputs, map))
    }

    def PRelu9[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        slope,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "PRelu", allInputs, map))
    }
  }
  trait Pad extends Operator {

    def Pad1[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        paddings: Option[(Array[Int])],
        value: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("mode" -> mode, "paddings" -> paddings, "value" -> value)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Pad", allInputs, map))
    }

    def Pad2[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        pads: Option[(Array[Int])],
        value: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("mode" -> mode, "pads" -> pads, "value" -> value)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Pad", allInputs, map))
    }

    def Pad11[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        data: Option[Tensor[T]],
        pads: Option[Tensor[Long]],
        constant_value: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs = (
        data,
        pads,
        constant_value,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[Long]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Pad",
          allInputs,
          map
        )
      )
    }
  }
  trait Pow extends Operator {

    def Pow1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        X: Option[Tensor[T]],
        Y: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        X,
        Y,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Pow", allInputs, map))
    }

    def Pow7[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]], Y: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        Y,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Pow", allInputs, map))
    }
  }
  trait QLinearConv extends Operator {

    def QLinearConv10[
        @sp T1: Numeric: ClassTag,
        @sp T2: Numeric: ClassTag,
        @sp T3: Numeric: ClassTag,
        @sp T4: Numeric: ClassTag
    ](
        name: String,
        auto_pad: Option[(String)] = None,
        dilations: Option[(Array[Int])] = None,
        group: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        x: Option[Tensor[T1]],
        x_scale: Option[Tensor[Float]],
        x_zero_point: Option[Tensor[T1]],
        w: Option[Tensor[T2]],
        w_scale: Option[Tensor[Float]],
        w_zero_point: Option[Tensor[T2]],
        y_scale: Option[Tensor[Float]],
        y_zero_point: Option[Tensor[T3]],
        B: Option[Tensor[T4]] = None
    )(implicit
        evT1: Contains[T1, Union[Byte]#or[UByte]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create],
        evT3: Contains[T3, Union[Byte]#or[UByte]#or[UNil]#create],
        evT4: Contains[T4, Union[Int]#or[UNil]#create]
    ): (Tensor[T3]) = {
      val map: Map[String, Any] = Map(
        "auto_pad"     -> auto_pad,
        "dilations"    -> dilations,
        "group"        -> group,
        "kernel_shape" -> kernel_shape,
        "pads"         -> pads,
        "strides"      -> strides
      )
      val allInputs = (x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B)
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[Float]],
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[Float]],
          Option[Tensor[T2]],
          Option[Tensor[Float]],
          Option[Tensor[T3]],
          Option[Tensor[T4]],
          Tensor[T3],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "QLinearConv",
          allInputs,
          map
        )
      )
    }
  }
  trait QLinearMatMul extends Operator {

    def QLinearMatMul10[
        @sp T1: Numeric: ClassTag,
        @sp T2: Numeric: ClassTag,
        @sp T3: Numeric: ClassTag
    ](
        name: String,
        a: Option[Tensor[T1]],
        a_scale: Option[Tensor[Float]],
        a_zero_point: Option[Tensor[T1]],
        b: Option[Tensor[T2]],
        b_scale: Option[Tensor[Float]],
        b_zero_point: Option[Tensor[T2]],
        y_scale: Option[Tensor[Float]],
        y_zero_point: Option[Tensor[T3]]
    )(implicit
        evT1: Contains[T1, Union[Byte]#or[UByte]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create],
        evT3: Contains[T3, Union[Byte]#or[UByte]#or[UNil]#create]
    ): (Tensor[T3]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        a,
        a_scale,
        a_zero_point,
        b,
        b_scale,
        b_zero_point,
        y_scale,
        y_zero_point,
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[Float]],
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[Float]],
          Option[Tensor[T2]],
          Option[Tensor[Float]],
          Option[Tensor[T3]],
          Any,
          Tensor[T3],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "QLinearMatMul",
          allInputs,
          map
        )
      )
    }
  }
  trait QuantizeLinear extends Operator {

    def QuantizeLinear10[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        x: Option[Tensor[T1]],
        y_scale: Option[Tensor[Float]],
        y_zero_point: Option[Tensor[T2]] = None
    )(implicit
        evT1: Contains[T1, Union[Float]#or[Int]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        x,
        y_scale,
        y_zero_point,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[Float]],
          Option[Tensor[T2]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T2],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "QuantizeLinear",
          allInputs,
          map
        )
      )
    }
  }
  trait RNN extends Operator {

    def RNN1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        output_sequence: Option[(Int)] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        R: Option[Tensor[T]],
        B: Option[Tensor[T]] = None,
        sequence_lens: Option[Tensor[T1]] = None,
        initial_h: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size,
        "output_sequence"  -> output_sequence
      )
      val allInputs = (
        X,
        W,
        R,
        B,
        sequence_lens,
        initial_h,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T1]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "RNN",
          allInputs,
          map
        )
      )
    }

    def RNN7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        activation_alpha: Option[(Array[Float])] = None,
        activation_beta: Option[(Array[Float])] = None,
        activations: Option[(Array[String])] = None,
        clip: Option[(Float)] = None,
        direction: Option[(String)] = None,
        hidden_size: Option[(Int)] = None,
        X: Option[Tensor[T]],
        W: Option[Tensor[T]],
        R: Option[Tensor[T]],
        B: Option[Tensor[T]] = None,
        sequence_lens: Option[Tensor[T1]] = None,
        initial_h: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "activation_alpha" -> activation_alpha,
        "activation_beta"  -> activation_beta,
        "activations"      -> activations,
        "clip"             -> clip,
        "direction"        -> direction,
        "hidden_size"      -> hidden_size
      )
      val allInputs = (
        X,
        W,
        R,
        B,
        sequence_lens,
        initial_h,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T1]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "RNN",
          allInputs,
          map
        )
      )
    }
  }
  trait RandomNormal extends Operator {

    def RandomNormal1[@sp T: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        mean: Option[(Float)] = None,
        scaleAttr: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        shape: Option[(Array[Int])]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "dtype"     -> dtype,
        "mean"      -> mean,
        "scaleAttr" -> scaleAttr,
        "seed"      -> seed,
        "shape"     -> shape
      )
      val allInputs = (
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "RandomNormal", allInputs, map))
    }
  }
  trait RandomNormalLike extends Operator {

    def RandomNormalLike1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        mean: Option[(Float)] = None,
        scaleAttr: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[
          T1,
          Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
            Int
          ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
            Complex[Double]
          ]#or[UNil]#create
        ],
        evT2: Contains[T2, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "mean" -> mean, "scaleAttr" -> scaleAttr, "seed" -> seed)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "RandomNormalLike", allInputs, map))
    }
  }
  trait RandomUniform extends Operator {

    def RandomUniform1[@sp T: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        high: Option[(Float)] = None,
        low: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        shape: Option[(Array[Int])]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "high" -> high, "low" -> low, "seed" -> seed, "shape" -> shape)
      val allInputs = (
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "RandomUniform", allInputs, map))
    }
  }
  trait RandomUniformLike extends Operator {

    def RandomUniformLike1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        high: Option[(Float)] = None,
        low: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[
          T1,
          Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
            Int
          ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
            Complex[Double]
          ]#or[UNil]#create
        ],
        evT2: Contains[T2, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T2]) = {
      val map: Map[String, Any] =
        Map("dtype" -> dtype, "high" -> high, "low" -> low, "seed" -> seed)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "RandomUniformLike", allInputs, map))
    }
  }
  trait Range extends Operator {

    def Range11[@sp T: Numeric: ClassTag](
        name: String,
        start: Option[Tensor[T]],
        limit: Option[Tensor[T]],
        delta: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float]#or[Double]#or[Short]#or[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        start,
        limit,
        delta,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Range",
          allInputs,
          map
        )
      )
    }
  }
  trait Reciprocal extends Operator {

    def Reciprocal1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Reciprocal", allInputs, map))
    }

    def Reciprocal6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Reciprocal", allInputs, map))
    }
  }
  trait ReduceL1 extends Operator {

    def ReduceL11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceL1", allInputs, map))
    }

    def ReduceL111[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceL1", allInputs, map))
    }
  }
  trait ReduceL2 extends Operator {

    def ReduceL21[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceL2", allInputs, map))
    }

    def ReduceL211[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceL2", allInputs, map))
    }
  }
  trait ReduceLogSum extends Operator {

    def ReduceLogSum1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceLogSum", allInputs, map))
    }

    def ReduceLogSum11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceLogSum", allInputs, map))
    }
  }
  trait ReduceLogSumExp extends Operator {

    def ReduceLogSumExp1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceLogSumExp", allInputs, map))
    }

    def ReduceLogSumExp11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceLogSumExp", allInputs, map))
    }
  }
  trait ReduceMax extends Operator {

    def ReduceMax1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceMax", allInputs, map))
    }

    def ReduceMax11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceMax", allInputs, map))
    }
  }
  trait ReduceMean extends Operator {

    def ReduceMean1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceMean", allInputs, map))
    }

    def ReduceMean11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceMean", allInputs, map))
    }
  }
  trait ReduceMin extends Operator {

    def ReduceMin1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceMin", allInputs, map))
    }

    def ReduceMin11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceMin", allInputs, map))
    }
  }
  trait ReduceProd extends Operator {

    def ReduceProd1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceProd", allInputs, map))
    }

    def ReduceProd11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceProd", allInputs, map))
    }
  }
  trait ReduceSum extends Operator {

    def ReduceSum1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceSum", allInputs, map))
    }

    def ReduceSum11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceSum", allInputs, map))
    }
  }
  trait ReduceSumSquare extends Operator {

    def ReduceSumSquare1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceSumSquare", allInputs, map))
    }

    def ReduceSumSquare11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "keepdims" -> keepdims)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ReduceSumSquare", allInputs, map))
    }
  }
  trait Relu extends Operator {

    def Relu1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Relu", allInputs, map))
    }

    def Relu6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Relu", allInputs, map))
    }
  }
  trait Reshape extends Operator {

    def Reshape1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        shape: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs, "shape" -> shape)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Reshape", allInputs, map))
    }

    def Reshape5[@sp T: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        shapeInput: Option[Tensor[Long]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data,
        shapeInput,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[Long]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Reshape", allInputs, map))
    }
  }
  trait Resize extends Operator {

    def Resize10[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs = (
        X,
        scales,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[Option[Tensor[T]], Option[Tensor[Float]], Any, Any, Any, Any, Any, Any, Any, Tensor[
        T
      ], Any, Any, Any, Any, Any, Any, Any, Any](name, "Resize", allInputs, map))
    }

    def Resize11[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        coordinate_transformation_mode: Option[(String)] = None,
        cubic_coeff_a: Option[(Float)] = None,
        exclude_outside: Option[(Int)] = None,
        extrapolation_value: Option[(Float)] = None,
        mode: Option[(String)] = None,
        nearest_mode: Option[(String)] = None,
        X: Option[Tensor[T1]],
        roi: Option[Tensor[T2]],
        scales: Option[Tensor[Float]],
        sizes: Option[Tensor[Long]] = None
    )(implicit
        evT1: Contains[
          T1,
          Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
            Int
          ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
            Complex[Double]
          ]#or[UNil]#create
        ],
        evT2: Contains[T2, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map(
        "coordinate_transformation_mode" -> coordinate_transformation_mode,
        "cubic_coeff_a"                  -> cubic_coeff_a,
        "exclude_outside"                -> exclude_outside,
        "extrapolation_value"            -> extrapolation_value,
        "mode"                           -> mode,
        "nearest_mode"                   -> nearest_mode
      )
      val allInputs = (
        X,
        roi,
        scales,
        sizes,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Option[Tensor[Float]],
          Option[Tensor[Long]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T1],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Resize",
          allInputs,
          map
        )
      )
    }
  }
  trait ReverseSequence extends Operator {

    def ReverseSequence10[@sp T: Numeric: ClassTag](
        name: String,
        batch_axis: Option[(Int)] = None,
        time_axis: Option[(Int)] = None,
        input: Option[Tensor[T]],
        sequence_lens: Option[Tensor[Long]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("batch_axis" -> batch_axis, "time_axis" -> time_axis)
      val allInputs = (
        input,
        sequence_lens,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[Option[Tensor[T]], Option[Tensor[Long]], Any, Any, Any, Any, Any, Any, Any, Tensor[
          T
        ], Any, Any, Any, Any, Any, Any, Any, Any](
          name,
          "ReverseSequence",
          allInputs,
          map
        )
      )
    }
  }
  trait RoiAlign extends Operator {

    def RoiAlign10[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        output_height: Option[(Int)] = None,
        output_width: Option[(Int)] = None,
        sampling_ratio: Option[(Int)] = None,
        spatial_scaleAttr: Option[(Float)] = None,
        X: Option[Tensor[T1]],
        rois: Option[Tensor[T1]],
        batch_indices: Option[Tensor[T2]]
    )(implicit
        evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Long]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map(
        "mode"              -> mode,
        "output_height"     -> output_height,
        "output_width"      -> output_width,
        "sampling_ratio"    -> sampling_ratio,
        "spatial_scaleAttr" -> spatial_scaleAttr
      )
      val allInputs = (
        X,
        rois,
        batch_indices,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T1]],
          Option[Tensor[T1]],
          Option[Tensor[T2]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T1],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "RoiAlign",
          allInputs,
          map
        )
      )
    }
  }
  trait Round extends Operator {

    def Round11[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Round", allInputs, map))
    }
  }
  trait SVMClassifier extends Operator {

    def SVMClassifier1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
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
        X: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[UNil]#create]
    ): (Tensor[T2]) = {
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
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T2],
        Tensor[Float],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SVMClassifier", allInputs, map))
    }
  }
  trait SVMRegressor extends Operator {

    def SVMRegressor1[@sp T: Numeric: ClassTag](
        name: String,
        coefficients: Option[(Array[Float])] = None,
        kernel_params: Option[(Array[Float])] = None,
        kernel_type: Option[(String)] = None,
        n_supports: Option[(Int)] = None,
        one_class: Option[(Int)] = None,
        post_transform: Option[(String)] = None,
        rho: Option[(Array[Float])] = None,
        support_vectors: Option[(Array[Float])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): (Tensor[Float]) = {
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
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Float],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SVMRegressor", allInputs, map))
    }
  }
  trait Scaler extends Operator {

    def Scaler1[@sp T: Numeric: ClassTag](
        name: String,
        offset: Option[(Array[Float])] = None,
        scaleAttr: Option[(Array[Float])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): (Tensor[Float]) = {
      val map: Map[String, Any] = Map("offset" -> offset, "scaleAttr" -> scaleAttr)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Float],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Scaler", allInputs, map))
    }
  }
  trait Scan extends Operator {

    def Scan8[@sp I: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        directions: Option[(Array[Int])] = None,
        num_scan_inputs: Option[(Int)],
        sequence_lens: Option[Tensor[I]] = None,
        initial_state_and_scan_inputs: Seq[Option[Tensor[V]]]
    )(implicit
        evI: Contains[I, Union[Long]#or[UNil]#create],
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[V]) = {
      val map: Map[String, Any] =
        Map("body" -> body, "directions" -> directions, "num_scan_inputs" -> num_scan_inputs)
      val allInputs = (
        sequence_lens,
        initial_state_and_scan_inputs.lift(0).flatten,
        initial_state_and_scan_inputs.lift(1).flatten,
        initial_state_and_scan_inputs.lift(2).flatten,
        initial_state_and_scan_inputs.lift(3).flatten,
        initial_state_and_scan_inputs.lift(4).flatten,
        initial_state_and_scan_inputs.lift(5).flatten,
        initial_state_and_scan_inputs.lift(6).flatten,
        initial_state_and_scan_inputs.lift(7).flatten
      )
      (
        callOp[
          Option[Tensor[I]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Tensor[V],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Scan",
          allInputs,
          map
        )
      )
    }

    def Scan9[@sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        num_scan_inputs: Option[(Int)],
        scan_input_axes: Option[(Array[Int])] = None,
        scan_input_directions: Option[(Array[Int])] = None,
        scan_output_axes: Option[(Array[Int])] = None,
        scan_output_directions: Option[(Array[Int])] = None,
        initial_state_and_scan_inputs: Seq[Option[Tensor[V]]]
    )(implicit
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[V]) = {
      val map: Map[String, Any] = Map(
        "body"                   -> body,
        "num_scan_inputs"        -> num_scan_inputs,
        "scan_input_axes"        -> scan_input_axes,
        "scan_input_directions"  -> scan_input_directions,
        "scan_output_axes"       -> scan_output_axes,
        "scan_output_directions" -> scan_output_directions
      )
      val allInputs = (
        initial_state_and_scan_inputs.lift(0).flatten,
        initial_state_and_scan_inputs.lift(1).flatten,
        initial_state_and_scan_inputs.lift(2).flatten,
        initial_state_and_scan_inputs.lift(3).flatten,
        initial_state_and_scan_inputs.lift(4).flatten,
        initial_state_and_scan_inputs.lift(5).flatten,
        initial_state_and_scan_inputs.lift(6).flatten,
        initial_state_and_scan_inputs.lift(7).flatten,
        initial_state_and_scan_inputs.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Tensor[V],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Scan",
          allInputs,
          map
        )
      )
    }

    def Scan11[@sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        num_scan_inputs: Option[(Int)],
        scan_input_axes: Option[(Array[Int])] = None,
        scan_input_directions: Option[(Array[Int])] = None,
        scan_output_axes: Option[(Array[Int])] = None,
        scan_output_directions: Option[(Array[Int])] = None,
        initial_state_and_scan_inputs: Seq[Option[Tensor[V]]]
    )(implicit
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[V]) = {
      val map: Map[String, Any] = Map(
        "body"                   -> body,
        "num_scan_inputs"        -> num_scan_inputs,
        "scan_input_axes"        -> scan_input_axes,
        "scan_input_directions"  -> scan_input_directions,
        "scan_output_axes"       -> scan_output_axes,
        "scan_output_directions" -> scan_output_directions
      )
      val allInputs = (
        initial_state_and_scan_inputs.lift(0).flatten,
        initial_state_and_scan_inputs.lift(1).flatten,
        initial_state_and_scan_inputs.lift(2).flatten,
        initial_state_and_scan_inputs.lift(3).flatten,
        initial_state_and_scan_inputs.lift(4).flatten,
        initial_state_and_scan_inputs.lift(5).flatten,
        initial_state_and_scan_inputs.lift(6).flatten,
        initial_state_and_scan_inputs.lift(7).flatten,
        initial_state_and_scan_inputs.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Option[Tensor[V]],
          Tensor[V],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Scan",
          allInputs,
          map
        )
      )
    }
  }
  trait Scatter extends Operator {

    def Scatter9[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]],
        updates: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        data,
        indices,
        updates,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[Tind]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Scatter",
          allInputs,
          map
        )
      )
    }

    def Scatter11[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]],
        updates: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        data,
        indices,
        updates,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[Tind]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Scatter",
          allInputs,
          map
        )
      )
    }
  }
  trait ScatterElements extends Operator {

    def ScatterElements11[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]],
        updates: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        data,
        indices,
        updates,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[Tind]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "ScatterElements",
          allInputs,
          map
        )
      )
    }
  }
  trait ScatterND extends Operator {

    def ScatterND11[@sp T: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Long]],
        updates: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data,
        indices,
        updates,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[Long]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "ScatterND",
          allInputs,
          map
        )
      )
    }
  }
  trait Selu extends Operator {

    def Selu1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        gamma: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("alpha" -> alpha, "consumed_inputs" -> consumed_inputs, "gamma" -> gamma)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Selu", allInputs, map))
    }

    def Selu6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        gamma: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("alpha" -> alpha, "gamma" -> gamma)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Selu", allInputs, map))
    }
  }
  trait SequenceAt extends Operator {

    def SequenceAt11[@sp S: Numeric: ClassTag, @sp I: Numeric: ClassTag, @sp T: Numeric: ClassTag](
        name: String,
        input_sequence: Option[S],
        position: Option[Tensor[I]]
    )(implicit
        evS: Contains[
          S,
          Union[Seq[Tensor[UByte]]]#or[Seq[Tensor[UShort]]]#or[Seq[Tensor[UInt]]]#or[Seq[Tensor[
            ULong
          ]]]#or[Seq[Tensor[Byte]]]#or[Seq[Tensor[Short]]]#or[Seq[Tensor[Int]]]#or[Seq[
            Tensor[Long]
          ]]#or[Seq[Tensor[Float16]]]#or[Seq[Tensor[Float]]]#or[Seq[Tensor[Double]]]#or[Seq[
            Tensor[String]
          ]]#or[Seq[Tensor[Boolean]]]#or[Seq[Tensor[Complex[Float]]]]#or[Seq[
            Tensor[Complex[Double]]
          ]]#or[UNil]#create
        ],
        evI: Contains[I, Union[Int]#or[Long]#or[UNil]#create],
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input_sequence,
        position,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[S],
        Option[Tensor[I]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SequenceAt", allInputs, map))
    }
  }
  trait SequenceConstruct extends Operator {

    def SequenceConstruct11[@sp T: Numeric: ClassTag, @sp S: Numeric: ClassTag](
        name: String,
        inputs: Seq[Option[Tensor[T]]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evS: Contains[
          S,
          Union[Seq[Tensor[UByte]]]#or[Seq[Tensor[UShort]]]#or[Seq[Tensor[UInt]]]#or[Seq[Tensor[
            ULong
          ]]]#or[Seq[Tensor[Byte]]]#or[Seq[Tensor[Short]]]#or[Seq[Tensor[Int]]]#or[Seq[
            Tensor[Long]
          ]]#or[Seq[Tensor[Float16]]]#or[Seq[Tensor[Float]]]#or[Seq[Tensor[Double]]]#or[Seq[
            Tensor[String]
          ]]#or[Seq[Tensor[Boolean]]]#or[Seq[Tensor[Complex[Float]]]]#or[Seq[
            Tensor[Complex[Double]]
          ]]#or[UNil]#create
        ]
    ): (S) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        inputs.lift(0).flatten,
        inputs.lift(1).flatten,
        inputs.lift(2).flatten,
        inputs.lift(3).flatten,
        inputs.lift(4).flatten,
        inputs.lift(5).flatten,
        inputs.lift(6).flatten,
        inputs.lift(7).flatten,
        inputs.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          S,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "SequenceConstruct",
          allInputs,
          map
        )
      )
    }
  }
  trait SequenceEmpty extends Operator {

    def SequenceEmpty11[@sp S: Numeric: ClassTag](name: String, dtype: Option[(Int)] = None)(
        implicit
        evS: Contains[
          S,
          Union[Seq[Tensor[UByte]]]#or[Seq[Tensor[UShort]]]#or[Seq[Tensor[UInt]]]#or[Seq[Tensor[
            ULong
          ]]]#or[Seq[Tensor[Byte]]]#or[Seq[Tensor[Short]]]#or[Seq[Tensor[Int]]]#or[Seq[
            Tensor[Long]
          ]]#or[Seq[Tensor[Float16]]]#or[Seq[Tensor[Float]]]#or[Seq[Tensor[Double]]]#or[Seq[
            Tensor[String]
          ]]#or[Seq[Tensor[Boolean]]]#or[Seq[Tensor[Complex[Float]]]]#or[Seq[
            Tensor[Complex[Double]]
          ]]#or[UNil]#create
        ]
    ): (S) = {
      val map: Map[String, Any] = Map("dtype" -> dtype)
      val allInputs = (
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        S,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SequenceEmpty", allInputs, map))
    }
  }
  trait SequenceErase extends Operator {

    def SequenceErase11[@sp S: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        input_sequence: Option[S],
        position: Option[Tensor[I]] = None
    )(implicit
        evS: Contains[
          S,
          Union[Seq[Tensor[UByte]]]#or[Seq[Tensor[UShort]]]#or[Seq[Tensor[UInt]]]#or[Seq[Tensor[
            ULong
          ]]]#or[Seq[Tensor[Byte]]]#or[Seq[Tensor[Short]]]#or[Seq[Tensor[Int]]]#or[Seq[
            Tensor[Long]
          ]]#or[Seq[Tensor[Float16]]]#or[Seq[Tensor[Float]]]#or[Seq[Tensor[Double]]]#or[Seq[
            Tensor[String]
          ]]#or[Seq[Tensor[Boolean]]]#or[Seq[Tensor[Complex[Float]]]]#or[Seq[
            Tensor[Complex[Double]]
          ]]#or[UNil]#create
        ],
        evI: Contains[I, Union[Int]#or[Long]#or[UNil]#create]
    ): (S) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input_sequence,
        position,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[S],
        Option[Tensor[I]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        S,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SequenceErase", allInputs, map))
    }
  }
  trait SequenceInsert extends Operator {

    def SequenceInsert11[
        @sp S: Numeric: ClassTag,
        @sp T: Numeric: ClassTag,
        @sp I: Numeric: ClassTag
    ](
        name: String,
        input_sequence: Option[S],
        tensor: Option[Tensor[T]],
        position: Option[Tensor[I]] = None
    )(implicit
        evS: Contains[
          S,
          Union[Seq[Tensor[UByte]]]#or[Seq[Tensor[UShort]]]#or[Seq[Tensor[UInt]]]#or[Seq[Tensor[
            ULong
          ]]]#or[Seq[Tensor[Byte]]]#or[Seq[Tensor[Short]]]#or[Seq[Tensor[Int]]]#or[Seq[
            Tensor[Long]
          ]]#or[Seq[Tensor[Float16]]]#or[Seq[Tensor[Float]]]#or[Seq[Tensor[Double]]]#or[Seq[
            Tensor[String]
          ]]#or[Seq[Tensor[Boolean]]]#or[Seq[Tensor[Complex[Float]]]]#or[Seq[
            Tensor[Complex[Double]]
          ]]#or[UNil]#create
        ],
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create],
        evI: Contains[I, Union[Int]#or[Long]#or[UNil]#create]
    ): (S) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input_sequence,
        tensor,
        position,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[S],
        Option[Tensor[T]],
        Option[Tensor[I]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        S,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SequenceInsert", allInputs, map))
    }
  }
  trait SequenceLength extends Operator {

    def SequenceLength11[@sp S: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        input_sequence: Option[S]
    )(implicit
        evS: Contains[
          S,
          Union[Seq[Tensor[UByte]]]#or[Seq[Tensor[UShort]]]#or[Seq[Tensor[UInt]]]#or[Seq[Tensor[
            ULong
          ]]]#or[Seq[Tensor[Byte]]]#or[Seq[Tensor[Short]]]#or[Seq[Tensor[Int]]]#or[Seq[
            Tensor[Long]
          ]]#or[Seq[Tensor[Float16]]]#or[Seq[Tensor[Float]]]#or[Seq[Tensor[Double]]]#or[Seq[
            Tensor[String]
          ]]#or[Seq[Tensor[Boolean]]]#or[Seq[Tensor[Complex[Float]]]]#or[Seq[
            Tensor[Complex[Double]]
          ]]#or[UNil]#create
        ],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): (Tensor[I]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input_sequence,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[S],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[I],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SequenceLength", allInputs, map))
    }
  }
  trait Shape extends Operator {

    def Shape1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evT1: Contains[T1, Union[Long]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Shape", allInputs, map))
    }
  }
  trait Shrink extends Operator {

    def Shrink9[@sp T: Numeric: ClassTag](
        name: String,
        bias: Option[(Float)] = None,
        lambd: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("bias" -> bias, "lambd" -> lambd)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Shrink", allInputs, map))
    }
  }
  trait Sigmoid extends Operator {

    def Sigmoid1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sigmoid", allInputs, map))
    }

    def Sigmoid6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sigmoid", allInputs, map))
    }
  }
  trait Sign extends Operator {

    def Sign9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sign", allInputs, map))
    }
  }
  trait Sin extends Operator {

    def Sin7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sin", allInputs, map))
    }
  }
  trait Sinh extends Operator {

    def Sinh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sinh", allInputs, map))
    }
  }
  trait Size extends Operator {

    def Size1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evT1: Contains[T1, Union[Long]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Size", allInputs, map))
    }
  }
  trait Slice extends Operator {

    def Slice1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        ends: Option[(Array[Int])],
        starts: Option[(Array[Int])],
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes, "ends" -> ends, "starts" -> starts)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Slice", allInputs, map))
    }

    def Slice10[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        starts: Option[Tensor[Tind]],
        ends: Option[Tensor[Tind]],
        axes: Option[Tensor[Tind]] = None,
        steps: Option[Tensor[Tind]] = None
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data,
        starts,
        ends,
        axes,
        steps,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[Tind]],
          Option[Tensor[Tind]],
          Option[Tensor[Tind]],
          Option[Tensor[Tind]],
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Slice",
          allInputs,
          map
        )
      )
    }

    def Slice11[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        starts: Option[Tensor[Tind]],
        ends: Option[Tensor[Tind]],
        axes: Option[Tensor[Tind]] = None,
        steps: Option[Tensor[Tind]] = None
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data,
        starts,
        ends,
        axes,
        steps,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[Tind]],
          Option[Tensor[Tind]],
          Option[Tensor[Tind]],
          Option[Tensor[Tind]],
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Slice",
          allInputs,
          map
        )
      )
    }
  }
  trait Softmax extends Operator {

    def Softmax1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Softmax", allInputs, map))
    }

    def Softmax11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Softmax", allInputs, map))
    }
  }
  trait Softplus extends Operator {

    def Softplus1[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Softplus", allInputs, map))
    }
  }
  trait Softsign extends Operator {

    def Softsign1[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Softsign", allInputs, map))
    }
  }
  trait SpaceToDepth extends Operator {

    def SpaceToDepth1[@sp T: Numeric: ClassTag](
        name: String,
        blocksize: Option[(Int)],
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("blocksize" -> blocksize)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SpaceToDepth", allInputs, map))
    }
  }
  trait Split extends Operator {

    def Split1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Option[Tensor[T]],
        split: Option[Tensor[T]] = None
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "splitAttr" -> splitAttr)
      val allInputs = (
        input,
        split,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Split", allInputs, map))
    }

    def Split2[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "splitAttr" -> splitAttr)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Split", allInputs, map))
    }

    def Split11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "splitAttr" -> splitAttr)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Split", allInputs, map))
    }
  }
  trait SplitToSequence extends Operator {

    def SplitToSequence11[
        @sp T: Numeric: ClassTag,
        @sp I: Numeric: ClassTag,
        @sp S: Numeric: ClassTag
    ](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        input: Option[Tensor[T]],
        split: Option[Tensor[I]] = None
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evI: Contains[I, Union[Int]#or[Long]#or[UNil]#create],
        evS: Contains[
          S,
          Union[Seq[Tensor[UByte]]]#or[Seq[Tensor[UShort]]]#or[Seq[Tensor[UInt]]]#or[Seq[Tensor[
            ULong
          ]]]#or[Seq[Tensor[Byte]]]#or[Seq[Tensor[Short]]]#or[Seq[Tensor[Int]]]#or[Seq[
            Tensor[Long]
          ]]#or[Seq[Tensor[Float16]]]#or[Seq[Tensor[Float]]]#or[Seq[Tensor[Double]]]#or[Seq[
            Tensor[String]
          ]]#or[Seq[Tensor[Boolean]]]#or[Seq[Tensor[Complex[Float]]]]#or[Seq[
            Tensor[Complex[Double]]
          ]]#or[UNil]#create
        ]
    ): (S) = {
      val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
      val allInputs = (
        input,
        split,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[I]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        S,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "SplitToSequence", allInputs, map))
    }
  }
  trait Sqrt extends Operator {

    def Sqrt1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sqrt", allInputs, map))
    }

//TESTING typesafe version
    def Sqrt6[@sp T: Numeric: ClassTag, A <: Axes](name: String, X: Option[TypesafeTensor[T, A]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (TypesafeTensor[T, A]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[Option[TypesafeTensor[T, A]], Any, Any, Any, Any, Any, Any, Any, Any, TypesafeTensor[
        T,
        A
      ], Any, Any, Any, Any, Any, Any, Any, Any](name, "Sqrt", allInputs, map))
    }
  }
  trait Squeeze extends Operator {

    def Squeeze1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Squeeze", allInputs, map))
    }

    def Squeeze11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Squeeze", allInputs, map))
    }
  }
  trait StringNormalizer extends Operator {

    def StringNormalizer10(
        name: String,
        case_change_action: Option[(String)] = None,
        is_case_sensitive: Option[(Int)] = None,
        locale: Option[(String)] = None,
        stopwords: Option[(Array[String])] = None,
        X: Option[Tensor[String]]
    ): (Tensor[String]) = {
      val map: Map[String, Any] = Map(
        "case_change_action" -> case_change_action,
        "is_case_sensitive"  -> is_case_sensitive,
        "locale"             -> locale,
        "stopwords"          -> stopwords
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[String]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[String],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "StringNormalizer", allInputs, map))
    }
  }
  trait Sub extends Operator {

    def Sub1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] =
        Map("axis" -> axis, "broadcast" -> broadcast, "consumed_inputs" -> consumed_inputs)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sub", allInputs, map))
    }

    def Sub6[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sub", allInputs, map))
    }

    def Sub7[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Sub", allInputs, map))
    }
  }
  trait Sum extends Operator {

    def Sum1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        data_0: Seq[Option[Tensor[T]]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Sum",
          allInputs,
          map
        )
      )
    }

    def Sum6[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Sum",
          allInputs,
          map
        )
      )
    }

    def Sum8[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        data_0.lift(0).flatten,
        data_0.lift(1).flatten,
        data_0.lift(2).flatten,
        data_0.lift(3).flatten,
        data_0.lift(4).flatten,
        data_0.lift(5).flatten,
        data_0.lift(6).flatten,
        data_0.lift(7).flatten,
        data_0.lift(8).flatten
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Sum",
          allInputs,
          map
        )
      )
    }
  }
  trait Tan extends Operator {

    def Tan7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Tan", allInputs, map))
    }
  }
  trait Tanh extends Operator {

    def Tanh1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs)
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Tanh", allInputs, map))
    }

    def Tanh6[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Tanh", allInputs, map))
    }
  }
  trait TfIdfVectorizer extends Operator {

    def TfIdfVectorizer9[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        max_gram_length: Option[(Int)],
        max_skip_count: Option[(Int)],
        min_gram_length: Option[(Int)],
        mode: Option[(String)],
        ngram_counts: Option[(Array[Int])],
        ngram_indexes: Option[(Array[Int])],
        pool_int64s: Option[(Array[Int])] = None,
        pool_strings: Option[(Array[String])] = None,
        weights: Option[(Array[Float])] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[String]#or[Int]#or[Long]#or[UNil]#create],
        evT1: Contains[T1, Union[Float]#or[UNil]#create]
    ): (Tensor[T1]) = {
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
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "TfIdfVectorizer", allInputs, map))
    }
  }
  trait ThresholdedRelu extends Operator {

    def ThresholdedRelu10[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("alpha" -> alpha)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ThresholdedRelu", allInputs, map))
    }
  }
  trait Tile extends Operator {

    def Tile1[@sp T: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        tiles: Option[Tensor[T]],
        axis: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        tiles,
        axis,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Tile",
          allInputs,
          map
        )
      )
    }

    def Tile6[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        repeats: Option[Tensor[T1]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ],
        evT1: Contains[T1, Union[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        input,
        repeats,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T1]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Tile", allInputs, map))
    }
  }
  trait TopK extends Operator {

    def TopK1[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        k: Option[(Int)],
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "k" -> k)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[I],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "TopK", allInputs, map))
    }

    def TopK10[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        X: Option[Tensor[T]],
        K: Option[Tensor[Long]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis)
      val allInputs = (
        X,
        K,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[Long]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[I],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "TopK", allInputs, map))
    }

    def TopK11[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        largest: Option[(Int)] = None,
        sorted: Option[(Int)] = None,
        X: Option[Tensor[T]],
        K: Option[Tensor[Long]]
    )(implicit
        evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "largest" -> largest, "sorted" -> sorted)
      val allInputs = (
        X,
        K,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[Long]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Tensor[I],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "TopK", allInputs, map))
    }
  }
  trait Transpose extends Operator {

    def Transpose1[@sp T: Numeric: ClassTag](
        name: String,
        perm: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("perm" -> perm)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Transpose", allInputs, map))
    }
  }
  trait TreeEnsembleClassifier extends Operator {

    def TreeEnsembleClassifier1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
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
        X: Option[Tensor[T1]]
    )(implicit
        evT1: Contains[T1, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[UNil]#create]
    ): (Tensor[T2]) = {
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
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[Option[Tensor[T1]], Any, Any, Any, Any, Any, Any, Any, Any, Tensor[T2], Tensor[
          Float
        ], Any, Any, Any, Any, Any, Any, Any](
          name,
          "TreeEnsembleClassifier",
          allInputs,
          map
        )
      )
    }
  }
  trait TreeEnsembleRegressor extends Operator {

    def TreeEnsembleRegressor1[@sp T: Numeric: ClassTag](
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
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): (Tensor[Float]) = {
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
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[Float],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "TreeEnsembleRegressor", allInputs, map))
    }
  }
  trait Unique extends Operator {

    def Unique11[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        sorted: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "sorted" -> sorted)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Tensor[Long],
          Tensor[Long],
          Tensor[Long],
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Unique",
          allInputs,
          map
        )
      )
    }
  }
  trait Unsqueeze extends Operator {

    def Unsqueeze1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])],
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Unsqueeze", allInputs, map))
    }

    def Unsqueeze11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])],
        data: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("axes" -> axes)
      val allInputs = (
        data,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Unsqueeze", allInputs, map))
    }
  }
  trait Upsample extends Operator {

    def Upsample1[@sp T: Numeric: ClassTag](
        name: String,
        height_scaleAttr: Option[(Float)],
        mode: Option[(String)] = None,
        width_scaleAttr: Option[(Float)],
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Boolean]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[
            UInt
          ]#or[ULong]#or[Byte]#or[Short]#or[String]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map(
        "height_scaleAttr" -> height_scaleAttr,
        "mode"             -> mode,
        "width_scaleAttr"  -> width_scaleAttr
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Upsample", allInputs, map))
    }

    def Upsample7[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        scaleAttrs: Option[(Array[Float])],
        X: Option[Tensor[T]]
    )(implicit
        evT: Contains[
          T,
          Union[Boolean]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[
            UInt
          ]#or[ULong]#or[Byte]#or[Short]#or[String]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("mode" -> mode, "scaleAttrs" -> scaleAttrs)
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Upsample", allInputs, map))
    }

    def Upsample9[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(implicit
        evT: Contains[
          T,
          Union[Boolean]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[
            UInt
          ]#or[ULong]#or[Byte]#or[Short]#or[String]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs = (
        X,
        scales,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[Option[Tensor[T]], Option[Tensor[Float]], Any, Any, Any, Any, Any, Any, Any, Tensor[
        T
      ], Any, Any, Any, Any, Any, Any, Any, Any](name, "Upsample", allInputs, map))
    }

    def Upsample10[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(implicit
        evT: Contains[
          T,
          Union[Boolean]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[
            UInt
          ]#or[ULong]#or[Byte]#or[Short]#or[String]#or[Complex[Float]]#or[Complex[Double]]#or[
            UNil
          ]#create
        ]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map("mode" -> mode)
      val allInputs = (
        X,
        scales,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[Option[Tensor[T]], Option[Tensor[Float]], Any, Any, Any, Any, Any, Any, Any, Tensor[
        T
      ], Any, Any, Any, Any, Any, Any, Any, Any](name, "Upsample", allInputs, map))
    }
  }
  trait Where extends Operator {

    def Where9[@sp B: ClassTag, @sp T: Numeric: ClassTag](
        name: String,
        condition: Option[Tensor[B]],
        X: Option[Tensor[T]],
        Y: Option[Tensor[T]]
    )(implicit
        evB: Contains[B, Union[Boolean]#or[UNil]#create],
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): (Tensor[T]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        condition,
        X,
        Y,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (
        callOp[
          Option[Tensor[B]],
          Option[Tensor[T]],
          Option[Tensor[T]],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Tensor[T],
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any,
          Any
        ](
          name,
          "Where",
          allInputs,
          map
        )
      )
    }
  }
  trait Xor extends Operator {

    def Xor1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map("axis" -> axis, "broadcast" -> broadcast)
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Xor", allInputs, map))
    }

    def Xor7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit
        evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): (Tensor[T1]) = {
      val map: Map[String, Any] = Map()
      val allInputs = (
        A,
        B,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[T]],
        Option[Tensor[T]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Tensor[T1],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "Xor", allInputs, map))
    }
  }
  trait ZipMap extends Operator {

    def ZipMap1[@sp T: Numeric: ClassTag](
        name: String,
        classlabels_int64s: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        X: Option[Tensor[Float]]
    )(implicit
        evT: Contains[
          T,
          Union[Seq[Map[String, Float]]]#or[Seq[Map[Long, Float]]]#or[UNil]#create
        ]
    ): (T) = {
      val map: Map[String, Any] = Map(
        "classlabels_int64s"  -> classlabels_int64s,
        "classlabels_strings" -> classlabels_strings
      )
      val allInputs = (
        X,
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any],
        None: Option[Any]
      )
      (callOp[
        Option[Tensor[Float]],
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        T,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any
      ](name, "ZipMap", allInputs, map))
    }
  }
}
