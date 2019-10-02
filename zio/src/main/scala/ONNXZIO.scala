package org.emergentorder

import zio.Task
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
import onnx._
import org.emergentorder.union._

package object onnxZIO {
  trait DataSourceZIO {
    def getParamsZIO[T: Numeric: ClassTag](name: String): Task[Tensor[T]]
  }
  trait AbsZIO extends Operator {

    def Abs1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Abs6ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait AcosZIO extends Operator {

    def Acos7ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait AcoshZIO extends Operator {

    def Acosh9ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait AddZIO extends Operator {

    def Add1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Add6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Add7ZIO[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait AndZIO extends Operator {

    def And1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

    def And7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait ArgMaxZIO extends Operator {

    def ArgMax1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[Long])]

  }
  trait ArgMinZIO extends Operator {

    def ArgMin1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[Long])]

  }
  trait ArrayFeatureExtractorZIO extends Operator {

    def ArrayFeatureExtractor1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        Y: Option[Tensor[Long]]
    )(
        implicit evT: Contains[
          T,
          Union[Float]#or[Double]#or[Long]#or[Int]#or[String]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait AsinZIO extends Operator {

    def Asin7ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait AsinhZIO extends Operator {

    def Asinh9ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait AtanZIO extends Operator {

    def Atan7ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait AtanhZIO extends Operator {

    def Atanh9ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait AveragePoolZIO extends Operator {

    def AveragePool1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def AveragePool7ZIO[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def AveragePool10ZIO[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait BatchNormalizationZIO extends Operator {

    def BatchNormalization1ZIO[@sp T: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]

    def BatchNormalization6ZIO[@sp T: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]

    def BatchNormalization7ZIO[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        momentum: Option[(Float)] = None,
        spatial: Option[(Int)] = None,
        X: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]],
        mean: Option[Tensor[T]],
        someVar: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]

    def BatchNormalization9ZIO[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        momentum: Option[(Float)] = None,
        X: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]],
        mean: Option[Tensor[T]],
        someVar: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]

  }
  trait BinarizerZIO extends Operator {

    def Binarizer1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        threshold: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait CastMapZIO extends Operator {

    def CastMap1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        cast_to: Option[(String)] = None,
        map_form: Option[(String)] = None,
        max_map: Option[(Int)] = None,
        X: Option[T1]
    )(
        implicit evT1: Contains[T1, Union[Map[Long, String]]#or[Map[Long, Float]]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Float]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait CastZIO extends Operator {

    def Cast1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(String)],
        input: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[
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
    ): Task[(Tensor[T2])]

    def Cast6ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(Int)],
        input: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[
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
    ): Task[(Tensor[T2])]

    def Cast9ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(Int)],
        input: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[
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
    ): Task[(Tensor[T2])]

  }
  trait CategoryMapperZIO extends Operator {

    def CategoryMapper1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        cats_int64s: Option[(Array[Int])] = None,
        cats_strings: Option[(Array[String])] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[T1, Union[String]#or[Long]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait CeilZIO extends Operator {

    def Ceil1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Ceil6ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ClipZIO extends Operator {

    def Clip1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        max: Option[(Float)] = None,
        min: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Clip6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        max: Option[(Float)] = None,
        min: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait CompressZIO extends Operator {

    def Compress9ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]],
        condition: Option[Tensor[T1]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ConcatZIO extends Operator {

    def Concat4ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)],
        inputs: Seq[Option[Tensor[T]]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ConstantOfShapeZIO extends Operator {

    def ConstantOfShape9ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        value: Option[(Tensor[T2])] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[T1, Union[Long]#or[UNil]#create],
        evT2: Contains[
          T2,
          Union[Float16]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[
            UShort
          ]#or[UInt]#or[ULong]#or[Boolean]#or[UNil]#create
        ]
    ): Task[(Tensor[T2])]

  }
  trait ConstantZIO extends Operator {

    def Constant1ZIO[@sp T: Numeric: ClassTag](name: String, value: Option[(Tensor[T])])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Constant9ZIO[@sp T: Numeric: ClassTag](name: String, value: Option[(Tensor[T])])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ConvIntegerZIO extends Operator {

    def ConvInteger10ZIO[
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
    )(
        implicit evT1: Contains[T1, Union[Byte]#or[UByte]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create],
        evT3: Contains[T3, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T3])]

  }
  trait ConvTransposeZIO extends Operator {

    def ConvTranspose1ZIO[@sp T: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ConvZIO extends Operator {

    def Conv1ZIO[@sp T: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait CosZIO extends Operator {

    def Cos7ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait CoshZIO extends Operator {

    def Cosh9ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait DepthToSpaceZIO extends Operator {

    def DepthToSpace1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        blocksize: Option[(Int)],
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait DequantizeLinearZIO extends Operator {

    def DequantizeLinear10ZIO[@sp T: Numeric: ClassTag](
        name: String,
        x: Option[Tensor[T]],
        x_scale: Option[Tensor[Float]],
        x_zero_point: Option[Tensor[T]] = None
    )(
        implicit evT: Contains[T, Union[Byte]#or[UByte]#or[Int]#or[UNil]#create]
    ): Task[(Tensor[Float])]

  }
  trait DictVectorizerZIO extends Operator {

    def DictVectorizer1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        int64_vocabulary: Option[(Array[Int])] = None,
        string_vocabulary: Option[(Array[String])] = None,
        X: Option[T1]
    )(
        implicit evT1: Contains[T1, Union[Map[String, Long]]#or[Map[Long, String]]#or[Map[
          Long,
          Float
        ]]#or[Map[Long, Double]]#or[Map[String, Float]]#or[Map[String, Double]]#or[UNil]#create],
        evT2: Contains[T2, Union[Long]#or[Float]#or[Double]#or[String]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait DivZIO extends Operator {

    def Div1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Div6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Div7ZIO[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait DropoutZIO extends Operator {

    def Dropout1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T])]

    def Dropout6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T])]

    def Dropout7ZIO[@sp T: Numeric: ClassTag](
        name: String,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T])]

    def Dropout10ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T1])]

  }
  trait EluZIO extends Operator {

    def Elu1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Elu6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait EqualZIO extends Operator {

    def Equal1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Boolean]#or[Int]#or[Long]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

    def Equal7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Boolean]#or[Int]#or[Long]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait ErfZIO extends Operator {

    def Erf9ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ExpZIO extends Operator {

    def Exp1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Exp6ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ExpandZIO extends Operator {

    def Expand8ZIO[@sp T: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        shape: Option[Tensor[Long]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait EyeLikeZIO extends Operator {

    def EyeLike9ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        k: Option[(Int)] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[
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
    ): Task[(Tensor[T2])]

  }
  trait FlattenZIO extends Operator {

    def Flatten1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Flatten9ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait FloorZIO extends Operator {

    def Floor1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Floor6ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait GRUZIO extends Operator {

    def GRU1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T])]

    def GRU3ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T])]

    def GRU7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T])]

  }
  trait GatherZIO extends Operator {

    def Gather1ZIO[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait GemmZIO extends Operator {

    def Gemm1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        broadcast: Option[(Int)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Gemm6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        broadcast: Option[(Int)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Gemm7ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Gemm9ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait GlobalAveragePoolZIO extends Operator {

    def GlobalAveragePool1ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait GlobalLpPoolZIO extends Operator {

    def GlobalLpPool1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        p: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def GlobalLpPool2ZIO[@sp T: Numeric: ClassTag](
        name: String,
        p: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait GlobalMaxPoolZIO extends Operator {

    def GlobalMaxPool1ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait GreaterZIO extends Operator {

    def Greater1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

    def Greater7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

    def Greater9ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait HardSigmoidZIO extends Operator {

    def HardSigmoid1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def HardSigmoid6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait HardmaxZIO extends Operator {

    def Hardmax1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait IdentityZIO extends Operator {

    def Identity1ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait IfZIO extends Operator {

    def If1ZIO[@sp B: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        else_branch: Option[(Graph)],
        then_branch: Option[(Graph)],
        cond: Option[Tensor[B]]
    )(
        implicit evB: Contains[B, Union[Boolean]#or[UNil]#create],
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): Task[(Tensor[V])]

  }
  trait ImputerZIO extends Operator {

    def Imputer1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        imputed_value_floats: Option[(Array[Float])] = None,
        imputed_value_int64s: Option[(Array[Int])] = None,
        replaced_value_float: Option[(Float)] = None,
        replaced_value_int64: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait InstanceNormalizationZIO extends Operator {

    def InstanceNormalization1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        epsilon: Option[(Float)] = None,
        input: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def InstanceNormalization6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        input: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait IsInfZIO extends Operator {

    def IsInf10ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        detect_negative: Option[(Int)] = None,
        detect_positive: Option[(Int)] = None,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[T1, Union[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait IsNaNZIO extends Operator {

    def IsNaN9ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait LRNZIO extends Operator {

    def LRN1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        bias: Option[(Float)] = None,
        size: Option[(Int)],
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait LSTMZIO extends Operator {

    def LSTM1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T], Tensor[T])]

    def LSTM7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T], Tensor[T])]

  }
  trait LabelEncoderZIO extends Operator {

    def LabelEncoder1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        classes_strings: Option[(Array[String])] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[T1, Union[String]#or[Long]#or[Float]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[Float]#or[UNil]#create]
    ): Task[(Tensor[T2])]

    def LabelEncoder2ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
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
    )(
        implicit evT1: Contains[T1, Union[String]#or[Long]#or[Float]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[Float]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait LeakyReluZIO extends Operator {

    def LeakyRelu1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def LeakyRelu6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait LessZIO extends Operator {

    def Less1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

    def Less7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

    def Less9ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait LinearClassifierZIO extends Operator {

    def LinearClassifier1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        classlabels_ints: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        coefficients: Option[(Array[Float])],
        intercepts: Option[(Array[Float])] = None,
        multi_class: Option[(Int)] = None,
        post_transform: Option[(String)] = None,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[T1, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T2], Tensor[Float])]

  }
  trait LinearRegressorZIO extends Operator {

    def LinearRegressor1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        coefficients: Option[(Array[Float])] = None,
        intercepts: Option[(Array[Float])] = None,
        post_transform: Option[(String)] = None,
        targets: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): Task[(Tensor[Float])]

  }
  trait LogSoftmaxZIO extends Operator {

    def LogSoftmax1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait LogZIO extends Operator {

    def Log1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Log6ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait LoopZIO extends Operator {

    def Loop1ZIO[@sp I: Numeric: ClassTag, @sp B: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        M: Option[Tensor[I]] = None,
        cond: Option[Tensor[B]] = None,
        v_initial: Seq[Option[Tensor[V]]]
    )(
        implicit evI: Contains[I, Union[Long]#or[UNil]#create],
        evB: Contains[B, Union[Boolean]#or[UNil]#create],
        evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): Task[(Tensor[V])]

  }
  trait LpNormalizationZIO extends Operator {

    def LpNormalization1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        p: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait LpPoolZIO extends Operator {

    def LpPool1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])] = None,
        p: Option[(Float)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def LpPool2ZIO[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        p: Option[(Int)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait MatMulIntegerZIO extends Operator {

    def MatMulInteger10ZIO[
        @sp T1: Numeric: ClassTag,
        @sp T2: Numeric: ClassTag,
        @sp T3: Numeric: ClassTag
    ](
        name: String,
        A: Option[Tensor[T1]],
        B: Option[Tensor[T2]],
        a_zero_point: Option[Tensor[T1]] = None,
        b_zero_point: Option[Tensor[T2]] = None
    )(
        implicit evT1: Contains[T1, Union[Byte]#or[UByte]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create],
        evT3: Contains[T3, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T3])]

  }
  trait MatMulZIO extends Operator {

    def MatMul1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def MatMul9ZIO[@sp T: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait MaxPoolZIO extends Operator {

    def MaxPool1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def MaxPool8ZIO[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[I])]

    def MaxPool10ZIO[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        dilations: Option[(Array[Int])] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[I])]

  }
  trait MaxRoiPoolZIO extends Operator {

    def MaxRoiPool1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        pooled_shape: Option[(Array[Int])],
        spatial_scaleAttr: Option[(Float)] = None,
        X: Option[Tensor[T]],
        rois: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait MaxUnpoolZIO extends Operator {

    def MaxUnpool9ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T1]],
        I: Option[Tensor[T2]],
        output_shapeInput: Option[Tensor[T2]] = None
    )(
        implicit evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait MaxZIO extends Operator {

    def Max6ZIO[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Max8ZIO[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait MeanVarianceNormalizationZIO extends Operator {

    def MeanVarianceNormalization9ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait MeanZIO extends Operator {

    def Mean6ZIO[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Mean8ZIO[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait MinZIO extends Operator {

    def Min6ZIO[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Min8ZIO[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ModZIO extends Operator {

    def Mod10ZIO[@sp T: Numeric: ClassTag](
        name: String,
        fmod: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait MulZIO extends Operator {

    def Mul1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Mul6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Mul7ZIO[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait MultinomialZIO extends Operator {

    def Multinomial7ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        sample_size: Option[(Int)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Int]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait NegZIO extends Operator {

    def Neg1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[Int]#or[Byte]#or[Short]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Neg6ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[Int]#or[Byte]#or[Short]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait NonMaxSuppressionZIO extends Operator {

    def NonMaxSuppression10ZIO(
        name: String,
        center_point_box: Option[(Int)] = None,
        boxes: Option[Tensor[Float]],
        scores: Option[Tensor[Float]],
        max_output_boxes_per_class: Option[Tensor[Long]] = None,
        iou_threshold: Option[Tensor[Float]] = None,
        score_threshold: Option[Tensor[Float]] = None
    ): Task[(Tensor[Long])]

  }
  trait NonZeroZIO extends Operator {

    def NonZero9ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[Long])]

  }
  trait NormalizerZIO extends Operator {

    def Normalizer1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        norm: Option[(String)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): Task[(Tensor[Float])]

  }
  trait NotZIO extends Operator {

    def Not1ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait OneHotEncoderZIO extends Operator {

    def OneHotEncoder1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        cats_int64s: Option[(Array[Int])] = None,
        cats_strings: Option[(Array[String])] = None,
        zeros: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[String]#or[Long]#or[Int]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[Float])]

  }
  trait OneHotZIO extends Operator {

    def OneHot9ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag, @sp T3: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        indices: Option[Tensor[T1]],
        depth: Option[Tensor[T2]],
        values: Option[Tensor[T3]]
    )(
        implicit evT1: Contains[
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
    ): Task[(Tensor[T3])]

  }
  trait OrZIO extends Operator {

    def Or1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

    def Or7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait PReluZIO extends Operator {

    def PRelu1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def PRelu6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def PRelu7ZIO[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def PRelu9ZIO[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait PadZIO extends Operator {

    def Pad1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        paddings: Option[(Array[Int])],
        value: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Pad2ZIO[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        pads: Option[(Array[Int])],
        value: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait PowZIO extends Operator {

    def Pow1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        X: Option[Tensor[T]],
        Y: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Pow7ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]], Y: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait QLinearConvZIO extends Operator {

    def QLinearConv10ZIO[
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
    )(
        implicit evT1: Contains[T1, Union[Byte]#or[UByte]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create],
        evT3: Contains[T3, Union[Byte]#or[UByte]#or[UNil]#create],
        evT4: Contains[T4, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T3])]

  }
  trait QLinearMatMulZIO extends Operator {

    def QLinearMatMul10ZIO[
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
    )(
        implicit evT1: Contains[T1, Union[Byte]#or[UByte]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create],
        evT3: Contains[T3, Union[Byte]#or[UByte]#or[UNil]#create]
    ): Task[(Tensor[T3])]

  }
  trait QuantizeLinearZIO extends Operator {

    def QuantizeLinear10ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        x: Option[Tensor[T1]],
        y_scale: Option[Tensor[Float]],
        y_zero_point: Option[Tensor[T2]] = None
    )(
        implicit evT1: Contains[T1, Union[Float]#or[Int]#or[UNil]#create],
        evT2: Contains[T2, Union[Byte]#or[UByte]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait RNNZIO extends Operator {

    def RNN1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T])]

    def RNN7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT1: Contains[T1, Union[Int]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[T])]

  }
  trait RandomNormalLikeZIO extends Operator {

    def RandomNormalLike1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        mean: Option[(Float)] = None,
        scaleAttr: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[
          T1,
          Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
            Int
          ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
            Complex[Double]
          ]#or[UNil]#create
        ],
        evT2: Contains[T2, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait RandomNormalZIO extends Operator {

    def RandomNormal1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        mean: Option[(Float)] = None,
        scaleAttr: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        shape: Option[(Array[Int])]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait RandomUniformLikeZIO extends Operator {

    def RandomUniformLike1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        high: Option[(Float)] = None,
        low: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: Contains[
          T1,
          Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
            Int
          ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
            Complex[Double]
          ]#or[UNil]#create
        ],
        evT2: Contains[T2, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T2])]

  }
  trait RandomUniformZIO extends Operator {

    def RandomUniform1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        high: Option[(Float)] = None,
        low: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        shape: Option[(Array[Int])]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ReciprocalZIO extends Operator {

    def Reciprocal1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Reciprocal6ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ReduceL1ZIO extends Operator {

    def ReduceL11ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceL2ZIO extends Operator {

    def ReduceL21ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceLogSumExpZIO extends Operator {

    def ReduceLogSumExp1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceLogSumZIO extends Operator {

    def ReduceLogSum1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceMaxZIO extends Operator {

    def ReduceMax1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceMeanZIO extends Operator {

    def ReduceMean1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceMinZIO extends Operator {

    def ReduceMin1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceProdZIO extends Operator {

    def ReduceProd1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceSumSquareZIO extends Operator {

    def ReduceSumSquare1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReduceSumZIO extends Operator {

    def ReduceSum1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[UInt]#or[ULong]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ReluZIO extends Operator {

    def Relu1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Relu6ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ReshapeZIO extends Operator {

    def Reshape1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        shape: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Reshape5ZIO[@sp T: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        shape: Option[Tensor[Long]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait ResizeZIO extends Operator {

    def Resize10ZIO[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ReverseSequenceZIO extends Operator {

    def ReverseSequence10ZIO[@sp T: Numeric: ClassTag](
        name: String,
        batch_axis: Option[(Int)] = None,
        time_axis: Option[(Int)] = None,
        input: Option[Tensor[T]],
        sequence_lens: Option[Tensor[Long]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait RoiAlignZIO extends Operator {

    def RoiAlign10ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        output_height: Option[(Int)] = None,
        output_width: Option[(Int)] = None,
        sampling_ratio: Option[(Int)] = None,
        spatial_scaleAttr: Option[(Float)] = None,
        X: Option[Tensor[T1]],
        rois: Option[Tensor[T1]],
        batch_indices: Option[Tensor[T2]]
    )(
        implicit evT1: Contains[T1, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evT2: Contains[T2, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait SVMClassifierZIO extends Operator {

    def SVMClassifier1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
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
    )(
        implicit evT1: Contains[T1, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T2], Tensor[Float])]

  }
  trait SVMRegressorZIO extends Operator {

    def SVMRegressor1ZIO[@sp T: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): Task[(Tensor[Float])]

  }
  trait ScalerZIO extends Operator {

    def Scaler1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        offset: Option[(Array[Float])] = None,
        scaleAttr: Option[(Array[Float])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): Task[(Tensor[Float])]

  }
  trait ScanZIO extends Operator {

    def Scan9ZIO[@sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        num_scan_inputs: Option[(Int)],
        scan_input_axes: Option[(Array[Int])] = None,
        scan_input_directions: Option[(Array[Int])] = None,
        scan_output_axes: Option[(Array[Int])] = None,
        scan_output_directions: Option[(Array[Int])] = None,
        initial_state_and_scan_inputs: Seq[Option[Tensor[V]]]
    )(
        implicit evV: Contains[V, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[V])]

  }
  trait ScatterZIO extends Operator {

    def Scatter9ZIO[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]],
        updates: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SeluZIO extends Operator {

    def Selu1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        gamma: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Selu6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        gamma: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait ShapeZIO extends Operator {

    def Shape1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evT1: Contains[T1, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait ShrinkZIO extends Operator {

    def Shrink9ZIO[@sp T: Numeric: ClassTag](
        name: String,
        bias: Option[(Float)] = None,
        lambd: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SigmoidZIO extends Operator {

    def Sigmoid1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Sigmoid6ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SignZIO extends Operator {

    def Sign9ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SinZIO extends Operator {

    def Sin7ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SinhZIO extends Operator {

    def Sinh9ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SizeZIO extends Operator {

    def Size1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evT1: Contains[T1, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait SliceZIO extends Operator {

    def Slice1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        ends: Option[(Array[Int])],
        starts: Option[(Array[Int])],
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Slice10ZIO[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        starts: Option[Tensor[Tind]],
        ends: Option[Tensor[Tind]],
        axes: Option[Tensor[Tind]] = None,
        steps: Option[Tensor[Tind]] = None
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create],
        evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SoftmaxZIO extends Operator {

    def Softmax1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SoftplusZIO extends Operator {

    def Softplus1ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SoftsignZIO extends Operator {

    def Softsign1ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SpaceToDepthZIO extends Operator {

    def SpaceToDepth1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        blocksize: Option[(Int)],
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SplitZIO extends Operator {

    def Split1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Option[Tensor[T]],
        split: Option[Tensor[T]] = None
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Split2ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait SqrtZIO extends Operator {

    def Sqrt1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Sqrt6ZIO[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait SqueezeZIO extends Operator {

    def Squeeze1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait StringNormalizerZIO extends Operator {

    def StringNormalizer10ZIO(
        name: String,
        case_change_action: Option[(String)] = None,
        is_case_sensitive: Option[(Int)] = None,
        locale: Option[(String)] = None,
        stopwords: Option[(Array[String])] = None,
        X: Option[Tensor[String]]
    ): Task[(Tensor[String])]

  }
  trait SubZIO extends Operator {

    def Sub1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Sub6ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Sub7ZIO[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait SumZIO extends Operator {

    def Sum6ZIO[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Sum8ZIO[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait TanZIO extends Operator {

    def Tan7ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait TanhZIO extends Operator {

    def Tanh1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

    def Tanh6ZIO[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait TfIdfVectorizerZIO extends Operator {

    def TfIdfVectorizer9ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[String]#or[Int]#or[Long]#or[UNil]#create],
        evT1: Contains[T1, Union[Float]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait ThresholdedReluZIO extends Operator {

    def ThresholdedRelu10ZIO[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait TileZIO extends Operator {

    def Tile1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        tiles: Option[Tensor[T]],
        axis: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Tile6ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        repeats: Option[Tensor[T1]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ],
        evT1: Contains[T1, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait TopKZIO extends Operator {

    def TopK1ZIO[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        k: Option[(Int)],
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[I])]

    def TopK10ZIO[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        X: Option[Tensor[T]],
        K: Option[Tensor[Long]]
    )(
        implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
        evI: Contains[I, Union[Long]#or[UNil]#create]
    ): Task[(Tensor[T], Tensor[I])]

  }
  trait TransposeZIO extends Operator {

    def Transpose1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        perm: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait TreeEnsembleClassifierZIO extends Operator {

    def TreeEnsembleClassifier1ZIO[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
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
    )(
        implicit evT1: Contains[T1, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create],
        evT2: Contains[T2, Union[String]#or[Long]#or[UNil]#create]
    ): Task[(Tensor[T2], Tensor[Float])]

  }
  trait TreeEnsembleRegressorZIO extends Operator {

    def TreeEnsembleRegressor1ZIO[@sp T: Numeric: ClassTag](
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
    )(
        implicit evT: Contains[T, Union[Float]#or[Double]#or[Long]#or[Int]#or[UNil]#create]
    ): Task[(Tensor[Float])]

  }
  trait UnsqueezeZIO extends Operator {

    def Unsqueeze1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])],
        data: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
          Int
        ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
          Complex[Double]
        ]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait UpsampleZIO extends Operator {

    def Upsample1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        height_scaleAttr: Option[(Float)],
        mode: Option[(String)] = None,
        width_scaleAttr: Option[(Float)],
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Boolean]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[
            UInt
          ]#or[ULong]#or[Byte]#or[Short]#or[String]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Upsample7ZIO[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        scaleAttrs: Option[(Array[Float])],
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Boolean]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[
            UInt
          ]#or[ULong]#or[Byte]#or[Short]#or[String]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Upsample9ZIO[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(
        implicit evT: Contains[
          T,
          Union[Boolean]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[
            UInt
          ]#or[ULong]#or[Byte]#or[Short]#or[String]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

    def Upsample10ZIO[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(
        implicit evT: Contains[
          T,
          Union[Boolean]#or[Int]#or[Long]#or[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[
            UInt
          ]#or[ULong]#or[Byte]#or[Short]#or[String]#or[Complex[Float]]#or[Complex[Double]]#or[UNil]#create
        ]
    ): Task[(Tensor[T])]

  }
  trait WhereZIO extends Operator {

    def Where9ZIO[@sp B: Numeric: ClassTag, @sp T: Numeric: ClassTag](
        name: String,
        condition: Option[Tensor[B]],
        X: Option[Tensor[T]],
        Y: Option[Tensor[T]]
    )(
        implicit evB: Contains[B, Union[Boolean]#or[UNil]#create],
        evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[Int]#or[
          Long
        ]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create]
    ): Task[(Tensor[T])]

  }
  trait XorZIO extends Operator {

    def Xor1ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

    def Xor7ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: Contains[T, Union[Boolean]#or[UNil]#create],
        evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
    ): Task[(Tensor[T1])]

  }
  trait ZipMapZIO extends Operator {

    def ZipMap1ZIO[@sp T: Numeric: ClassTag](
        name: String,
        classlabels_int64s: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        X: Option[Tensor[Float]]
    )(
        implicit evT: Contains[
          T,
          Union[Seq[Map[String, Float]]]#or[Seq[Map[Long, Float]]]#or[UNil]#create
        ]
    ): Task[(T)]

  }
}
