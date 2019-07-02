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
package object onnx {

  //TODO: Use Dim / Shape from scala-infer
  trait Dim

  sealed trait Axes

  sealed trait Scalar                                    extends Axes
  sealed trait Vec[T <: Dim]                             extends Axes
  sealed trait Mat[T <: Dim, U <: Dim]                   extends Axes
  sealed trait Tuple3OfDim[T <: Dim, U <: Dim, V <: Dim] extends Axes

  type TypesafeTensor[T, A <: Axes] = Tuple2[Array[T], Array[Int]]

  type Tensor[T] = TypesafeTensor[T, Axes]

  trait Operator
  trait Graph

  import org.emergentorder.union.UnionType._
  trait DataSource {
    def getParams[T: Numeric: ClassTag](name: String)(
        implicit ev: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): Tensor[T]
  }
  trait Abs extends Operator {

    def Abs1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Abs6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Acos extends Operator {

    def Acos7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Acosh extends Operator {

    def Acosh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Add extends Operator {

    def Add1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Add6[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Add7[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait And extends Operator {

    def And1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

    def And7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

  }
  trait ArgMax extends Operator {

    def ArgMax1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[Long])

  }
  trait ArgMin extends Operator {

    def ArgMin1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[Long])

  }
  trait ArrayFeatureExtractor extends Operator {

    def ArrayFeatureExtractor1[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        Y: Option[Tensor[Long]]
    )(implicit evT: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int TypeOr String)#check[T])
        : (Tensor[T])

  }
  trait Asin extends Operator {

    def Asin7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Asinh extends Operator {

    def Asinh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Atan extends Operator {

    def Atan7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Atanh extends Operator {

    def Atanh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait AveragePool extends Operator {

    def AveragePool1[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def AveragePool7[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def AveragePool10[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        ceil_mode: Option[(Int)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])

    def BatchNormalization9[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        momentum: Option[(Float)] = None,
        X: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]],
        mean: Option[Tensor[T]],
        someVar: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])

  }
  trait Binarizer extends Operator {

    def Binarizer1[@sp T: Numeric: ClassTag](
        name: String,
        threshold: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T]): (Tensor[T])

  }
  trait Cast extends Operator {

    def Cast1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(String)],
        input: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[
          T1
        ],
        evT2: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[
          T2
        ]
    ): (Tensor[T2])

    def Cast6[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(Int)],
        input: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[
          T1
        ],
        evT2: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[
          T2
        ]
    ): (Tensor[T2])

    def Cast9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        to: Option[(Int)],
        input: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[
          T1
        ],
        evT2: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[
          T2
        ]
    ): (Tensor[T2])

  }
  trait CastMap extends Operator {

    def CastMap1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        cast_to: Option[(String)] = None,
        map_form: Option[(String)] = None,
        max_map: Option[(Int)] = None,
        X: Option[T1]
    )(
        implicit evT1: (UNil TypeOr Map[Long, String] TypeOr Map[Long, Float])#check[T1],
        evT2: (UNil TypeOr String TypeOr Float TypeOr Long)#check[T2]
    ): (Tensor[T2])

  }
  trait CategoryMapper extends Operator {

    def CategoryMapper1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        cats_int64s: Option[(Array[Int])] = None,
        cats_strings: Option[(Array[String])] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr String TypeOr Long)#check[T1],
        evT2: (UNil TypeOr String TypeOr Long)#check[T2]
    ): (Tensor[T2])

  }
  trait Ceil extends Operator {

    def Ceil1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Ceil6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Clip extends Operator {

    def Clip1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        max: Option[(Float)] = None,
        min: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Clip6[@sp T: Numeric: ClassTag](
        name: String,
        max: Option[(Float)] = None,
        min: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Compress extends Operator {

    def Compress9[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]],
        condition: Option[Tensor[T1]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T])

  }
  trait Concat extends Operator {

    def Concat4[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)],
        inputs: Seq[Option[Tensor[T]]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait Constant extends Operator {

    def Constant1[@sp T: Numeric: ClassTag](name: String, value: Option[(Tensor[T])])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Constant9[@sp T: Numeric: ClassTag](name: String, value: Option[(Tensor[T])])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait ConstantOfShape extends Operator {

    def ConstantOfShape9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        value: Option[(Tensor[T2])] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr Long)#check[T1],
        evT2: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[
          T2
        ]
    ): (Tensor[T2])

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
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

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
    )(
        implicit evT1: (UNil TypeOr Byte TypeOr UByte)#check[T1],
        evT2: (UNil TypeOr Byte TypeOr UByte)#check[T2],
        evT3: (UNil TypeOr Int)#check[T3]
    ): (Tensor[T3])

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
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Cos extends Operator {

    def Cos7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Cosh extends Operator {

    def Cosh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait DepthToSpace extends Operator {

    def DepthToSpace1[@sp T: Numeric: ClassTag](
        name: String,
        blocksize: Option[(Int)],
        input: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait DequantizeLinear extends Operator {

    def DequantizeLinear10[@sp T: Numeric: ClassTag](
        name: String,
        x: Option[Tensor[T]],
        x_scale: Option[Tensor[Float]],
        x_zero_point: Option[Tensor[T]] = None
    )(implicit evT: (UNil TypeOr Byte TypeOr UByte TypeOr Int)#check[T]): (Tensor[Float])

  }
  trait DictVectorizer extends Operator {

    def DictVectorizer1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        int64_vocabulary: Option[(Array[Int])] = None,
        string_vocabulary: Option[(Array[String])] = None,
        X: Option[T1]
    )(
        implicit evT1: (UNil TypeOr Map[String, Long] TypeOr Map[Long, String] TypeOr Map[
          Long,
          Float
        ] TypeOr Map[Long, Double] TypeOr Map[String, Float] TypeOr Map[String, Double])#check[T1],
        evT2: (UNil TypeOr Long TypeOr Float TypeOr Double TypeOr String)#check[T2]
    ): (Tensor[T2])

  }
  trait Div extends Operator {

    def Div1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Div6[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Div7[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Dropout extends Operator {

    def Dropout1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T], Tensor[T])

    def Dropout6[@sp T: Numeric: ClassTag](
        name: String,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T], Tensor[T])

    def Dropout7[@sp T: Numeric: ClassTag](
        name: String,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T], Tensor[T])

    def Dropout10[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T], Tensor[T1])

  }
  trait Elu extends Operator {

    def Elu1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Elu6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Equal extends Operator {

    def Equal1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

    def Equal7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

  }
  trait Erf extends Operator {

    def Erf9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Exp extends Operator {

    def Exp1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Exp6[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Expand extends Operator {

    def Expand8[@sp T: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        shape: Option[Tensor[Long]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait EyeLike extends Operator {

    def EyeLike9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        k: Option[(Int)] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[
          T1
        ],
        evT2: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[
          T2
        ]
    ): (Tensor[T2])

  }
  trait Flatten extends Operator {

    def Flatten1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Flatten9[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait Floor extends Operator {

    def Floor1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Floor6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evT1: (UNil TypeOr Int)#check[T1]
    ): (Tensor[T], Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evT1: (UNil TypeOr Int)#check[T1]
    ): (Tensor[T], Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evT1: (UNil TypeOr Int)#check[T1]
    ): (Tensor[T], Tensor[T])

  }
  trait Gather extends Operator {

    def Gather1[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T],
        evTind: (UNil TypeOr Int TypeOr Long)#check[Tind]
    ): (Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

    def Gemm7[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

    def Gemm9[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        transA: Option[(Int)] = None,
        transB: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]],
        C: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait GlobalAveragePool extends Operator {

    def GlobalAveragePool1[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait GlobalLpPool extends Operator {

    def GlobalLpPool1[@sp T: Numeric: ClassTag](
        name: String,
        p: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def GlobalLpPool2[@sp T: Numeric: ClassTag](
        name: String,
        p: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait GlobalMaxPool extends Operator {

    def GlobalMaxPool1[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Greater extends Operator {

    def Greater1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

    def Greater7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

    def Greater9[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

  }
  trait HardSigmoid extends Operator {

    def HardSigmoid1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def HardSigmoid6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Hardmax extends Operator {

    def Hardmax1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Identity extends Operator {

    def Identity1[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait If extends Operator {

    def If1[@sp B: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        else_branch: Option[(Graph)],
        then_branch: Option[(Graph)],
        cond: Option[Tensor[B]]
    )(
        implicit evB: (UNil TypeOr Boolean)#check[B],
        evV: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[V]
    ): (Tensor[V])

  }
  trait Imputer extends Operator {

    def Imputer1[@sp T: Numeric: ClassTag](
        name: String,
        imputed_value_floats: Option[(Array[Float])] = None,
        imputed_value_int64s: Option[(Array[Int])] = None,
        replaced_value_float: Option[(Float)] = None,
        replaced_value_int64: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T]): (Tensor[T])

  }
  trait InstanceNormalization extends Operator {

    def InstanceNormalization1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        epsilon: Option[(Float)] = None,
        input: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def InstanceNormalization6[@sp T: Numeric: ClassTag](
        name: String,
        epsilon: Option[(Float)] = None,
        input: Option[Tensor[T]],
        scale: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait IsInf extends Operator {

    def IsInf10[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        detect_negative: Option[(Int)] = None,
        detect_positive: Option[(Int)] = None,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr Float TypeOr Double)#check[T1],
        evT2: (UNil TypeOr Boolean)#check[T2]
    ): (Tensor[T2])

  }
  trait IsNaN extends Operator {

    def IsNaN9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],
        evT2: (UNil TypeOr Boolean)#check[T2]
    ): (Tensor[T2])

  }
  trait LRN extends Operator {

    def LRN1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        beta: Option[(Float)] = None,
        bias: Option[(Float)] = None,
        size: Option[(Int)],
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evT1: (UNil TypeOr Int)#check[T1]
    ): (Tensor[T], Tensor[T], Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evT1: (UNil TypeOr Int)#check[T1]
    ): (Tensor[T], Tensor[T], Tensor[T])

  }
  trait LabelEncoder extends Operator {

    def LabelEncoder1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        classes_strings: Option[(Array[String])] = None,
        default_int64: Option[(Int)] = None,
        default_string: Option[(String)] = None,
        X: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[
          T1
        ],
        evT2: (UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T2]
    ): (Tensor[T2])

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
    )(
        implicit evT1: (UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[
          T1
        ],
        evT2: (UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T2]
    ): (Tensor[T2])

  }
  trait LeakyRelu extends Operator {

    def LeakyRelu1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def LeakyRelu6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Less extends Operator {

    def Less1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

    def Less7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

    def Less9[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

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
    )(
        implicit evT1: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],
        evT2: (UNil TypeOr String TypeOr Long)#check[T2]
    ): (Tensor[T2], Tensor[Float])

  }
  trait LinearRegressor extends Operator {

    def LinearRegressor1[@sp T: Numeric: ClassTag](
        name: String,
        coefficients: Option[(Array[Float])] = None,
        intercepts: Option[(Array[Float])] = None,
        post_transform: Option[(String)] = None,
        targets: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T]
    ): (Tensor[Float])

  }
  trait Log extends Operator {

    def Log1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Log6[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait LogSoftmax extends Operator {

    def LogSoftmax1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Loop extends Operator {

    def Loop1[@sp I: Numeric: ClassTag, @sp B: Numeric: ClassTag, @sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        M: Option[Tensor[I]] = None,
        cond: Option[Tensor[B]] = None,
        v_initial: Seq[Option[Tensor[V]]]
    )(
        implicit evI: (UNil TypeOr Long)#check[I],
        evB: (UNil TypeOr Boolean)#check[B],
        evV: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[V]
    ): (Tensor[V])

  }
  trait LpNormalization extends Operator {

    def LpNormalization1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        p: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

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
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def LpPool2[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        p: Option[(Int)] = None,
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait MatMul extends Operator {

    def MatMul1[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

    def MatMul9[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

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
    )(
        implicit evT1: (UNil TypeOr Byte TypeOr UByte)#check[T1],
        evT2: (UNil TypeOr Byte TypeOr UByte)#check[T2],
        evT3: (UNil TypeOr Int)#check[T3]
    ): (Tensor[T3])

  }
  trait Max extends Operator {

    def Max6[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

    def Max8[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait MaxPool extends Operator {

    def MaxPool1[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def MaxPool8[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evI: (UNil TypeOr Long)#check[I]
    ): (Tensor[T], Tensor[I])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evI: (UNil TypeOr Long)#check[I]
    ): (Tensor[T], Tensor[I])

  }
  trait MaxRoiPool extends Operator {

    def MaxRoiPool1[@sp T: Numeric: ClassTag](
        name: String,
        pooled_shape: Option[(Array[Int])],
        spatial_scaleAttr: Option[(Float)] = None,
        X: Option[Tensor[T]],
        rois: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

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
    )(
        implicit evT1: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],
        evT2: (UNil TypeOr Long)#check[T2]
    ): (Tensor[T1])

  }
  trait Mean extends Operator {

    def Mean6[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

    def Mean8[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait MeanVarianceNormalization extends Operator {

    def MeanVarianceNormalization9[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Min extends Operator {

    def Min6[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

    def Min8[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Mod extends Operator {

    def Mod10[@sp T: Numeric: ClassTag](
        name: String,
        fmod: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Mul extends Operator {

    def Mul1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Mul6[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Mul7[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Multinomial extends Operator {

    def Multinomial7[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        sample_size: Option[(Int)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],
        evT2: (UNil TypeOr Int TypeOr Long)#check[T2]
    ): (Tensor[T2])

  }
  trait Neg extends Operator {

    def Neg1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Neg6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

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
    ): (Tensor[Long])

  }
  trait NonZero extends Operator {

    def NonZero9[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[Long])

  }
  trait Normalizer extends Operator {

    def Normalizer1[@sp T: Numeric: ClassTag](
        name: String,
        norm: Option[(String)] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T]
    ): (Tensor[Float])

  }
  trait Not extends Operator {

    def Not1[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Boolean)#check[T]
    ): (Tensor[T])

  }
  trait OneHot extends Operator {

    def OneHot9[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag, @sp T3: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        indices: Option[Tensor[T1]],
        depth: Option[Tensor[T2]],
        values: Option[Tensor[T3]]
    )(
        implicit evT1: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T1
        ],
        evT2: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T2
        ],
        evT3: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T3]
    ): (Tensor[T3])

  }
  trait OneHotEncoder extends Operator {

    def OneHotEncoder1[@sp T: Numeric: ClassTag](
        name: String,
        cats_int64s: Option[(Array[Int])] = None,
        cats_strings: Option[(Array[String])] = None,
        zeros: Option[(Int)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr String TypeOr Long TypeOr Int TypeOr Float TypeOr Double)#check[T])
        : (Tensor[Float])

  }
  trait Or extends Operator {

    def Or1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

    def Or7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

  }
  trait PRelu extends Operator {

    def PRelu1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

    def PRelu6[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

    def PRelu7[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

    def PRelu9[@sp T: Numeric: ClassTag](
        name: String,
        X: Option[Tensor[T]],
        slope: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Pad extends Operator {

    def Pad1[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        paddings: Option[(Array[Int])],
        value: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Pad2[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        pads: Option[(Array[Int])],
        value: Option[(Float)] = None,
        data: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Pow extends Operator {

    def Pow1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        X: Option[Tensor[T]],
        Y: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Pow7[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]], Y: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

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
    )(
        implicit evT1: (UNil TypeOr Byte TypeOr UByte)#check[T1],
        evT2: (UNil TypeOr Byte TypeOr UByte)#check[T2],
        evT3: (UNil TypeOr Byte TypeOr UByte)#check[T3],
        evT4: (UNil TypeOr Int)#check[T4]
    ): (Tensor[T3])

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
    )(
        implicit evT1: (UNil TypeOr Byte TypeOr UByte)#check[T1],
        evT2: (UNil TypeOr Byte TypeOr UByte)#check[T2],
        evT3: (UNil TypeOr Byte TypeOr UByte)#check[T3]
    ): (Tensor[T3])

  }
  trait QuantizeLinear extends Operator {

    def QuantizeLinear10[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        x: Option[Tensor[T1]],
        y_scale: Option[Tensor[Float]],
        y_zero_point: Option[Tensor[T2]] = None
    )(
        implicit evT1: (UNil TypeOr Float TypeOr Int)#check[T1],
        evT2: (UNil TypeOr Byte TypeOr UByte)#check[T2]
    ): (Tensor[T2])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evT1: (UNil TypeOr Int)#check[T1]
    ): (Tensor[T], Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evT1: (UNil TypeOr Int)#check[T1]
    ): (Tensor[T], Tensor[T])

  }
  trait RandomNormal extends Operator {

    def RandomNormal1[@sp T: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        mean: Option[(Float)] = None,
        scaleAttr: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        shape: Option[(Array[Int])]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait RandomNormalLike extends Operator {

    def RandomNormalLike1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        mean: Option[(Float)] = None,
        scaleAttr: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T1],
        evT2: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2]
    ): (Tensor[T2])

  }
  trait RandomUniform extends Operator {

    def RandomUniform1[@sp T: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        high: Option[(Float)] = None,
        low: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        shape: Option[(Array[Int])]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait RandomUniformLike extends Operator {

    def RandomUniformLike1[@sp T1: Numeric: ClassTag, @sp T2: Numeric: ClassTag](
        name: String,
        dtype: Option[(Int)] = None,
        high: Option[(Float)] = None,
        low: Option[(Float)] = None,
        seed: Option[(Float)] = None,
        input: Option[Tensor[T1]]
    )(
        implicit evT1: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T1],
        evT2: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2]
    ): (Tensor[T2])

  }
  trait Reciprocal extends Operator {

    def Reciprocal1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Reciprocal6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait ReduceL1 extends Operator {

    def ReduceL11[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceL2 extends Operator {

    def ReduceL21[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceLogSum extends Operator {

    def ReduceLogSum1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceLogSumExp extends Operator {

    def ReduceLogSumExp1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceMax extends Operator {

    def ReduceMax1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceMean extends Operator {

    def ReduceMean1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceMin extends Operator {

    def ReduceMin1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceProd extends Operator {

    def ReduceProd1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceSum extends Operator {

    def ReduceSum1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait ReduceSumSquare extends Operator {

    def ReduceSumSquare1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        keepdims: Option[(Int)] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Relu extends Operator {

    def Relu1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Relu6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Reshape extends Operator {

    def Reshape1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        shape: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Reshape5[@sp T: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        shape: Option[Tensor[Long]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait Resize extends Operator {

    def Resize10[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait ReverseSequence extends Operator {

    def ReverseSequence10[@sp T: Numeric: ClassTag](
        name: String,
        batch_axis: Option[(Int)] = None,
        time_axis: Option[(Int)] = None,
        input: Option[Tensor[T]],
        sequence_lens: Option[Tensor[Long]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

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
    )(
        implicit evT1: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],
        evT2: (UNil TypeOr Long)#check[T2]
    ): (Tensor[T1])

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
    )(
        implicit evT1: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],
        evT2: (UNil TypeOr String TypeOr Long)#check[T2]
    ): (Tensor[T2], Tensor[Float])

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
    )(
        implicit evT: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T]
    ): (Tensor[Float])

  }
  trait Scaler extends Operator {

    def Scaler1[@sp T: Numeric: ClassTag](
        name: String,
        offset: Option[(Array[Float])] = None,
        scaleAttr: Option[(Array[Float])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T]
    ): (Tensor[Float])

  }
  trait Scan extends Operator {

    def Scan9[@sp V: Numeric: ClassTag](
        name: String,
        body: Option[(Graph)],
        num_scan_inputs: Option[(Int)],
        scan_input_axes: Option[(Array[Int])] = None,
        scan_input_directions: Option[(Array[Int])] = None,
        scan_output_axes: Option[(Array[Int])] = None,
        scan_output_directions: Option[(Array[Int])] = None,
        initial_state_and_scan_inputs: Seq[Option[Tensor[V]]]
    )(
        implicit evV: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[V]
    ): (Tensor[V])

  }
  trait Scatter extends Operator {

    def Scatter9[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        data: Option[Tensor[T]],
        indices: Option[Tensor[Tind]],
        updates: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T],
        evTind: (UNil TypeOr Int TypeOr Long)#check[Tind]
    ): (Tensor[T])

  }
  trait Selu extends Operator {

    def Selu1[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        gamma: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Selu6[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        gamma: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Shape extends Operator {

    def Shape1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T],
        evT1: (UNil TypeOr Long)#check[T1]
    ): (Tensor[T1])

  }
  trait Shrink extends Operator {

    def Shrink9[@sp T: Numeric: ClassTag](
        name: String,
        bias: Option[(Float)] = None,
        lambd: Option[(Float)] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Sigmoid extends Operator {

    def Sigmoid1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Sigmoid6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Sign extends Operator {

    def Sign9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Sin extends Operator {

    def Sin7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Sinh extends Operator {

    def Sinh9[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Size extends Operator {

    def Size1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T],
        evT1: (UNil TypeOr Long)#check[T1]
    ): (Tensor[T1])

  }
  trait Slice extends Operator {

    def Slice1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        ends: Option[(Array[Int])],
        starts: Option[(Array[Int])],
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Slice10[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
        name: String,
        data: Option[Tensor[T]],
        starts: Option[Tensor[Tind]],
        ends: Option[Tensor[Tind]],
        axes: Option[Tensor[Tind]] = None,
        steps: Option[Tensor[Tind]] = None
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T],
        evTind: (UNil TypeOr Int TypeOr Long)#check[Tind]
    ): (Tensor[T])

  }
  trait Softmax extends Operator {

    def Softmax1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Softplus extends Operator {

    def Softplus1[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Softsign extends Operator {

    def Softsign1[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait SpaceToDepth extends Operator {

    def SpaceToDepth1[@sp T: Numeric: ClassTag](
        name: String,
        blocksize: Option[(Int)],
        input: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait Split extends Operator {

    def Split1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Option[Tensor[T]],
        split: Option[Tensor[T]] = None
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Split2[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        splitAttr: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait Sqrt extends Operator {

    def Sqrt1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Sqrt6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Squeeze extends Operator {

    def Squeeze1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait StringNormalizer extends Operator {

    def StringNormalizer10(
        name: String,
        case_change_action: Option[(String)] = None,
        is_case_sensitive: Option[(Int)] = None,
        locale: Option[(String)] = None,
        stopwords: Option[(Array[String])] = None,
        X: Option[Tensor[String]]
    ): (Tensor[String])

  }
  trait Sub extends Operator {

    def Sub1[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        consumed_inputs: Option[(Array[Int])] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Sub6[@sp T: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

    def Sub7[@sp T: Numeric: ClassTag](name: String, A: Option[Tensor[T]], B: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
          T
        ]
    ): (Tensor[T])

  }
  trait Sum extends Operator {

    def Sum6[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

    def Sum8[@sp T: Numeric: ClassTag](name: String, data_0: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Tan extends Operator {

    def Tan7[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

  }
  trait Tanh extends Operator {

    def Tanh1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        input: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

    def Tanh6[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
    ): (Tensor[T])

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
    )(
        implicit evT: (UNil TypeOr String TypeOr Int TypeOr Long)#check[T],
        evT1: (UNil TypeOr Float)#check[T1]
    ): (Tensor[T1])

  }
  trait ThresholdedRelu extends Operator {

    def ThresholdedRelu10[@sp T: Numeric: ClassTag](
        name: String,
        alpha: Option[(Float)] = None,
        X: Option[Tensor[T]]
    )(implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]): (Tensor[T])

  }
  trait Tile extends Operator {

    def Tile1[@sp T: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        tiles: Option[Tensor[T]],
        axis: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Tile6[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        input: Option[Tensor[T]],
        repeats: Option[Tensor[T1]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T],
        evT1: (UNil TypeOr Long)#check[T1]
    ): (Tensor[T])

  }
  trait TopK extends Operator {

    def TopK1[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        k: Option[(Int)],
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evI: (UNil TypeOr Long)#check[I]
    ): (Tensor[T], Tensor[I])

    def TopK10[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        X: Option[Tensor[T]],
        K: Option[Tensor[Long]]
    )(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evI: (UNil TypeOr Long)#check[I]
    ): (Tensor[T], Tensor[I])

  }
  trait Transpose extends Operator {

    def Transpose1[@sp T: Numeric: ClassTag](
        name: String,
        perm: Option[(Array[Int])] = None,
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

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
    )(
        implicit evT1: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],
        evT2: (UNil TypeOr String TypeOr Long)#check[T2]
    ): (Tensor[T2], Tensor[Float])

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
    )(
        implicit evT: (UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T]
    ): (Tensor[Float])

  }
  trait Unsqueeze extends Operator {

    def Unsqueeze1[@sp T: Numeric: ClassTag](
        name: String,
        axes: Option[(Array[Int])],
        data: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait Upsample extends Operator {

    def Upsample1[@sp T: Numeric: ClassTag](
        name: String,
        height_scaleAttr: Option[(Float)],
        mode: Option[(String)] = None,
        width_scaleAttr: Option[(Float)],
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Upsample7[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        scaleAttrs: Option[(Array[Float])],
        X: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Upsample9[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(
        implicit evT: (UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

    def Upsample10[@sp T: Numeric: ClassTag](
        name: String,
        mode: Option[(String)] = None,
        X: Option[Tensor[T]],
        scales: Option[Tensor[Float]]
    )(
        implicit evT: (UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait Where extends Operator {

    def Where9[@sp B: Numeric: ClassTag, @sp T: Numeric: ClassTag](
        name: String,
        condition: Option[Tensor[B]],
        X: Option[Tensor[T]],
        Y: Option[Tensor[T]]
    )(
        implicit evB: (UNil TypeOr Boolean)#check[B],
        evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float
        ] TypeOr Complex[Double])#check[T]
    ): (Tensor[T])

  }
  trait Xor extends Operator {

    def Xor1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        axis: Option[(Int)] = None,
        broadcast: Option[(Int)] = None,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

    def Xor7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
        name: String,
        A: Option[Tensor[T]],
        B: Option[Tensor[T]]
    )(
        implicit evT: (UNil TypeOr Boolean)#check[T],
        evT1: (UNil TypeOr Boolean)#check[T1]
    ): (Tensor[T1])

  }
  trait ZipMap extends Operator {

    def ZipMap1[@sp T: Numeric: ClassTag](
        name: String,
        classlabels_int64s: Option[(Array[Int])] = None,
        classlabels_strings: Option[(Array[String])] = None,
        X: Option[Tensor[Float]]
    )(
        implicit evT: (UNil TypeOr Seq[Map[String, Float]] TypeOr Seq[Map[Long, Float]])#check[T]
    ): (T)

  }
}
