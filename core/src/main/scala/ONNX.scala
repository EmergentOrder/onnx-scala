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
package object onnx {

  trait Dim

  sealed trait Axes

  sealed trait Scalar extends Axes
  sealed trait Vec[T <: Dim] extends Axes
  sealed trait Mat[T <: Dim, U <: Dim] extends Axes
  sealed trait Tuple3OfDim[T <: Dim, U <: Dim, V <: Dim] extends Axes

  type TypesafeTensor[T, A <: Axes] = Tuple2[Array[T], Array[Int]]

  type Tensor[T] = TypesafeTensor[T, Axes]
  type SparseTensor[T] = Tensor[T]

  type XInt = Int with Singleton

  object TensorFactory {
    def getTensor[T](data: Array[T], t: Array[Int]): Tensor[T] = {
      require(data.size == t.foldLeft(1)(_ * _))
      (data, t)
    }
   }
  

  
    trait Operator {
    def callOp[T: ClassTag](
        name: String,
        opName: String,
        inputs: Option[NonEmptyTuple],
        //    outName: String,
        attrs: Map[String, Any]
    ): Tuple1[T]
  }

  abstract class Model(onnxBytes: Array[Byte]) extends Operator{
    def fullModel[
      T: ClassTag
    ](
      inputs: Option[NonEmptyTuple]
  ): Tuple1[T]
  }

        trait Graph
trait DataSource {
  def getParams[T : Numeric:ClassTag](name: String): Tensor[T]
}
trait Abs6[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Abs6(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Abs",allInputs, map))
}
}
trait Abs1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Abs1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Abs",allInputs, map))
}
}

trait Acos7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Acos7(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Acos",allInputs, map))
}
}

trait Acosh9[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Acosh9(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Acosh",allInputs, map))
}
}

trait Adagrad1[@sp T1 <: Float | Double : Numeric:ClassTag,@sp T2 <: Long : Numeric:ClassTag,@sp T3 <: Float | Double : Numeric:ClassTag] extends Operator {
  def Adagrad1(name: String,decay_factor : Option[(Float)] = None,epsilon : Option[(Float)] = None,norm_coefficient : Option[(Float)] = None,R: Tensor[T1], T: Tensor[T2],inputs: Seq[Tensor[T3]])
    : Tuple1[Tensor[T3]]
 = {
val map: Map[String, Any] = Map("decay_factor" -> decay_factor 
,"epsilon" -> epsilon 
,"norm_coefficient" -> norm_coefficient 
)
val allInputs = Some(R,T,inputs(0),inputs(1),inputs(2),inputs(3),inputs(4),inputs(5),inputs(6) *: () )
(callOp[Tensor[T3]](name,"Adagrad",allInputs, map))
}
}

trait Add7[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Add7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Add",allInputs, map))
}
}
trait Add6[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Add6(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Add",allInputs, map))
}
}
trait Add1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Add1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
,"consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Add",allInputs, map))
}
}

trait And7[@sp T <: Boolean : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def And7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"And",allInputs, map))
}
}
trait And1[@sp T <: Boolean : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def And1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"And",allInputs, map))
}
}

trait ArgMax12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ArgMax12(name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,select_last_index : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"keepdims" -> keepdims 
,"select_last_index" -> select_last_index 
)
val allInputs = Some(data *: () )
(callOp[Tensor[Long]](name,"ArgMax",allInputs, map))
}
}
trait ArgMax11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ArgMax11(name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[Long]](name,"ArgMax",allInputs, map))
}
}
trait ArgMax1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ArgMax1(name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[Long]](name,"ArgMax",allInputs, map))
}
}

trait ArgMin12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ArgMin12(name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,select_last_index : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"keepdims" -> keepdims 
,"select_last_index" -> select_last_index 
)
val allInputs = Some(data *: () )
(callOp[Tensor[Long]](name,"ArgMin",allInputs, map))
}
}
trait ArgMin11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ArgMin11(name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[Long]](name,"ArgMin",allInputs, map))
}
}
trait ArgMin1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ArgMin1(name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[Long]](name,"ArgMin",allInputs, map))
}
}

trait ArrayFeatureExtractor1[@sp T <: Float | Double | Long | Int | String : Numeric:ClassTag] extends Operator {
  def ArrayFeatureExtractor1(name: String,X: Tensor[T], Y: Tensor[Long])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X,Y *: () )
(callOp[Tensor[T]](name,"ArrayFeatureExtractor",allInputs, map))
}
}

trait Asin7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Asin7(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Asin",allInputs, map))
}
}

trait Asinh9[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Asinh9(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Asinh",allInputs, map))
}
}

trait Atan7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Atan7(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Atan",allInputs, map))
}
}

trait Atanh9[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Atanh9(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Atanh",allInputs, map))
}
}

trait AveragePool11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def AveragePool11(name: String,auto_pad : Option[(String)] = None,ceil_mode : Option[(Int)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"ceil_mode" -> ceil_mode 
,"count_include_pad" -> count_include_pad 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"AveragePool",allInputs, map))
}
}
trait AveragePool10[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def AveragePool10(name: String,auto_pad : Option[(String)] = None,ceil_mode : Option[(Int)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"ceil_mode" -> ceil_mode 
,"count_include_pad" -> count_include_pad 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"AveragePool",allInputs, map))
}
}
trait AveragePool7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def AveragePool7(name: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"count_include_pad" -> count_include_pad 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"AveragePool",allInputs, map))
}
}
trait AveragePool1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def AveragePool1(name: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"AveragePool",allInputs, map))
}
}

trait BatchNormalization12[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def BatchNormalization12(name: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,X: Tensor[T], scale: Tensor[T], B: Tensor[T], mean: Tensor[T], someVar: Tensor[T],training_mode: Option[Tensor[T1]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("epsilon" -> epsilon 
,"momentum" -> momentum 
)
val allInputs = Some(X,scale,B,mean,someVar,training_mode *: () )
(callOp[Tensor[T]](name,"BatchNormalization",allInputs, map))
}
}
trait BatchNormalization9[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def BatchNormalization9(name: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,X: Tensor[T], scale: Tensor[T], B: Tensor[T], mean: Tensor[T], someVar: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("epsilon" -> epsilon 
,"momentum" -> momentum 
)
val allInputs = Some(X,scale,B,mean,someVar *: () )
(callOp[Tensor[T]](name,"BatchNormalization",allInputs, map))
}
}
trait BatchNormalization7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def BatchNormalization7(name: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Tensor[T], scale: Tensor[T], B: Tensor[T], mean: Tensor[T], someVar: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("epsilon" -> epsilon 
,"momentum" -> momentum 
,"spatial" -> spatial 
)
val allInputs = Some(X,scale,B,mean,someVar *: () )
(callOp[Tensor[T]](name,"BatchNormalization",allInputs, map))
}
}
trait BatchNormalization6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def BatchNormalization6(name: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Tensor[T], scale: Tensor[T], B: Tensor[T], mean: Tensor[T], someVar: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("epsilon" -> epsilon 
,"is_test" -> is_test 
,"momentum" -> momentum 
,"spatial" -> spatial 
)
val allInputs = Some(X,scale,B,mean,someVar *: () )
(callOp[Tensor[T]](name,"BatchNormalization",allInputs, map))
}
}
trait BatchNormalization1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def BatchNormalization1(name: String,consumed_inputs : (Array[Int]),epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Tensor[T], scale: Tensor[T], B: Tensor[T], mean: Tensor[T], someVar: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
,"epsilon" -> epsilon 
,"is_test" -> is_test 
,"momentum" -> momentum 
,"spatial" -> spatial 
)
val allInputs = Some(X,scale,B,mean,someVar *: () )
(callOp[Tensor[T]](name,"BatchNormalization",allInputs, map))
}
}

trait Binarizer1[@sp T <: Float | Double | Long | Int : Numeric:ClassTag] extends Operator {
  def Binarizer1(name: String,threshold : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("threshold" -> threshold 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Binarizer",allInputs, map))
}
}

trait BitShift11[@sp T <: UByte | UShort | UInt | ULong : Numeric:ClassTag] extends Operator {
  def BitShift11(name: String,direction : (String),X: Tensor[T], Y: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("direction" -> direction 
)
val allInputs = Some(X,Y *: () )
(callOp[Tensor[T]](name,"BitShift",allInputs, map))
}
}

trait Cast9[@sp T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String : Numeric:ClassTag] extends Operator {
  def Cast9(name: String,to : (Int),input: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("to" -> to 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T2]](name,"Cast",allInputs, map))
}
}
trait Cast6[@sp T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String : Numeric:ClassTag] extends Operator {
  def Cast6(name: String,to : (Int),input: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("to" -> to 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T2]](name,"Cast",allInputs, map))
}
}
trait Cast1[@sp T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean | String : Numeric:ClassTag] extends Operator {
  def Cast1(name: String,to : (String),input: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("to" -> to 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T2]](name,"Cast",allInputs, map))
}
}

trait CastMap1[@sp T1 <: Map[Long, String] | Map[Long, Float] : Numeric:ClassTag,@sp T2 <: String | Float | Long : Numeric:ClassTag] extends Operator {
  def CastMap1(name: String,cast_to : Option[(String)] = None,map_form : Option[(String)] = None,max_map : Option[(Int)] = None,X: T1)
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("cast_to" -> cast_to 
,"map_form" -> map_form 
,"max_map" -> max_map 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"CastMap",allInputs, map))
}
}

trait CategoryMapper1[@sp T1 <: String | Long : Numeric:ClassTag,@sp T2 <: String | Long : Numeric:ClassTag] extends Operator {
  def CategoryMapper1(name: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,X: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("cats_int64s" -> cats_int64s 
,"cats_strings" -> cats_strings 
,"default_int64" -> default_int64 
,"default_string" -> default_string 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"CategoryMapper",allInputs, map))
}
}

trait Ceil6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Ceil6(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Ceil",allInputs, map))
}
}
trait Ceil1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Ceil1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Ceil",allInputs, map))
}
}

trait Celu12[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Celu12(name: String,alpha : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Celu",allInputs, map))
}
}

trait Clip12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Clip12(name: String,input: Tensor[T],min: Option[Tensor[T]] = None, max: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input,min,max *: () )
(callOp[Tensor[T]](name,"Clip",allInputs, map))
}
}
trait Clip11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Clip11(name: String,input: Tensor[T],min: Option[Tensor[T]] = None, max: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input,min,max *: () )
(callOp[Tensor[T]](name,"Clip",allInputs, map))
}
}
trait Clip6[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Clip6(name: String,max : Option[(Float)] = None,min : Option[(Float)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("max" -> max 
,"min" -> min 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Clip",allInputs, map))
}
}
trait Clip1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Clip1(name: String,consumed_inputs : Option[(Array[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
,"max" -> max 
,"min" -> min 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Clip",allInputs, map))
}
}

trait Compress11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Compress11(name: String,axis : Option[(Int)] = None,input: Tensor[T], condition: Tensor[T1])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input,condition *: () )
(callOp[Tensor[T]](name,"Compress",allInputs, map))
}
}
trait Compress9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Compress9(name: String,axis : Option[(Int)] = None,input: Tensor[T], condition: Tensor[T1])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input,condition *: () )
(callOp[Tensor[T]](name,"Compress",allInputs, map))
}
}

trait Concat11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Concat11(name: String,axis : (Int),inputs: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(inputs(0),inputs(1),inputs(2),inputs(3),inputs(4),inputs(5),inputs(6),inputs(7),inputs(8) *: () )
(callOp[Tensor[T]](name,"Concat",allInputs, map))
}
}
trait Concat4[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Concat4(name: String,axis : (Int),inputs: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(inputs(0),inputs(1),inputs(2),inputs(3),inputs(4),inputs(5),inputs(6),inputs(7),inputs(8) *: () )
(callOp[Tensor[T]](name,"Concat",allInputs, map))
}
}
trait Concat1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Concat1(name: String,axis : Option[(Int)] = None,inputs: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(inputs(0),inputs(1),inputs(2),inputs(3),inputs(4),inputs(5),inputs(6),inputs(7),inputs(8) *: () )
(callOp[Tensor[T]](name,"Concat",allInputs, map))
}
}

trait ConcatFromSequence11[@sp S <: Seq[Tensor[UByte]] | Seq[Tensor[UShort]] | Seq[Tensor[UInt]] | Seq[Tensor[ULong]] | Seq[Tensor[Byte]] | Seq[Tensor[Short]] | Seq[Tensor[Int]] | Seq[Tensor[Long]] | Seq[Tensor[Float16]] | Seq[Tensor[Float]] | Seq[Tensor[Double]] | Seq[Tensor[String]] | Seq[Tensor[Boolean]] | Seq[Tensor[Complex[Float]]] | Seq[Tensor[Complex[Double]]] : Numeric:ClassTag,@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def ConcatFromSequence11(name: String,axis : (Int),new_axis : Option[(Int)] = None,input_sequence: S)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"new_axis" -> new_axis 
)
val allInputs = Some(input_sequence *: () )
(callOp[Tensor[T]](name,"ConcatFromSequence",allInputs, map))
}
}

trait Constant12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Constant12(name: String,sparse_value : Option[(SparseTensor[T])] = None,value : Option[(Tensor[T])] = None,value_float : Option[(Float)] = None,value_floats : Option[(Array[Float])] = None,value_int : Option[(Int)] = None,value_ints : Option[(Array[Int])] = None,value_string : Option[(String)] = None,value_strings : Option[(Array[String])] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("sparse_value" -> sparse_value 
,"value" -> value 
,"value_float" -> value_float 
,"value_floats" -> value_floats 
,"value_int" -> value_int 
,"value_ints" -> value_ints 
,"value_string" -> value_string 
,"value_strings" -> value_strings 
)
val allInputs = None
(callOp[Tensor[T]](name,"Constant",allInputs, map))
}
}
trait Constant11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Constant11(name: String,sparse_value : Option[(SparseTensor[T])] = None,value : Option[(Tensor[T])] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("sparse_value" -> sparse_value 
,"value" -> value 
)
val allInputs = None
(callOp[Tensor[T]](name,"Constant",allInputs, map))
}
}
trait Constant9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Constant9(name: String,value : (Tensor[T]))
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("value" -> value 
)
val allInputs = None
(callOp[Tensor[T]](name,"Constant",allInputs, map))
}
}
trait Constant1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Constant1(name: String,value : (Tensor[T]))
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("value" -> value 
)
val allInputs = None
(callOp[Tensor[T]](name,"Constant",allInputs, map))
}
}

trait ConstantOfShape9[@sp T1 <: Long : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag] extends Operator {
  def ConstantOfShape9(name: String,value : Option[(Tensor[T2])] = None,input: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("value" -> value 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T2]](name,"ConstantOfShape",allInputs, map))
}
}

trait Conv11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Conv11(name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T], W: Tensor[T],B: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"dilations" -> dilations 
,"group" -> group 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X,W,B *: () )
(callOp[Tensor[T]](name,"Conv",allInputs, map))
}
}
trait Conv1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Conv1(name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T], W: Tensor[T],B: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"dilations" -> dilations 
,"group" -> group 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X,W,B *: () )
(callOp[Tensor[T]](name,"Conv",allInputs, map))
}
}

trait ConvInteger10[@sp T1 <: Byte | UByte : Numeric:ClassTag,@sp T2 <: Byte | UByte : Numeric:ClassTag,@sp T3 <: Int : Numeric:ClassTag] extends Operator {
  def ConvInteger10(name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,x: Tensor[T1], w: Tensor[T2],x_zero_point: Option[Tensor[T1]] = None, w_zero_point: Option[Tensor[T2]] = None)
    : Tuple1[Tensor[T3]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"dilations" -> dilations 
,"group" -> group 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(x,w,x_zero_point,w_zero_point *: () )
(callOp[Tensor[T3]](name,"ConvInteger",allInputs, map))
}
}

trait ConvTranspose11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ConvTranspose11(name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,output_padding : Option[(Array[Int])] = None,output_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T], W: Tensor[T],B: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"dilations" -> dilations 
,"group" -> group 
,"kernel_shape" -> kernel_shape 
,"output_padding" -> output_padding 
,"output_shape" -> output_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X,W,B *: () )
(callOp[Tensor[T]](name,"ConvTranspose",allInputs, map))
}
}
trait ConvTranspose1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ConvTranspose1(name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,output_padding : Option[(Array[Int])] = None,output_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T], W: Tensor[T],B: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"dilations" -> dilations 
,"group" -> group 
,"kernel_shape" -> kernel_shape 
,"output_padding" -> output_padding 
,"output_shape" -> output_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X,W,B *: () )
(callOp[Tensor[T]](name,"ConvTranspose",allInputs, map))
}
}

trait Cos7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Cos7(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Cos",allInputs, map))
}
}

trait Cosh9[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Cosh9(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Cosh",allInputs, map))
}
}

trait CumSum11[@sp T <: UInt | ULong | Int | Long | Float | Double : Numeric:ClassTag,@sp T2 <: Int | Long : Numeric:ClassTag] extends Operator {
  def CumSum11(name: String,exclusive : Option[(Int)] = None,reverse : Option[(Int)] = None,x: Tensor[T], axis: Tensor[T2])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("exclusive" -> exclusive 
,"reverse" -> reverse 
)
val allInputs = Some(x,axis *: () )
(callOp[Tensor[T]](name,"CumSum",allInputs, map))
}
}

trait DepthToSpace11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def DepthToSpace11(name: String,blocksize : (Int),mode : Option[(String)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("blocksize" -> blocksize 
,"mode" -> mode 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"DepthToSpace",allInputs, map))
}
}
trait DepthToSpace1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def DepthToSpace1(name: String,blocksize : (Int),input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("blocksize" -> blocksize 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"DepthToSpace",allInputs, map))
}
}

trait DequantizeLinear10[@sp T <: Byte | UByte | Int : Numeric:ClassTag] extends Operator {
  def DequantizeLinear10(name: String,x: Tensor[T], x_scale: Tensor[Float],x_zero_point: Option[Tensor[T]] = None)
    : Tuple1[Tensor[Float]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(x,x_scale,x_zero_point *: () )
(callOp[Tensor[Float]](name,"DequantizeLinear",allInputs, map))
}
}

trait Det11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Det11(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Det",allInputs, map))
}
}

trait DictVectorizer1[@sp T1 <: Map[String, Long] | Map[Long, String] | Map[Long, Float] | Map[Long, Double] | Map[String, Float] | Map[String, Double] : Numeric:ClassTag,@sp T2 <: Long | Float | Double | String : Numeric:ClassTag] extends Operator {
  def DictVectorizer1(name: String,int64_vocabulary : Option[(Array[Int])] = None,string_vocabulary : Option[(Array[String])] = None,X: T1)
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("int64_vocabulary" -> int64_vocabulary 
,"string_vocabulary" -> string_vocabulary 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"DictVectorizer",allInputs, map))
}
}

trait Div7[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Div7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Div",allInputs, map))
}
}
trait Div6[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Div6(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Div",allInputs, map))
}
}
trait Div1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Div1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
,"consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Div",allInputs, map))
}
}

trait Dropout12[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Float16 | Float | Double | Boolean : Numeric:ClassTag,@sp T2 <: Boolean : Numeric:ClassTag] extends Operator {
  def Dropout12(name: String,seed : Option[(Int)] = None,data: Tensor[T],ratio: Option[Tensor[T1]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("seed" -> seed 
)
val allInputs = Some(data,ratio *: () )
(callOp[Tensor[T]](name,"Dropout",allInputs, map))
}
}
trait Dropout10[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Float16 | Float | Double | Boolean : Numeric:ClassTag] extends Operator {
  def Dropout10(name: String,ratio : Option[(Float)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("ratio" -> ratio 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Dropout",allInputs, map))
}
}
trait Dropout7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Dropout7(name: String,ratio : Option[(Float)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("ratio" -> ratio 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Dropout",allInputs, map))
}
}
trait Dropout6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Dropout6(name: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("is_test" -> is_test 
,"ratio" -> ratio 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Dropout",allInputs, map))
}
}
trait Dropout1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Dropout1(name: String,consumed_inputs : Option[(Array[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
,"is_test" -> is_test 
,"ratio" -> ratio 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Dropout",allInputs, map))
}
}

trait DynamicQuantizeLinear11[@sp T1 <: Float : Numeric:ClassTag,@sp T2 <: UByte : Numeric:ClassTag] extends Operator {
  def DynamicQuantizeLinear11(name: String,x: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(x *: () )
(callOp[Tensor[T2]](name,"DynamicQuantizeLinear",allInputs, map))
}
}

trait Einsum12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Einsum12(name: String,equation : (String),Inputs: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("equation" -> equation 
)
val allInputs = Some(Inputs(0),Inputs(1),Inputs(2),Inputs(3),Inputs(4),Inputs(5),Inputs(6),Inputs(7),Inputs(8) *: () )
(callOp[Tensor[T]](name,"Einsum",allInputs, map))
}
}

trait Elu6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Elu6(name: String,alpha : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Elu",allInputs, map))
}
}
trait Elu1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Elu1(name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Elu",allInputs, map))
}
}

trait Equal11[@sp T <: Boolean | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Equal11(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Equal",allInputs, map))
}
}
trait Equal7[@sp T <: Boolean | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Equal7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Equal",allInputs, map))
}
}
trait Equal1[@sp T <: Boolean | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Equal1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Equal",allInputs, map))
}
}

trait Erf9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Erf9(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Erf",allInputs, map))
}
}

trait Exp6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Exp6(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Exp",allInputs, map))
}
}
trait Exp1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Exp1(name: String,consumed_inputs : Option[(Array[Int])] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Exp",allInputs, map))
}
}

trait Expand8[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Expand8(name: String,input: Tensor[T], shapeInput: Tensor[Long])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input,shapeInput *: () )
(callOp[Tensor[T]](name,"Expand",allInputs, map))
}
}

trait EyeLike9[@sp T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag] extends Operator {
  def EyeLike9(name: String,dtype : Option[(Int)] = None,k : Option[(Int)] = None,input: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("dtype" -> dtype 
,"k" -> k 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T2]](name,"EyeLike",allInputs, map))
}
}

trait FeatureVectorizer1[@sp T1 <: Int | Long | Float | Double : Numeric:ClassTag] extends Operator {
  def FeatureVectorizer1(name: String,inputdimensions : Option[(Array[Int])] = None,X: Seq[Tensor[T1]])
    : Tuple1[Tensor[Float]]
 = {
val map: Map[String, Any] = Map("inputdimensions" -> inputdimensions 
)
val allInputs = Some(X(0),X(1),X(2),X(3),X(4),X(5),X(6),X(7),X(8) *: () )
(callOp[Tensor[Float]](name,"FeatureVectorizer",allInputs, map))
}
}

trait Flatten11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Flatten11(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Flatten",allInputs, map))
}
}
trait Flatten9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Flatten9(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Flatten",allInputs, map))
}
}
trait Flatten1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Flatten1(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Flatten",allInputs, map))
}
}

trait Floor6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Floor6(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Floor",allInputs, map))
}
}
trait Floor1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Floor1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Floor",allInputs, map))
}
}

trait GRU7[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Int : Numeric:ClassTag] extends Operator {
  def GRU7(name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,X: Tensor[T], W: Tensor[T], R: Tensor[T],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("activation_alpha" -> activation_alpha 
,"activation_beta" -> activation_beta 
,"activations" -> activations 
,"clip" -> clip 
,"direction" -> direction 
,"hidden_size" -> hidden_size 
,"linear_before_reset" -> linear_before_reset 
)
val allInputs = Some(X,W,R,B,sequence_lens,initial_h *: () )
(callOp[Tensor[T]](name,"GRU",allInputs, map))
}
}
trait GRU3[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Int : Numeric:ClassTag] extends Operator {
  def GRU3(name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Tensor[T], W: Tensor[T], R: Tensor[T],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("activation_alpha" -> activation_alpha 
,"activation_beta" -> activation_beta 
,"activations" -> activations 
,"clip" -> clip 
,"direction" -> direction 
,"hidden_size" -> hidden_size 
,"linear_before_reset" -> linear_before_reset 
,"output_sequence" -> output_sequence 
)
val allInputs = Some(X,W,R,B,sequence_lens,initial_h *: () )
(callOp[Tensor[T]](name,"GRU",allInputs, map))
}
}
trait GRU1[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Int : Numeric:ClassTag] extends Operator {
  def GRU1(name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Tensor[T], W: Tensor[T], R: Tensor[T],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("activation_alpha" -> activation_alpha 
,"activation_beta" -> activation_beta 
,"activations" -> activations 
,"clip" -> clip 
,"direction" -> direction 
,"hidden_size" -> hidden_size 
,"output_sequence" -> output_sequence 
)
val allInputs = Some(X,W,R,B,sequence_lens,initial_h *: () )
(callOp[Tensor[T]](name,"GRU",allInputs, map))
}
}

trait Gather11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def Gather11(name: String,axis : Option[(Int)] = None,data: Tensor[T], indices: Tensor[Tind])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(data,indices *: () )
(callOp[Tensor[T]](name,"Gather",allInputs, map))
}
}
trait Gather1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def Gather1(name: String,axis : Option[(Int)] = None,data: Tensor[T], indices: Tensor[Tind])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(data,indices *: () )
(callOp[Tensor[T]](name,"Gather",allInputs, map))
}
}

trait GatherElements11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def GatherElements11(name: String,axis : Option[(Int)] = None,data: Tensor[T], indices: Tensor[Tind])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(data,indices *: () )
(callOp[Tensor[T]](name,"GatherElements",allInputs, map))
}
}

trait GatherND12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def GatherND12(name: String,batch_dims : Option[(Int)] = None,data: Tensor[T], indices: Tensor[Long])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("batch_dims" -> batch_dims 
)
val allInputs = Some(data,indices *: () )
(callOp[Tensor[T]](name,"GatherND",allInputs, map))
}
}
trait GatherND11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def GatherND11(name: String,data: Tensor[T], indices: Tensor[Long])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data,indices *: () )
(callOp[Tensor[T]](name,"GatherND",allInputs, map))
}
}

trait Gemm11[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def Gemm11(name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Tensor[T], B: Tensor[T],C: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
,"transA" -> transA 
,"transB" -> transB 
)
val allInputs = Some(A,B,C *: () )
(callOp[Tensor[T]](name,"Gemm",allInputs, map))
}
}
trait Gemm9[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def Gemm9(name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Tensor[T], B: Tensor[T], C: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
,"transA" -> transA 
,"transB" -> transB 
)
val allInputs = Some(A,B,C *: () )
(callOp[Tensor[T]](name,"Gemm",allInputs, map))
}
}
trait Gemm7[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def Gemm7(name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Tensor[T], B: Tensor[T], C: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
,"transA" -> transA 
,"transB" -> transB 
)
val allInputs = Some(A,B,C *: () )
(callOp[Tensor[T]](name,"Gemm",allInputs, map))
}
}
trait Gemm6[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def Gemm6(name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Tensor[T], B: Tensor[T], C: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
,"broadcast" -> broadcast 
,"transA" -> transA 
,"transB" -> transB 
)
val allInputs = Some(A,B,C *: () )
(callOp[Tensor[T]](name,"Gemm",allInputs, map))
}
}
trait Gemm1[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def Gemm1(name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Tensor[T], B: Tensor[T], C: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
,"broadcast" -> broadcast 
,"transA" -> transA 
,"transB" -> transB 
)
val allInputs = Some(A,B,C *: () )
(callOp[Tensor[T]](name,"Gemm",allInputs, map))
}
}

trait GlobalAveragePool1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def GlobalAveragePool1(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"GlobalAveragePool",allInputs, map))
}
}

trait GlobalLpPool2[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def GlobalLpPool2(name: String,p : Option[(Int)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("p" -> p 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"GlobalLpPool",allInputs, map))
}
}
trait GlobalLpPool1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def GlobalLpPool1(name: String,p : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("p" -> p 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"GlobalLpPool",allInputs, map))
}
}

trait GlobalMaxPool1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def GlobalMaxPool1(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"GlobalMaxPool",allInputs, map))
}
}

trait Gradient1[@sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Gradient1(name: String,xs : (Array[String]),y : (String),zs : Option[(Array[String])] = None,Inputs: Seq[Tensor[T1]])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("xs" -> xs 
,"y" -> y 
,"zs" -> zs 
)
val allInputs = Some(Inputs(0),Inputs(1),Inputs(2),Inputs(3),Inputs(4),Inputs(5),Inputs(6),Inputs(7),Inputs(8) *: () )
(callOp[Tensor[T2]](name,"Gradient",allInputs, map))
}
}

trait GraphCall1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def GraphCall1(name: String,graph_name : (String),Inputs: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("graph_name" -> graph_name 
)
val allInputs = Some(Inputs(0),Inputs(1),Inputs(2),Inputs(3),Inputs(4),Inputs(5),Inputs(6),Inputs(7),Inputs(8) *: () )
(callOp[Tensor[T]](name,"GraphCall",allInputs, map))
}
}

trait Greater9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Greater9(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Greater",allInputs, map))
}
}
trait Greater7[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Greater7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Greater",allInputs, map))
}
}
trait Greater1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Greater1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Greater",allInputs, map))
}
}

trait GreaterOrEqual12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def GreaterOrEqual12(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"GreaterOrEqual",allInputs, map))
}
}

trait HardSigmoid6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def HardSigmoid6(name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"HardSigmoid",allInputs, map))
}
}
trait HardSigmoid1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def HardSigmoid1(name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
,"consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"HardSigmoid",allInputs, map))
}
}

trait Hardmax11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Hardmax11(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Hardmax",allInputs, map))
}
}
trait Hardmax1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Hardmax1(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Hardmax",allInputs, map))
}
}

trait Identity1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Identity1(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Identity",allInputs, map))
}
}

trait If11[@sp B <: Boolean : Numeric:ClassTag,@sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def If11(name: String,else_branch : (Graph),then_branch : (Graph),cond: Tensor[B])
    : Tuple1[Tensor[V]]
 = {
val map: Map[String, Any] = Map("else_branch" -> else_branch 
,"then_branch" -> then_branch 
)
val allInputs = Some(cond *: () )
(callOp[Tensor[V]](name,"If",allInputs, map))
}
}
trait If1[@sp B <: Boolean : Numeric:ClassTag,@sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def If1(name: String,else_branch : (Graph),then_branch : (Graph),cond: Tensor[B])
    : Tuple1[Tensor[V]]
 = {
val map: Map[String, Any] = Map("else_branch" -> else_branch 
,"then_branch" -> then_branch 
)
val allInputs = Some(cond *: () )
(callOp[Tensor[V]](name,"If",allInputs, map))
}
}

trait Imputer1[@sp T <: Float | Double | Long | Int : Numeric:ClassTag] extends Operator {
  def Imputer1(name: String,imputed_value_floats : Option[(Array[Float])] = None,imputed_value_int64s : Option[(Array[Int])] = None,replaced_value_float : Option[(Float)] = None,replaced_value_int64 : Option[(Int)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("imputed_value_floats" -> imputed_value_floats 
,"imputed_value_int64s" -> imputed_value_int64s 
,"replaced_value_float" -> replaced_value_float 
,"replaced_value_int64" -> replaced_value_int64 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Imputer",allInputs, map))
}
}

trait InstanceNormalization6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def InstanceNormalization6(name: String,epsilon : Option[(Float)] = None,input: Tensor[T], scale: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("epsilon" -> epsilon 
)
val allInputs = Some(input,scale,B *: () )
(callOp[Tensor[T]](name,"InstanceNormalization",allInputs, map))
}
}
trait InstanceNormalization1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def InstanceNormalization1(name: String,consumed_inputs : Option[(Array[Int])] = None,epsilon : Option[(Float)] = None,input: Tensor[T], scale: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
,"epsilon" -> epsilon 
)
val allInputs = Some(input,scale,B *: () )
(callOp[Tensor[T]](name,"InstanceNormalization",allInputs, map))
}
}

trait Inverse12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Inverse12(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Inverse",allInputs, map))
}
}

trait IsInf10[@sp T1 <: Float | Double : Numeric:ClassTag,@sp T2 <: Boolean : Numeric:ClassTag] extends Operator {
  def IsInf10(name: String,detect_negative : Option[(Int)] = None,detect_positive : Option[(Int)] = None,X: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("detect_negative" -> detect_negative 
,"detect_positive" -> detect_positive 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"IsInf",allInputs, map))
}
}

trait IsNaN9[@sp T1 <: Float16 | Float | Double : Numeric:ClassTag,@sp T2 <: Boolean : Numeric:ClassTag] extends Operator {
  def IsNaN9(name: String,X: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"IsNaN",allInputs, map))
}
}

trait LRN1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LRN1(name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : (Int),X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
,"bias" -> bias 
,"size" -> size 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"LRN",allInputs, map))
}
}

trait LSTM7[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Int : Numeric:ClassTag] extends Operator {
  def LSTM7(name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,X: Tensor[T], W: Tensor[T], R: Tensor[T],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("activation_alpha" -> activation_alpha 
,"activation_beta" -> activation_beta 
,"activations" -> activations 
,"clip" -> clip 
,"direction" -> direction 
,"hidden_size" -> hidden_size 
,"input_forget" -> input_forget 
)
val allInputs = Some(X,W,R,B,sequence_lens,initial_h,initial_c,P *: () )
(callOp[Tensor[T]](name,"LSTM",allInputs, map))
}
}
trait LSTM1[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Int : Numeric:ClassTag] extends Operator {
  def LSTM1(name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Tensor[T], W: Tensor[T], R: Tensor[T],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("activation_alpha" -> activation_alpha 
,"activation_beta" -> activation_beta 
,"activations" -> activations 
,"clip" -> clip 
,"direction" -> direction 
,"hidden_size" -> hidden_size 
,"input_forget" -> input_forget 
,"output_sequence" -> output_sequence 
)
val allInputs = Some(X,W,R,B,sequence_lens,initial_h,initial_c,P *: () )
(callOp[Tensor[T]](name,"LSTM",allInputs, map))
}
}

trait LabelEncoder2[@sp T1 <: String | Long | Float : Numeric:ClassTag,@sp T2 <: String | Long | Float : Numeric:ClassTag] extends Operator {
  def LabelEncoder2(name: String,default_float : Option[(Float)] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,keys_floats : Option[(Array[Float])] = None,keys_int64s : Option[(Array[Int])] = None,keys_strings : Option[(Array[String])] = None,values_floats : Option[(Array[Float])] = None,values_int64s : Option[(Array[Int])] = None,values_strings : Option[(Array[String])] = None,X: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("default_float" -> default_float 
,"default_int64" -> default_int64 
,"default_string" -> default_string 
,"keys_floats" -> keys_floats 
,"keys_int64s" -> keys_int64s 
,"keys_strings" -> keys_strings 
,"values_floats" -> values_floats 
,"values_int64s" -> values_int64s 
,"values_strings" -> values_strings 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"LabelEncoder",allInputs, map))
}
}
trait LabelEncoder1[@sp T1 <: String | Long | Float : Numeric:ClassTag,@sp T2 <: String | Long | Float : Numeric:ClassTag] extends Operator {
  def LabelEncoder1(name: String,classes_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,X: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("classes_strings" -> classes_strings 
,"default_int64" -> default_int64 
,"default_string" -> default_string 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"LabelEncoder",allInputs, map))
}
}

trait LeakyRelu6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LeakyRelu6(name: String,alpha : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"LeakyRelu",allInputs, map))
}
}
trait LeakyRelu1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LeakyRelu1(name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"LeakyRelu",allInputs, map))
}
}

trait Less9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Less9(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Less",allInputs, map))
}
}
trait Less7[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Less7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Less",allInputs, map))
}
}
trait Less1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Less1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Less",allInputs, map))
}
}

trait LessOrEqual12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def LessOrEqual12(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"LessOrEqual",allInputs, map))
}
}

trait LinearClassifier1[@sp T1 <: Float | Double | Long | Int : Numeric:ClassTag,@sp T2 <: String | Long : Numeric:ClassTag] extends Operator {
  def LinearClassifier1(name: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : (Array[Float]),intercepts : Option[(Array[Float])] = None,multi_class : Option[(Int)] = None,post_transform : Option[(String)] = None,X: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("classlabels_ints" -> classlabels_ints 
,"classlabels_strings" -> classlabels_strings 
,"coefficients" -> coefficients 
,"intercepts" -> intercepts 
,"multi_class" -> multi_class 
,"post_transform" -> post_transform 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"LinearClassifier",allInputs, map))
}
}

trait LinearRegressor1[@sp T <: Float | Double | Long | Int : Numeric:ClassTag] extends Operator {
  def LinearRegressor1(name: String,coefficients : Option[(Array[Float])] = None,intercepts : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,targets : Option[(Int)] = None,X: Tensor[T])
    : Tuple1[Tensor[Float]]
 = {
val map: Map[String, Any] = Map("coefficients" -> coefficients 
,"intercepts" -> intercepts 
,"post_transform" -> post_transform 
,"targets" -> targets 
)
val allInputs = Some(X *: () )
(callOp[Tensor[Float]](name,"LinearRegressor",allInputs, map))
}
}

trait Log6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Log6(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Log",allInputs, map))
}
}
trait Log1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Log1(name: String,consumed_inputs : Option[(Array[Int])] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Log",allInputs, map))
}
}

trait LogSoftmax11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LogSoftmax11(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"LogSoftmax",allInputs, map))
}
}
trait LogSoftmax1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LogSoftmax1(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"LogSoftmax",allInputs, map))
}
}

trait Loop11[@sp I <: Long : Numeric:ClassTag,@sp B <: Boolean : Numeric:ClassTag,@sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Loop11(name: String,body : (Graph),M: Option[Tensor[I]] = None, cond: Option[Tensor[B]] = None,v_initial: Seq[Tensor[V]])
    : Tuple1[Tensor[V]]
 = {
val map: Map[String, Any] = Map("body" -> body 
)
val allInputs = Some(M,cond,v_initial(0),v_initial(1),v_initial(2),v_initial(3),v_initial(4),v_initial(5),v_initial(6) *: () )
(callOp[Tensor[V]](name,"Loop",allInputs, map))
}
}
trait Loop1[@sp I <: Long : Numeric:ClassTag,@sp B <: Boolean : Numeric:ClassTag,@sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Loop1(name: String,body : (Graph),M: Option[Tensor[I]] = None, cond: Option[Tensor[B]] = None,v_initial: Seq[Tensor[V]])
    : Tuple1[Tensor[V]]
 = {
val map: Map[String, Any] = Map("body" -> body 
)
val allInputs = Some(M,cond,v_initial(0),v_initial(1),v_initial(2),v_initial(3),v_initial(4),v_initial(5),v_initial(6) *: () )
(callOp[Tensor[V]](name,"Loop",allInputs, map))
}
}

trait LpNormalization1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LpNormalization1(name: String,axis : Option[(Int)] = None,p : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"p" -> p 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"LpNormalization",allInputs, map))
}
}

trait LpPool11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LpPool11(name: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),p : Option[(Int)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"kernel_shape" -> kernel_shape 
,"p" -> p 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"LpPool",allInputs, map))
}
}
trait LpPool2[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LpPool2(name: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),p : Option[(Int)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"kernel_shape" -> kernel_shape 
,"p" -> p 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"LpPool",allInputs, map))
}
}
trait LpPool1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def LpPool1(name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])] = None,p : Option[(Float)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"kernel_shape" -> kernel_shape 
,"p" -> p 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"LpPool",allInputs, map))
}
}

trait MatMul9[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def MatMul9(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"MatMul",allInputs, map))
}
}
trait MatMul1[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def MatMul1(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"MatMul",allInputs, map))
}
}

trait MatMulInteger10[@sp T1 <: Byte | UByte : Numeric:ClassTag,@sp T2 <: Byte | UByte : Numeric:ClassTag,@sp T3 <: Int : Numeric:ClassTag] extends Operator {
  def MatMulInteger10(name: String,A: Tensor[T1], B: Tensor[T2],a_zero_point: Option[Tensor[T1]] = None, b_zero_point: Option[Tensor[T2]] = None)
    : Tuple1[Tensor[T3]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B,a_zero_point,b_zero_point *: () )
(callOp[Tensor[T3]](name,"MatMulInteger",allInputs, map))
}
}

trait Max12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Max12(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Max",allInputs, map))
}
}
trait Max8[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Max8(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Max",allInputs, map))
}
}
trait Max6[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Max6(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Max",allInputs, map))
}
}
trait Max1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Max1(name: String,consumed_inputs : Option[(Array[Int])] = None,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Max",allInputs, map))
}
}

trait MaxPool12[@sp T <: Float16 | Float | Double | Byte | UByte : Numeric:ClassTag,@sp I <: Long : Numeric:ClassTag] extends Operator {
  def MaxPool12(name: String,auto_pad : Option[(String)] = None,ceil_mode : Option[(Int)] = None,dilations : Option[(Array[Int])] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"ceil_mode" -> ceil_mode 
,"dilations" -> dilations 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"storage_order" -> storage_order 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"MaxPool",allInputs, map))
}
}
trait MaxPool11[@sp T <: Float16 | Float | Double | Byte | UByte : Numeric:ClassTag,@sp I <: Long : Numeric:ClassTag] extends Operator {
  def MaxPool11(name: String,auto_pad : Option[(String)] = None,ceil_mode : Option[(Int)] = None,dilations : Option[(Array[Int])] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"ceil_mode" -> ceil_mode 
,"dilations" -> dilations 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"storage_order" -> storage_order 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"MaxPool",allInputs, map))
}
}
trait MaxPool10[@sp T <: Float16 | Float | Double | Byte | UByte : Numeric:ClassTag,@sp I <: Long : Numeric:ClassTag] extends Operator {
  def MaxPool10(name: String,auto_pad : Option[(String)] = None,ceil_mode : Option[(Int)] = None,dilations : Option[(Array[Int])] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"ceil_mode" -> ceil_mode 
,"dilations" -> dilations 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"storage_order" -> storage_order 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"MaxPool",allInputs, map))
}
}
trait MaxPool8[@sp T <: Float16 | Float | Double | Byte | UByte : Numeric:ClassTag,@sp I <: Long : Numeric:ClassTag] extends Operator {
  def MaxPool8(name: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"storage_order" -> storage_order 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"MaxPool",allInputs, map))
}
}
trait MaxPool1[@sp T <: Float16 | Float | Double | Byte | UByte : Numeric:ClassTag] extends Operator {
  def MaxPool1(name: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"MaxPool",allInputs, map))
}
}

trait MaxRoiPool1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def MaxRoiPool1(name: String,pooled_shape : (Array[Int]),spatial_scaleAttr : Option[(Float)] = None,X: Tensor[T], rois: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("pooled_shape" -> pooled_shape 
,"spatial_scaleAttr" -> spatial_scaleAttr 
)
val allInputs = Some(X,rois *: () )
(callOp[Tensor[T]](name,"MaxRoiPool",allInputs, map))
}
}

trait MaxUnpool11[@sp T1 <: Float16 | Float | Double : Numeric:ClassTag,@sp T2 <: Long : Numeric:ClassTag] extends Operator {
  def MaxUnpool11(name: String,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T1], I: Tensor[T2],output_shapeInput: Option[Tensor[T2]] = None)
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X,I,output_shapeInput *: () )
(callOp[Tensor[T1]](name,"MaxUnpool",allInputs, map))
}
}
trait MaxUnpool9[@sp T1 <: Float16 | Float | Double : Numeric:ClassTag,@sp T2 <: Long : Numeric:ClassTag] extends Operator {
  def MaxUnpool9(name: String,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T1], I: Tensor[T2],output_shapeInput: Option[Tensor[T2]] = None)
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X,I,output_shapeInput *: () )
(callOp[Tensor[T1]](name,"MaxUnpool",allInputs, map))
}
}

trait Mean8[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Mean8(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Mean",allInputs, map))
}
}
trait Mean6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Mean6(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Mean",allInputs, map))
}
}
trait Mean1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Mean1(name: String,consumed_inputs : Option[(Array[Int])] = None,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Mean",allInputs, map))
}
}

trait MeanSquaredDistance12[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def MeanSquaredDistance12(name: String,reduction : Option[(String)] = None,scores: Tensor[T], labels: Tensor[T],weights: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("reduction" -> reduction 
)
val allInputs = Some(scores,labels,weights *: () )
(callOp[Tensor[T]](name,"MeanSquaredDistance",allInputs, map))
}
}

trait MeanVarianceNormalization9[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def MeanVarianceNormalization9(name: String,axes : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"MeanVarianceNormalization",allInputs, map))
}
}

trait Min12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Min12(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Min",allInputs, map))
}
}
trait Min8[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Min8(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Min",allInputs, map))
}
}
trait Min6[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Min6(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Min",allInputs, map))
}
}
trait Min1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Min1(name: String,consumed_inputs : Option[(Array[Int])] = None,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Min",allInputs, map))
}
}

trait Mod10[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Mod10(name: String,fmod : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("fmod" -> fmod 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Mod",allInputs, map))
}
}

trait Momentum1[@sp T1 <: Float | Double : Numeric:ClassTag,@sp T2 <: Long : Numeric:ClassTag,@sp T3 <: Float | Double : Numeric:ClassTag] extends Operator {
  def Momentum1(name: String,alpha : (Float),beta : (Float),mode : (String),norm_coefficient : (Float),R: Tensor[T1], T: Tensor[T2],inputs: Seq[Tensor[T3]])
    : Tuple1[Tensor[T3]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"beta" -> beta 
,"mode" -> mode 
,"norm_coefficient" -> norm_coefficient 
)
val allInputs = Some(R,T,inputs(0),inputs(1),inputs(2),inputs(3),inputs(4),inputs(5),inputs(6) *: () )
(callOp[Tensor[T3]](name,"Momentum",allInputs, map))
}
}

trait Mul7[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Mul7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Mul",allInputs, map))
}
}
trait Mul6[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Mul6(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Mul",allInputs, map))
}
}
trait Mul1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Mul1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
,"consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Mul",allInputs, map))
}
}

trait Multinomial7[@sp T1 <: Float16 | Float | Double : Numeric:ClassTag,@sp T2 <: Int | Long : Numeric:ClassTag] extends Operator {
  def Multinomial7(name: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None,input: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("dtype" -> dtype 
,"sample_size" -> sample_size 
,"seed" -> seed 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T2]](name,"Multinomial",allInputs, map))
}
}

trait Neg6[@sp T <: Float | Int | Byte | Short | Long | Float16 | Double : Numeric:ClassTag] extends Operator {
  def Neg6(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Neg",allInputs, map))
}
}
trait Neg1[@sp T <: Float | Int | Byte | Short | Long | Float16 | Double : Numeric:ClassTag] extends Operator {
  def Neg1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Neg",allInputs, map))
}
}

trait NegativeLogLikelihoodLoss12[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def NegativeLogLikelihoodLoss12(name: String,reduction : Option[(String)] = None,input: Tensor[T], target: Tensor[Tind],weight: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("reduction" -> reduction 
)
val allInputs = Some(input,target,weight *: () )
(callOp[Tensor[T]](name,"NegativeLogLikelihoodLoss",allInputs, map))
}
}

trait NonMaxSuppression11 extends Operator {
  def NonMaxSuppression11(name: String,center_point_box : Option[(Int)] = None,boxes: Tensor[Float], scores: Tensor[Float],max_output_boxes_per_class: Option[Tensor[Long]] = None, iou_threshold: Option[Tensor[Float]] = None, score_threshold: Option[Tensor[Float]] = None)
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map("center_point_box" -> center_point_box 
)
val allInputs = Some(boxes,scores,max_output_boxes_per_class,iou_threshold,score_threshold *: () )
(callOp[Tensor[Long]](name,"NonMaxSuppression",allInputs, map))
}
}
trait NonMaxSuppression10 extends Operator {
  def NonMaxSuppression10(name: String,center_point_box : Option[(Int)] = None,boxes: Tensor[Float], scores: Tensor[Float],max_output_boxes_per_class: Option[Tensor[Long]] = None, iou_threshold: Option[Tensor[Float]] = None, score_threshold: Option[Tensor[Float]] = None)
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map("center_point_box" -> center_point_box 
)
val allInputs = Some(boxes,scores,max_output_boxes_per_class,iou_threshold,score_threshold *: () )
(callOp[Tensor[Long]](name,"NonMaxSuppression",allInputs, map))
}
}

trait NonZero9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def NonZero9(name: String,X: Tensor[T])
    : Tuple1[Tensor[Long]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[Long]](name,"NonZero",allInputs, map))
}
}

trait Normalizer1[@sp T <: Float | Double | Long | Int : Numeric:ClassTag] extends Operator {
  def Normalizer1(name: String,norm : Option[(String)] = None,X: Tensor[T])
    : Tuple1[Tensor[Float]]
 = {
val map: Map[String, Any] = Map("norm" -> norm 
)
val allInputs = Some(X *: () )
(callOp[Tensor[Float]](name,"Normalizer",allInputs, map))
}
}

trait Not1[@sp T <: Boolean : Numeric:ClassTag] extends Operator {
  def Not1(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Not",allInputs, map))
}
}

trait OneHot11[@sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T2 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T3 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def OneHot11(name: String,axis : Option[(Int)] = None,indices: Tensor[T1], depth: Tensor[T2], values: Tensor[T3])
    : Tuple1[Tensor[T3]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(indices,depth,values *: () )
(callOp[Tensor[T3]](name,"OneHot",allInputs, map))
}
}
trait OneHot9[@sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T2 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T3 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def OneHot9(name: String,axis : Option[(Int)] = None,indices: Tensor[T1], depth: Tensor[T2], values: Tensor[T3])
    : Tuple1[Tensor[T3]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(indices,depth,values *: () )
(callOp[Tensor[T3]](name,"OneHot",allInputs, map))
}
}

trait OneHotEncoder1[@sp T <: String | Long | Int | Float | Double : Numeric:ClassTag] extends Operator {
  def OneHotEncoder1(name: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,zeros : Option[(Int)] = None,X: Tensor[T])
    : Tuple1[Tensor[Float]]
 = {
val map: Map[String, Any] = Map("cats_int64s" -> cats_int64s 
,"cats_strings" -> cats_strings 
,"zeros" -> zeros 
)
val allInputs = Some(X *: () )
(callOp[Tensor[Float]](name,"OneHotEncoder",allInputs, map))
}
}

trait Or7[@sp T <: Boolean : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Or7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Or",allInputs, map))
}
}
trait Or1[@sp T <: Boolean : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Or1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Or",allInputs, map))
}
}

trait PRelu9[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def PRelu9(name: String,X: Tensor[T], slope: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X,slope *: () )
(callOp[Tensor[T]](name,"PRelu",allInputs, map))
}
}
trait PRelu7[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def PRelu7(name: String,X: Tensor[T], slope: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X,slope *: () )
(callOp[Tensor[T]](name,"PRelu",allInputs, map))
}
}
trait PRelu6[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def PRelu6(name: String,X: Tensor[T], slope: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X,slope *: () )
(callOp[Tensor[T]](name,"PRelu",allInputs, map))
}
}
trait PRelu1[@sp T <: Float16 | Float | Double | UInt | ULong | Int | Long : Numeric:ClassTag] extends Operator {
  def PRelu1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T], slope: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X,slope *: () )
(callOp[Tensor[T]](name,"PRelu",allInputs, map))
}
}

trait Pad11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Pad11(name: String,mode : Option[(String)] = None,data: Tensor[T], pads: Tensor[Long],constant_value: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("mode" -> mode 
)
val allInputs = Some(data,pads,constant_value *: () )
(callOp[Tensor[T]](name,"Pad",allInputs, map))
}
}
trait Pad2[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Pad2(name: String,mode : Option[(String)] = None,pads : (Array[Int]),value : Option[(Float)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("mode" -> mode 
,"pads" -> pads 
,"value" -> value 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Pad",allInputs, map))
}
}
trait Pad1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Pad1(name: String,mode : Option[(String)] = None,paddings : (Array[Int]),value : Option[(Float)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("mode" -> mode 
,"paddings" -> paddings 
,"value" -> value 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Pad",allInputs, map))
}
}

trait Pow12[@sp T <: Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Pow12(name: String,X: Tensor[T], Y: Tensor[T1])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X,Y *: () )
(callOp[Tensor[T]](name,"Pow",allInputs, map))
}
}
trait Pow7[@sp T <: Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Pow7(name: String,X: Tensor[T], Y: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X,Y *: () )
(callOp[Tensor[T]](name,"Pow",allInputs, map))
}
}
trait Pow1[@sp T <: Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Pow1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,X: Tensor[T], Y: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(X,Y *: () )
(callOp[Tensor[T]](name,"Pow",allInputs, map))
}
}

trait QLinearConv10[@sp T1 <: Byte | UByte : Numeric:ClassTag,@sp T2 <: Byte | UByte : Numeric:ClassTag,@sp T3 <: Byte | UByte : Numeric:ClassTag,@sp T4 <: Int : Numeric:ClassTag] extends Operator {
  def QLinearConv10(name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,x: Tensor[T1], x_scale: Tensor[Float], x_zero_point: Tensor[T1], w: Tensor[T2], w_scale: Tensor[Float], w_zero_point: Tensor[T2], y_scale: Tensor[Float], y_zero_point: Tensor[T3],B: Option[Tensor[T4]] = None)
    : Tuple1[Tensor[T3]]
 = {
val map: Map[String, Any] = Map("auto_pad" -> auto_pad 
,"dilations" -> dilations 
,"group" -> group 
,"kernel_shape" -> kernel_shape 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(x,x_scale,x_zero_point,w,w_scale,w_zero_point,y_scale,y_zero_point,B *: () )
(callOp[Tensor[T3]](name,"QLinearConv",allInputs, map))
}
}

trait QLinearMatMul10[@sp T1 <: Byte | UByte : Numeric:ClassTag,@sp T2 <: Byte | UByte : Numeric:ClassTag,@sp T3 <: Byte | UByte : Numeric:ClassTag] extends Operator {
  def QLinearMatMul10(name: String,a: Tensor[T1], a_scale: Tensor[Float], a_zero_point: Tensor[T1], b: Tensor[T2], b_scale: Tensor[Float], b_zero_point: Tensor[T2], y_scale: Tensor[Float], y_zero_point: Tensor[T3])
    : Tuple1[Tensor[T3]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(a,a_scale,a_zero_point,b,b_scale,b_zero_point,y_scale,y_zero_point *: () )
(callOp[Tensor[T3]](name,"QLinearMatMul",allInputs, map))
}
}

trait QuantizeLinear10[@sp T1 <: Float | Int : Numeric:ClassTag,@sp T2 <: Byte | UByte : Numeric:ClassTag] extends Operator {
  def QuantizeLinear10(name: String,x: Tensor[T1], y_scale: Tensor[Float],y_zero_point: Option[Tensor[T2]] = None)
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(x,y_scale,y_zero_point *: () )
(callOp[Tensor[T2]](name,"QuantizeLinear",allInputs, map))
}
}

trait RNN7[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Int : Numeric:ClassTag] extends Operator {
  def RNN7(name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,X: Tensor[T], W: Tensor[T], R: Tensor[T],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("activation_alpha" -> activation_alpha 
,"activation_beta" -> activation_beta 
,"activations" -> activations 
,"clip" -> clip 
,"direction" -> direction 
,"hidden_size" -> hidden_size 
)
val allInputs = Some(X,W,R,B,sequence_lens,initial_h *: () )
(callOp[Tensor[T]](name,"RNN",allInputs, map))
}
}
trait RNN1[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp T1 <: Int : Numeric:ClassTag] extends Operator {
  def RNN1(name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Tensor[T], W: Tensor[T], R: Tensor[T],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("activation_alpha" -> activation_alpha 
,"activation_beta" -> activation_beta 
,"activations" -> activations 
,"clip" -> clip 
,"direction" -> direction 
,"hidden_size" -> hidden_size 
,"output_sequence" -> output_sequence 
)
val allInputs = Some(X,W,R,B,sequence_lens,initial_h *: () )
(callOp[Tensor[T]](name,"RNN",allInputs, map))
}
}

trait RandomNormal1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def RandomNormal1(name: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None,shape : (Array[Int]))
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("dtype" -> dtype 
,"mean" -> mean 
,"scaleAttr" -> scaleAttr 
,"seed" -> seed 
,"shape" -> shape 
)
val allInputs = None
(callOp[Tensor[T]](name,"RandomNormal",allInputs, map))
}
}

trait RandomNormalLike1[@sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def RandomNormalLike1(name: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None,input: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("dtype" -> dtype 
,"mean" -> mean 
,"scaleAttr" -> scaleAttr 
,"seed" -> seed 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T2]](name,"RandomNormalLike",allInputs, map))
}
}

trait RandomUniform1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def RandomUniform1(name: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None,shape : (Array[Int]))
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("dtype" -> dtype 
,"high" -> high 
,"low" -> low 
,"seed" -> seed 
,"shape" -> shape 
)
val allInputs = None
(callOp[Tensor[T]](name,"RandomUniform",allInputs, map))
}
}

trait RandomUniformLike1[@sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def RandomUniformLike1(name: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None,input: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("dtype" -> dtype 
,"high" -> high 
,"low" -> low 
,"seed" -> seed 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T2]](name,"RandomUniformLike",allInputs, map))
}
}

trait Range11[@sp T <: Float | Double | Short | Int | Long : Numeric:ClassTag] extends Operator {
  def Range11(name: String,start: Tensor[T], limit: Tensor[T], delta: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(start,limit,delta *: () )
(callOp[Tensor[T]](name,"Range",allInputs, map))
}
}

trait Reciprocal6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Reciprocal6(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Reciprocal",allInputs, map))
}
}
trait Reciprocal1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Reciprocal1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Reciprocal",allInputs, map))
}
}

trait ReduceL111[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceL111(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceL1",allInputs, map))
}
}
trait ReduceL11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceL11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceL1",allInputs, map))
}
}

trait ReduceL211[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceL211(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceL2",allInputs, map))
}
}
trait ReduceL21[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceL21(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceL2",allInputs, map))
}
}

trait ReduceLogSum11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceLogSum11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceLogSum",allInputs, map))
}
}
trait ReduceLogSum1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceLogSum1(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceLogSum",allInputs, map))
}
}

trait ReduceLogSumExp11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceLogSumExp11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceLogSumExp",allInputs, map))
}
}
trait ReduceLogSumExp1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceLogSumExp1(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceLogSumExp",allInputs, map))
}
}

trait ReduceMax12[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte : Numeric:ClassTag] extends Operator {
  def ReduceMax12(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceMax",allInputs, map))
}
}
trait ReduceMax11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte : Numeric:ClassTag] extends Operator {
  def ReduceMax11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceMax",allInputs, map))
}
}
trait ReduceMax1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte : Numeric:ClassTag] extends Operator {
  def ReduceMax1(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceMax",allInputs, map))
}
}

trait ReduceMean11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceMean11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceMean",allInputs, map))
}
}
trait ReduceMean1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceMean1(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceMean",allInputs, map))
}
}

trait ReduceMin12[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte : Numeric:ClassTag] extends Operator {
  def ReduceMin12(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceMin",allInputs, map))
}
}
trait ReduceMin11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte : Numeric:ClassTag] extends Operator {
  def ReduceMin11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceMin",allInputs, map))
}
}
trait ReduceMin1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double | UByte | Byte : Numeric:ClassTag] extends Operator {
  def ReduceMin1(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceMin",allInputs, map))
}
}

trait ReduceProd11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceProd11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceProd",allInputs, map))
}
}
trait ReduceProd1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceProd1(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceProd",allInputs, map))
}
}

trait ReduceSum11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceSum11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceSum",allInputs, map))
}
}
trait ReduceSum1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceSum1(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceSum",allInputs, map))
}
}

trait ReduceSumSquare11[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceSumSquare11(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceSumSquare",allInputs, map))
}
}
trait ReduceSumSquare1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ReduceSumSquare1(name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"keepdims" -> keepdims 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"ReduceSumSquare",allInputs, map))
}
}

trait Relu6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Relu6(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Relu",allInputs, map))
}
}
trait Relu1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Relu1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Relu",allInputs, map))
}
}

trait Reshape5[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Reshape5(name: String,data: Tensor[T], shapeInput: Tensor[Long])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data,shapeInput *: () )
(callOp[Tensor[T]](name,"Reshape",allInputs, map))
}
}
trait Reshape1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Reshape1(name: String,consumed_inputs : Option[(Array[Int])] = None,shape : Option[(Array[Int])] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
,"shape" -> shape 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Reshape",allInputs, map))
}
}

trait Resize11[@sp T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T2 <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Resize11(name: String,coordinate_transformation_mode : Option[(String)] = None,cubic_coeff_a : Option[(Float)] = None,exclude_outside : Option[(Int)] = None,extrapolation_value : Option[(Float)] = None,mode : Option[(String)] = None,nearest_mode : Option[(String)] = None,X: Tensor[T1], roi: Tensor[T2], scales: Tensor[Float],sizes: Option[Tensor[Long]] = None)
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("coordinate_transformation_mode" -> coordinate_transformation_mode 
,"cubic_coeff_a" -> cubic_coeff_a 
,"exclude_outside" -> exclude_outside 
,"extrapolation_value" -> extrapolation_value 
,"mode" -> mode 
,"nearest_mode" -> nearest_mode 
)
val allInputs = Some(X,roi,scales,sizes *: () )
(callOp[Tensor[T1]](name,"Resize",allInputs, map))
}
}
trait Resize10[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Resize10(name: String,mode : Option[(String)] = None,X: Tensor[T], scales: Tensor[Float])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("mode" -> mode 
)
val allInputs = Some(X,scales *: () )
(callOp[Tensor[T]](name,"Resize",allInputs, map))
}
}

trait ReverseSequence10[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def ReverseSequence10(name: String,batch_axis : Option[(Int)] = None,time_axis : Option[(Int)] = None,input: Tensor[T], sequence_lens: Tensor[Long])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("batch_axis" -> batch_axis 
,"time_axis" -> time_axis 
)
val allInputs = Some(input,sequence_lens *: () )
(callOp[Tensor[T]](name,"ReverseSequence",allInputs, map))
}
}

trait RoiAlign10[@sp T1 <: Float16 | Float | Double : Numeric:ClassTag,@sp T2 <: Long : Numeric:ClassTag] extends Operator {
  def RoiAlign10(name: String,mode : Option[(String)] = None,output_height : Option[(Int)] = None,output_width : Option[(Int)] = None,sampling_ratio : Option[(Int)] = None,spatial_scaleAttr : Option[(Float)] = None,X: Tensor[T1], rois: Tensor[T1], batch_indices: Tensor[T2])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("mode" -> mode 
,"output_height" -> output_height 
,"output_width" -> output_width 
,"sampling_ratio" -> sampling_ratio 
,"spatial_scaleAttr" -> spatial_scaleAttr 
)
val allInputs = Some(X,rois,batch_indices *: () )
(callOp[Tensor[T1]](name,"RoiAlign",allInputs, map))
}
}

trait Round11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Round11(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Round",allInputs, map))
}
}

trait SVMClassifier1[@sp T1 <: Float | Double | Long | Int : Numeric:ClassTag,@sp T2 <: String | Long : Numeric:ClassTag] extends Operator {
  def SVMClassifier1(name: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,post_transform : Option[(String)] = None,prob_a : Option[(Array[Float])] = None,prob_b : Option[(Array[Float])] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None,vectors_per_class : Option[(Array[Int])] = None,X: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("classlabels_ints" -> classlabels_ints 
,"classlabels_strings" -> classlabels_strings 
,"coefficients" -> coefficients 
,"kernel_params" -> kernel_params 
,"kernel_type" -> kernel_type 
,"post_transform" -> post_transform 
,"prob_a" -> prob_a 
,"prob_b" -> prob_b 
,"rho" -> rho 
,"support_vectors" -> support_vectors 
,"vectors_per_class" -> vectors_per_class 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"SVMClassifier",allInputs, map))
}
}

trait SVMRegressor1[@sp T <: Float | Double | Long | Int : Numeric:ClassTag] extends Operator {
  def SVMRegressor1(name: String,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,n_supports : Option[(Int)] = None,one_class : Option[(Int)] = None,post_transform : Option[(String)] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None,X: Tensor[T])
    : Tuple1[Tensor[Float]]
 = {
val map: Map[String, Any] = Map("coefficients" -> coefficients 
,"kernel_params" -> kernel_params 
,"kernel_type" -> kernel_type 
,"n_supports" -> n_supports 
,"one_class" -> one_class 
,"post_transform" -> post_transform 
,"rho" -> rho 
,"support_vectors" -> support_vectors 
)
val allInputs = Some(X *: () )
(callOp[Tensor[Float]](name,"SVMRegressor",allInputs, map))
}
}

trait Scaler1[@sp T <: Float | Double | Long | Int : Numeric:ClassTag] extends Operator {
  def Scaler1(name: String,offset : Option[(Array[Float])] = None,scaleAttr : Option[(Array[Float])] = None,X: Tensor[T])
    : Tuple1[Tensor[Float]]
 = {
val map: Map[String, Any] = Map("offset" -> offset 
,"scaleAttr" -> scaleAttr 
)
val allInputs = Some(X *: () )
(callOp[Tensor[Float]](name,"Scaler",allInputs, map))
}
}

trait Scan11[@sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Scan11(name: String,body : (Graph),num_scan_inputs : (Int),scan_input_axes : Option[(Array[Int])] = None,scan_input_directions : Option[(Array[Int])] = None,scan_output_axes : Option[(Array[Int])] = None,scan_output_directions : Option[(Array[Int])] = None,initial_state_and_scan_inputs: Seq[Tensor[V]])
    : Tuple1[Tensor[V]]
 = {
val map: Map[String, Any] = Map("body" -> body 
,"num_scan_inputs" -> num_scan_inputs 
,"scan_input_axes" -> scan_input_axes 
,"scan_input_directions" -> scan_input_directions 
,"scan_output_axes" -> scan_output_axes 
,"scan_output_directions" -> scan_output_directions 
)
val allInputs = Some(initial_state_and_scan_inputs(0),initial_state_and_scan_inputs(1),initial_state_and_scan_inputs(2),initial_state_and_scan_inputs(3),initial_state_and_scan_inputs(4),initial_state_and_scan_inputs(5),initial_state_and_scan_inputs(6),initial_state_and_scan_inputs(7),initial_state_and_scan_inputs(8) *: () )
(callOp[Tensor[V]](name,"Scan",allInputs, map))
}
}
trait Scan9[@sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Scan9(name: String,body : (Graph),num_scan_inputs : (Int),scan_input_axes : Option[(Array[Int])] = None,scan_input_directions : Option[(Array[Int])] = None,scan_output_axes : Option[(Array[Int])] = None,scan_output_directions : Option[(Array[Int])] = None,initial_state_and_scan_inputs: Seq[Tensor[V]])
    : Tuple1[Tensor[V]]
 = {
val map: Map[String, Any] = Map("body" -> body 
,"num_scan_inputs" -> num_scan_inputs 
,"scan_input_axes" -> scan_input_axes 
,"scan_input_directions" -> scan_input_directions 
,"scan_output_axes" -> scan_output_axes 
,"scan_output_directions" -> scan_output_directions 
)
val allInputs = Some(initial_state_and_scan_inputs(0),initial_state_and_scan_inputs(1),initial_state_and_scan_inputs(2),initial_state_and_scan_inputs(3),initial_state_and_scan_inputs(4),initial_state_and_scan_inputs(5),initial_state_and_scan_inputs(6),initial_state_and_scan_inputs(7),initial_state_and_scan_inputs(8) *: () )
(callOp[Tensor[V]](name,"Scan",allInputs, map))
}
}
trait Scan8[@sp I <: Long : Numeric:ClassTag,@sp V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Scan8(name: String,body : (Graph),directions : Option[(Array[Int])] = None,num_scan_inputs : (Int),sequence_lens: Option[Tensor[I]] = None,initial_state_and_scan_inputs: Seq[Tensor[V]])
    : Tuple1[Tensor[V]]
 = {
val map: Map[String, Any] = Map("body" -> body 
,"directions" -> directions 
,"num_scan_inputs" -> num_scan_inputs 
)
val allInputs = Some(sequence_lens,initial_state_and_scan_inputs(0),initial_state_and_scan_inputs(1),initial_state_and_scan_inputs(2),initial_state_and_scan_inputs(3),initial_state_and_scan_inputs(4),initial_state_and_scan_inputs(5),initial_state_and_scan_inputs(6),initial_state_and_scan_inputs(7) *: () )
(callOp[Tensor[V]](name,"Scan",allInputs, map))
}
}

trait Scatter11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def Scatter11(name: String,axis : Option[(Int)] = None,data: Tensor[T], indices: Tensor[Tind], updates: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(data,indices,updates *: () )
(callOp[Tensor[T]](name,"Scatter",allInputs, map))
}
}
trait Scatter9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def Scatter9(name: String,axis : Option[(Int)] = None,data: Tensor[T], indices: Tensor[Tind], updates: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(data,indices,updates *: () )
(callOp[Tensor[T]](name,"Scatter",allInputs, map))
}
}

trait ScatterElements11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def ScatterElements11(name: String,axis : Option[(Int)] = None,data: Tensor[T], indices: Tensor[Tind], updates: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(data,indices,updates *: () )
(callOp[Tensor[T]](name,"ScatterElements",allInputs, map))
}
}

trait ScatterND11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def ScatterND11(name: String,data: Tensor[T], indices: Tensor[Long], updates: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data,indices,updates *: () )
(callOp[Tensor[T]](name,"ScatterND",allInputs, map))
}
}

trait Selu6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Selu6(name: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"gamma" -> gamma 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Selu",allInputs, map))
}
}
trait Selu1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Selu1(name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,gamma : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
,"consumed_inputs" -> consumed_inputs 
,"gamma" -> gamma 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Selu",allInputs, map))
}
}

trait SequenceAt11[@sp S <: Seq[Tensor[UByte]] | Seq[Tensor[UShort]] | Seq[Tensor[UInt]] | Seq[Tensor[ULong]] | Seq[Tensor[Byte]] | Seq[Tensor[Short]] | Seq[Tensor[Int]] | Seq[Tensor[Long]] | Seq[Tensor[Float16]] | Seq[Tensor[Float]] | Seq[Tensor[Double]] | Seq[Tensor[String]] | Seq[Tensor[Boolean]] | Seq[Tensor[Complex[Float]]] | Seq[Tensor[Complex[Double]]] : Numeric:ClassTag,@sp I <: Int | Long : Numeric:ClassTag,@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def SequenceAt11(name: String,input_sequence: S, position: Tensor[I])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input_sequence,position *: () )
(callOp[Tensor[T]](name,"SequenceAt",allInputs, map))
}
}

trait SequenceConstruct11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp S <: Seq[Tensor[UByte]] | Seq[Tensor[UShort]] | Seq[Tensor[UInt]] | Seq[Tensor[ULong]] | Seq[Tensor[Byte]] | Seq[Tensor[Short]] | Seq[Tensor[Int]] | Seq[Tensor[Long]] | Seq[Tensor[Float16]] | Seq[Tensor[Float]] | Seq[Tensor[Double]] | Seq[Tensor[String]] | Seq[Tensor[Boolean]] | Seq[Tensor[Complex[Float]]] | Seq[Tensor[Complex[Double]]] : Numeric:ClassTag] extends Operator {
  def SequenceConstruct11(name: String,inputs: Seq[Tensor[T]])
    : Tuple1[S]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(inputs(0),inputs(1),inputs(2),inputs(3),inputs(4),inputs(5),inputs(6),inputs(7),inputs(8) *: () )
(callOp[S](name,"SequenceConstruct",allInputs, map))
}
}

trait SequenceEmpty11[@sp S <: Seq[Tensor[UByte]] | Seq[Tensor[UShort]] | Seq[Tensor[UInt]] | Seq[Tensor[ULong]] | Seq[Tensor[Byte]] | Seq[Tensor[Short]] | Seq[Tensor[Int]] | Seq[Tensor[Long]] | Seq[Tensor[Float16]] | Seq[Tensor[Float]] | Seq[Tensor[Double]] | Seq[Tensor[String]] | Seq[Tensor[Boolean]] | Seq[Tensor[Complex[Float]]] | Seq[Tensor[Complex[Double]]] : Numeric:ClassTag] extends Operator {
  def SequenceEmpty11(name: String,dtype : Option[(Int)] = None)
    : Tuple1[S]
 = {
val map: Map[String, Any] = Map("dtype" -> dtype 
)
val allInputs = None
(callOp[S](name,"SequenceEmpty",allInputs, map))
}
}

trait SequenceErase11[@sp S <: Seq[Tensor[UByte]] | Seq[Tensor[UShort]] | Seq[Tensor[UInt]] | Seq[Tensor[ULong]] | Seq[Tensor[Byte]] | Seq[Tensor[Short]] | Seq[Tensor[Int]] | Seq[Tensor[Long]] | Seq[Tensor[Float16]] | Seq[Tensor[Float]] | Seq[Tensor[Double]] | Seq[Tensor[String]] | Seq[Tensor[Boolean]] | Seq[Tensor[Complex[Float]]] | Seq[Tensor[Complex[Double]]] : Numeric:ClassTag,@sp I <: Int | Long : Numeric:ClassTag] extends Operator {
  def SequenceErase11(name: String,input_sequence: S,position: Option[Tensor[I]] = None)
    : Tuple1[S]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input_sequence,position *: () )
(callOp[S](name,"SequenceErase",allInputs, map))
}
}

trait SequenceInsert11[@sp S <: Seq[Tensor[UByte]] | Seq[Tensor[UShort]] | Seq[Tensor[UInt]] | Seq[Tensor[ULong]] | Seq[Tensor[Byte]] | Seq[Tensor[Short]] | Seq[Tensor[Int]] | Seq[Tensor[Long]] | Seq[Tensor[Float16]] | Seq[Tensor[Float]] | Seq[Tensor[Double]] | Seq[Tensor[String]] | Seq[Tensor[Boolean]] | Seq[Tensor[Complex[Float]]] | Seq[Tensor[Complex[Double]]] : Numeric:ClassTag,@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp I <: Int | Long : Numeric:ClassTag] extends Operator {
  def SequenceInsert11(name: String,input_sequence: S, tensor: Tensor[T],position: Option[Tensor[I]] = None)
    : Tuple1[S]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input_sequence,tensor,position *: () )
(callOp[S](name,"SequenceInsert",allInputs, map))
}
}

trait SequenceLength11[@sp S <: Seq[Tensor[UByte]] | Seq[Tensor[UShort]] | Seq[Tensor[UInt]] | Seq[Tensor[ULong]] | Seq[Tensor[Byte]] | Seq[Tensor[Short]] | Seq[Tensor[Int]] | Seq[Tensor[Long]] | Seq[Tensor[Float16]] | Seq[Tensor[Float]] | Seq[Tensor[Double]] | Seq[Tensor[String]] | Seq[Tensor[Boolean]] | Seq[Tensor[Complex[Float]]] | Seq[Tensor[Complex[Double]]] : Numeric:ClassTag,@sp I <: Long : Numeric:ClassTag] extends Operator {
  def SequenceLength11(name: String,input_sequence: S)
    : Tuple1[Tensor[I]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input_sequence *: () )
(callOp[Tensor[I]](name,"SequenceLength",allInputs, map))
}
}

trait Shape1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T1 <: Long : Numeric:ClassTag] extends Operator {
  def Shape1(name: String,data: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data *: () )
(callOp[Tensor[T1]](name,"Shape",allInputs, map))
}
}

trait Shrink9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Shrink9(name: String,bias : Option[(Float)] = None,lambd : Option[(Float)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("bias" -> bias 
,"lambd" -> lambd 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Shrink",allInputs, map))
}
}

trait Sigmoid6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sigmoid6(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Sigmoid",allInputs, map))
}
}
trait Sigmoid1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sigmoid1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Sigmoid",allInputs, map))
}
}

trait Sign9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sign9(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Sign",allInputs, map))
}
}

trait Sin7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sin7(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Sin",allInputs, map))
}
}

trait Sinh9[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sinh9(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Sinh",allInputs, map))
}
}

trait Size1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T1 <: Long : Numeric:ClassTag] extends Operator {
  def Size1(name: String,data: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data *: () )
(callOp[Tensor[T1]](name,"Size",allInputs, map))
}
}

trait Slice11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def Slice11(name: String,data: Tensor[T], starts: Tensor[Tind], ends: Tensor[Tind],axes: Option[Tensor[Tind]] = None, steps: Option[Tensor[Tind]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data,starts,ends,axes,steps *: () )
(callOp[Tensor[T]](name,"Slice",allInputs, map))
}
}
trait Slice10[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def Slice10(name: String,data: Tensor[T], starts: Tensor[Tind], ends: Tensor[Tind],axes: Option[Tensor[Tind]] = None, steps: Option[Tensor[Tind]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data,starts,ends,axes,steps *: () )
(callOp[Tensor[T]](name,"Slice",allInputs, map))
}
}
trait Slice1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Slice1(name: String,axes : Option[(Array[Int])] = None,ends : (Array[Int]),starts : (Array[Int]),data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
,"ends" -> ends 
,"starts" -> starts 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Slice",allInputs, map))
}
}

trait Softmax11[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Softmax11(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Softmax",allInputs, map))
}
}
trait Softmax1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Softmax1(name: String,axis : Option[(Int)] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Softmax",allInputs, map))
}
}

trait SoftmaxCrossEntropyLoss12[@sp T <: Float16 | Float | Double : Numeric:ClassTag,@sp Tind <: Int | Long : Numeric:ClassTag] extends Operator {
  def SoftmaxCrossEntropyLoss12(name: String,reduction : Option[(String)] = None,scores: Tensor[T], labels: Tensor[Tind],weights: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("reduction" -> reduction 
)
val allInputs = Some(scores,labels,weights *: () )
(callOp[Tensor[T]](name,"SoftmaxCrossEntropyLoss",allInputs, map))
}
}

trait Softplus1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Softplus1(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Softplus",allInputs, map))
}
}

trait Softsign1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Softsign1(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Softsign",allInputs, map))
}
}

trait SpaceToDepth1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def SpaceToDepth1(name: String,blocksize : (Int),input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("blocksize" -> blocksize 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"SpaceToDepth",allInputs, map))
}
}

trait Split11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Split11(name: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"splitAttr" -> splitAttr 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Split",allInputs, map))
}
}
trait Split2[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Split2(name: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"splitAttr" -> splitAttr 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Split",allInputs, map))
}
}
trait Split1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Split1(name: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None,input: Tensor[T],split: Option[Tensor[T]] = None)
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"splitAttr" -> splitAttr 
)
val allInputs = Some(input,split *: () )
(callOp[Tensor[T]](name,"Split",allInputs, map))
}
}

trait SplitToSequence11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp I <: Int | Long : Numeric:ClassTag,@sp S <: Seq[Tensor[UByte]] | Seq[Tensor[UShort]] | Seq[Tensor[UInt]] | Seq[Tensor[ULong]] | Seq[Tensor[Byte]] | Seq[Tensor[Short]] | Seq[Tensor[Int]] | Seq[Tensor[Long]] | Seq[Tensor[Float16]] | Seq[Tensor[Float]] | Seq[Tensor[Double]] | Seq[Tensor[String]] | Seq[Tensor[Boolean]] | Seq[Tensor[Complex[Float]]] | Seq[Tensor[Complex[Double]]] : Numeric:ClassTag] extends Operator {
  def SplitToSequence11(name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,input: Tensor[T],split: Option[Tensor[I]] = None)
    : Tuple1[S]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"keepdims" -> keepdims 
)
val allInputs = Some(input,split *: () )
(callOp[S](name,"SplitToSequence",allInputs, map))
}
}

trait Sqrt6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sqrt6(name: String,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Sqrt",allInputs, map))
}
}
trait Sqrt1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sqrt1(name: String,consumed_inputs : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Sqrt",allInputs, map))
}
}

trait Squeeze11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Squeeze11(name: String,axes : Option[(Array[Int])] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Squeeze",allInputs, map))
}
}
trait Squeeze1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Squeeze1(name: String,axes : Option[(Array[Int])] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Squeeze",allInputs, map))
}
}

trait StringNormalizer10 extends Operator {
  def StringNormalizer10(name: String,case_change_action : Option[(String)] = None,is_case_sensitive : Option[(Int)] = None,locale : Option[(String)] = None,stopwords : Option[(Array[String])] = None,X: Tensor[String])
    : Tuple1[Tensor[String]]
 = {
val map: Map[String, Any] = Map("case_change_action" -> case_change_action 
,"is_case_sensitive" -> is_case_sensitive 
,"locale" -> locale 
,"stopwords" -> stopwords 
)
val allInputs = Some(X *: () )
(callOp[Tensor[String]](name,"StringNormalizer",allInputs, map))
}
}

trait Sub7[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sub7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Sub",allInputs, map))
}
}
trait Sub6[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sub6(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Sub",allInputs, map))
}
}
trait Sub1[@sp T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sub1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
,"consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T]](name,"Sub",allInputs, map))
}
}

trait Sum8[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sum8(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Sum",allInputs, map))
}
}
trait Sum6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sum6(name: String,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Sum",allInputs, map))
}
}
trait Sum1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Sum1(name: String,consumed_inputs : Option[(Array[Int])] = None,data_0: Seq[Tensor[T]])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(data_0(0),data_0(1),data_0(2),data_0(3),data_0(4),data_0(5),data_0(6),data_0(7),data_0(8) *: () )
(callOp[Tensor[T]](name,"Sum",allInputs, map))
}
}

trait Tan7[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Tan7(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Tan",allInputs, map))
}
}

trait Tanh6[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Tanh6(name: String,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Tanh",allInputs, map))
}
}
trait Tanh1[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def Tanh1(name: String,consumed_inputs : Option[(Array[Int])] = None,input: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("consumed_inputs" -> consumed_inputs 
)
val allInputs = Some(input *: () )
(callOp[Tensor[T]](name,"Tanh",allInputs, map))
}
}

trait TfIdfVectorizer9[@sp T <: String | Int | Long : Numeric:ClassTag,@sp T1 <: Float : Numeric:ClassTag] extends Operator {
  def TfIdfVectorizer9(name: String,max_gram_length : (Int),max_skip_count : (Int),min_gram_length : (Int),mode : (String),ngram_counts : (Array[Int]),ngram_indexes : (Array[Int]),pool_int64s : Option[(Array[Int])] = None,pool_strings : Option[(Array[String])] = None,weights : Option[(Array[Float])] = None,X: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("max_gram_length" -> max_gram_length 
,"max_skip_count" -> max_skip_count 
,"min_gram_length" -> min_gram_length 
,"mode" -> mode 
,"ngram_counts" -> ngram_counts 
,"ngram_indexes" -> ngram_indexes 
,"pool_int64s" -> pool_int64s 
,"pool_strings" -> pool_strings 
,"weights" -> weights 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T1]](name,"TfIdfVectorizer",allInputs, map))
}
}

trait ThresholdedRelu10[@sp T <: Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def ThresholdedRelu10(name: String,alpha : Option[(Float)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("alpha" -> alpha 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"ThresholdedRelu",allInputs, map))
}
}

trait Tile6[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag,@sp T1 <: Long : Numeric:ClassTag] extends Operator {
  def Tile6(name: String,input: Tensor[T], repeats: Tensor[T1])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input,repeats *: () )
(callOp[Tensor[T]](name,"Tile",allInputs, map))
}
}
trait Tile1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Tile1(name: String,input: Tensor[T], tiles: Tensor[T], axis: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(input,tiles,axis *: () )
(callOp[Tensor[T]](name,"Tile",allInputs, map))
}
}

trait TopK11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp I <: Long : Numeric:ClassTag] extends Operator {
  def TopK11(name: String,axis : Option[(Int)] = None,largest : Option[(Int)] = None,sorted : Option[(Int)] = None,X: Tensor[T], K: Tensor[Long])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"largest" -> largest 
,"sorted" -> sorted 
)
val allInputs = Some(X,K *: () )
(callOp[Tensor[T]](name,"TopK",allInputs, map))
}
}
trait TopK10[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp I <: Long : Numeric:ClassTag] extends Operator {
  def TopK10(name: String,axis : Option[(Int)] = None,X: Tensor[T], K: Tensor[Long])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
)
val allInputs = Some(X,K *: () )
(callOp[Tensor[T]](name,"TopK",allInputs, map))
}
}
trait TopK1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag,@sp I <: Long : Numeric:ClassTag] extends Operator {
  def TopK1(name: String,axis : Option[(Int)] = None,k : (Int),X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"k" -> k 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"TopK",allInputs, map))
}
}

trait Transpose1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Transpose1(name: String,perm : Option[(Array[Int])] = None,data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("perm" -> perm 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Transpose",allInputs, map))
}
}

trait TreeEnsembleClassifier1[@sp T1 <: Float | Double | Long | Int : Numeric:ClassTag,@sp T2 <: String | Long : Numeric:ClassTag] extends Operator {
  def TreeEnsembleClassifier1(name: String,base_values : Option[(Array[Float])] = None,class_ids : Option[(Array[Int])] = None,class_nodeids : Option[(Array[Int])] = None,class_treeids : Option[(Array[Int])] = None,class_weights : Option[(Array[Float])] = None,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,X: Tensor[T1])
    : Tuple1[Tensor[T2]]
 = {
val map: Map[String, Any] = Map("base_values" -> base_values 
,"class_ids" -> class_ids 
,"class_nodeids" -> class_nodeids 
,"class_treeids" -> class_treeids 
,"class_weights" -> class_weights 
,"classlabels_int64s" -> classlabels_int64s 
,"classlabels_strings" -> classlabels_strings 
,"nodes_falsenodeids" -> nodes_falsenodeids 
,"nodes_featureids" -> nodes_featureids 
,"nodes_hitrates" -> nodes_hitrates 
,"nodes_missing_value_tracks_true" -> nodes_missing_value_tracks_true 
,"nodes_modes" -> nodes_modes 
,"nodes_nodeids" -> nodes_nodeids 
,"nodes_treeids" -> nodes_treeids 
,"nodes_truenodeids" -> nodes_truenodeids 
,"nodes_values" -> nodes_values 
,"post_transform" -> post_transform 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T2]](name,"TreeEnsembleClassifier",allInputs, map))
}
}

trait TreeEnsembleRegressor1[@sp T <: Float | Double | Long | Int : Numeric:ClassTag] extends Operator {
  def TreeEnsembleRegressor1(name: String,aggregate_function : Option[(String)] = None,base_values : Option[(Array[Float])] = None,n_targets : Option[(Int)] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,target_ids : Option[(Array[Int])] = None,target_nodeids : Option[(Array[Int])] = None,target_treeids : Option[(Array[Int])] = None,target_weights : Option[(Array[Float])] = None,X: Tensor[T])
    : Tuple1[Tensor[Float]]
 = {
val map: Map[String, Any] = Map("aggregate_function" -> aggregate_function 
,"base_values" -> base_values 
,"n_targets" -> n_targets 
,"nodes_falsenodeids" -> nodes_falsenodeids 
,"nodes_featureids" -> nodes_featureids 
,"nodes_hitrates" -> nodes_hitrates 
,"nodes_missing_value_tracks_true" -> nodes_missing_value_tracks_true 
,"nodes_modes" -> nodes_modes 
,"nodes_nodeids" -> nodes_nodeids 
,"nodes_treeids" -> nodes_treeids 
,"nodes_truenodeids" -> nodes_truenodeids 
,"nodes_values" -> nodes_values 
,"post_transform" -> post_transform 
,"target_ids" -> target_ids 
,"target_nodeids" -> target_nodeids 
,"target_treeids" -> target_treeids 
,"target_weights" -> target_weights 
)
val allInputs = Some(X *: () )
(callOp[Tensor[Float]](name,"TreeEnsembleRegressor",allInputs, map))
}
}

trait UnfoldToDepth12[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag] extends Operator {
  def UnfoldToDepth12(name: String,block_size : (Array[Int]),dilations : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("block_size" -> block_size 
,"dilations" -> dilations 
,"pads" -> pads 
,"strides" -> strides 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"UnfoldToDepth",allInputs, map))
}
}

trait Unique11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Unique11(name: String,axis : Option[(Int)] = None,sorted : Option[(Int)] = None,X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"sorted" -> sorted 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Unique",allInputs, map))
}
}

trait Unsqueeze11[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Unsqueeze11(name: String,axes : (Array[Int]),data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Unsqueeze",allInputs, map))
}
}
trait Unsqueeze1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Unsqueeze1(name: String,axes : (Array[Int]),data: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("axes" -> axes 
)
val allInputs = Some(data *: () )
(callOp[Tensor[T]](name,"Unsqueeze",allInputs, map))
}
}

trait Upsample10[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Upsample10(name: String,mode : Option[(String)] = None,X: Tensor[T], scales: Tensor[Float])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("mode" -> mode 
)
val allInputs = Some(X,scales *: () )
(callOp[Tensor[T]](name,"Upsample",allInputs, map))
}
}
trait Upsample9[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Upsample9(name: String,mode : Option[(String)] = None,X: Tensor[T], scales: Tensor[Float])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("mode" -> mode 
)
val allInputs = Some(X,scales *: () )
(callOp[Tensor[T]](name,"Upsample",allInputs, map))
}
}
trait Upsample7[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Upsample7(name: String,mode : Option[(String)] = None,scaleAttrs : (Array[Float]),X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("mode" -> mode 
,"scaleAttrs" -> scaleAttrs 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Upsample",allInputs, map))
}
}
trait Upsample1[@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Upsample1(name: String,height_scaleAttr : (Float),mode : Option[(String)] = None,width_scaleAttr : (Float),X: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map("height_scaleAttr" -> height_scaleAttr 
,"mode" -> mode 
,"width_scaleAttr" -> width_scaleAttr 
)
val allInputs = Some(X *: () )
(callOp[Tensor[T]](name,"Upsample",allInputs, map))
}
}

trait Where9[@sp B <: Boolean : Numeric:ClassTag,@sp T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag] extends Operator {
  def Where9(name: String,condition: Tensor[B], X: Tensor[T], Y: Tensor[T])
    : Tuple1[Tensor[T]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(condition,X,Y *: () )
(callOp[Tensor[T]](name,"Where",allInputs, map))
}
}

trait Xor7[@sp T <: Boolean : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Xor7(name: String,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map()
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Xor",allInputs, map))
}
}
trait Xor1[@sp T <: Boolean : Numeric:ClassTag,@sp T1 <: Boolean : Numeric:ClassTag] extends Operator {
  def Xor1(name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Tensor[T], B: Tensor[T])
    : Tuple1[Tensor[T1]]
 = {
val map: Map[String, Any] = Map("axis" -> axis 
,"broadcast" -> broadcast 
)
val allInputs = Some(A,B *: () )
(callOp[Tensor[T1]](name,"Xor",allInputs, map))
}
}

trait ZipMap1[@sp T <: Seq[Map[String, Float]] | Seq[Map[Long, Float]] : Numeric:ClassTag] extends Operator {
  def ZipMap1(name: String,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,X: Tensor[Float])
    : Tuple1[T]
 = {
val map: Map[String, Any] = Map("classlabels_int64s" -> classlabels_int64s 
,"classlabels_strings" -> classlabels_strings 
)
val allInputs = Some(X *: () )
(callOp[T](name,"ZipMap",allInputs, map))
}
}
}
