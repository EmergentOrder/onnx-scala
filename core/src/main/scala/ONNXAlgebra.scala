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
import singleton.ops._

package object onnx {
type |:[+A1, +A2] = Either[A1, A2]
  type Tensor[U] = Tuple2[Array[U],  Array[Int with Singleton]]
  trait Operator
trait Graph
object UnionType {

      trait inv[-A] {}

      sealed trait OrR {
        type L <: OrR
        type R
        type invIntersect
        type intersect
      }

      sealed class TypeOr[A <: OrR, B] extends OrR {
        type L = A
        type R = B

        type intersect = (L#intersect with R)
        type invIntersect = (L#invIntersect with inv[R])
        type check[X] = invIntersect <:< inv[X]
      }

      object UNil extends OrR {
        type intersect = Any
        type invIntersect = inv[Nothing]
      }
      type UNil = UNil.type

    }
    
    import UnionType._
    trait DataSource {
  def inputData[T : Numeric:ClassTag:Field](implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Tensor[T]
  def getParams[T : Numeric:ClassTag:Field](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Tensor[T]
  def getAttributes[T : Numeric:ClassTag:Field](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Tensor[T]
}
trait Abs extends Operator {

  def Abs1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Abs6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Acos extends Operator {

  def Acos7[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Add extends Operator {

  def Add1[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Add6[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Add7[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Affine extends Operator {

  def Affine1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait And extends Operator {

  def And1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])


  def And7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])

}
trait ArgMax extends Operator {

  def ArgMax1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[Long])

}
trait ArgMin extends Operator {

  def ArgMin1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[Long])

}
trait ArrayFeatureExtractor extends Operator {

  def ArrayFeatureExtractor1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, Y: Tensor[Long], Yname: String)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int TypeOr String)#check[T])    : (Tensor[T])

}
trait Asin extends Operator {

  def Asin7[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Atan extends Operator {

  def Atan7[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait AveragePool extends Operator {

  def AveragePool1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def AveragePool7[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait BatchNormalization extends Operator {

  def BatchNormalization1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,consumed_inputs : (Array[Int]),epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])


  def BatchNormalization6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])


  def BatchNormalization7[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])

}
trait Binarizer extends Operator {

  def Binarizer1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,threshold : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[T])

}
trait Cast extends Operator {

  def Cast1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,to : (String))
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : (Tensor[T2])


  def Cast6[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,to : (Int))
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : (Tensor[T2])

}
trait CastMap extends Operator {

  def CastMap1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: T1, Xname: String,cast_to : Option[(String)] = None,map_form : Option[(String)] = None,max_map : Option[(Int)] = None)
(implicit evT1:(UNil TypeOr Map[Long, String] TypeOr Map[Long, Float])#check[T1],evT2:(UNil TypeOr String TypeOr Float TypeOr Long)#check[T2])    : (Tensor[T2])

}
trait CategoryMapper extends Operator {

  def CategoryMapper1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2])

}
trait Ceil extends Operator {

  def Ceil1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Ceil6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Clip extends Operator {

  def Clip1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Array[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Clip6[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,max : Option[(Float)] = None,min : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Concat extends Operator {

  def Concat1[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])


  def Concat4[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait Constant extends Operator {

  def Constant1[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ConstantFill extends Operator {

  def ConstantFill1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Option[Tensor[T1]] = None,dtype : Option[(Int)] = None,extra_shape : Option[(Array[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Array[Int])] = None,value : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Int TypeOr Long TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float TypeOr Int TypeOr Long TypeOr Boolean)#check[T2])    : (Tensor[T2])

}
trait Conv extends Operator {

  def Conv1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String,B: Option[Tensor[T]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ConvTranspose extends Operator {

  def ConvTranspose1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String,B: Option[Tensor[T]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,output_padding : Option[(Array[Int])] = None,output_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Cos extends Operator {

  def Cos7[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Crop extends Operator {

  def Crop1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,border : Option[(Array[Int])] = None,scaleAttr : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait DepthToSpace extends Operator {

  def DepthToSpace1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,blocksize : (Int))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait DictVectorizer extends Operator {

  def DictVectorizer1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: T1, Xname: String,int64_vocabulary : Option[(Array[Int])] = None,string_vocabulary : Option[(Array[String])] = None)
(implicit evT1:(UNil TypeOr Map[String, Long] TypeOr Map[Long, String] TypeOr Map[Long, Float] TypeOr Map[Long, Double] TypeOr Map[String, Float] TypeOr Map[String, Double])#check[T1],evT2:(UNil TypeOr Long TypeOr Float TypeOr Double TypeOr String)#check[T2])    : (Tensor[T2])

}
trait Div extends Operator {

  def Div1[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Div6[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Div7[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Dropout extends Operator {

  def Dropout1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,consumed_inputs : Option[(Array[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T], Tensor[T])


  def Dropout6[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T], Tensor[T])


  def Dropout7[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T], Tensor[T])

}
trait Elu extends Operator {

  def Elu1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Elu6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Equal extends Operator {

  def Equal1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])


  def Equal7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])

}
trait Exp extends Operator {

  def Exp1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Exp6[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Expand extends Operator {

  def Expand8[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, shape: Tensor[Long], shapename: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait FeatureVectorizer extends Operator {

  def FeatureVectorizer1(name: String)
    : (Tensor[Float])

}
trait Flatten extends Operator {

  def Flatten1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Floor extends Operator {

  def Floor1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Floor6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait GRU extends Operator {

  def GRU1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T], Tensor[T])


  def GRU3[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T], Tensor[T])


  def GRU7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T], Tensor[T])

}
trait GRUUnit extends Operator {

  def GRUUnit1[@sp T : Numeric:ClassTag:Field](name: String,hidden_prev: Tensor[T], hidden_prevname: String, gates: Tensor[T], gatesname: String, seq_lengths: Tensor[T], seq_lengthsname: String, t: Tensor[T], tname: String,drop_states : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Gather extends Operator {

  def Gather1[@sp T : Numeric:ClassTag:Field,@sp Tind : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String, indices: Tensor[Tind], indicesname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : (Tensor[T])

}
trait Gemm extends Operator {

  def Gemm1[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Gemm6[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Gemm7[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait GivenTensorFill extends Operator {

  def GivenTensorFill1[@sp T : Numeric:ClassTag:Field](name: String,shapeInput: Option[Tensor[T]] = None,extra_shape : Option[(Array[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Array[Int])] = None,values : Option[(Array[Float])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait GlobalAveragePool extends Operator {

  def GlobalAveragePool1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait GlobalLpPool extends Operator {

  def GlobalLpPool1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,p : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def GlobalLpPool2[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,p : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait GlobalMaxPool extends Operator {

  def GlobalMaxPool1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Greater extends Operator {

  def Greater1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])


  def Greater7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])

}
trait HardSigmoid extends Operator {

  def HardSigmoid1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def HardSigmoid6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Hardmax extends Operator {

  def Hardmax1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Identity extends Operator {

  def Identity1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait If extends Operator {

  def If1[@sp B : Numeric:ClassTag:Field,@sp V : Numeric:ClassTag:Field](name: String,cond: Tensor[B], condname: String,else_branch : (Graph),then_branch : (Graph))
(implicit evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : (Tensor[V])

}
trait ImageScaler extends Operator {

  def ImageScaler1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,bias : Option[(Array[Float])] = None,scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Imputer extends Operator {

  def Imputer1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,imputed_value_floats : Option[(Array[Float])] = None,imputed_value_int64s : Option[(Array[Int])] = None,replaced_value_float : Option[(Float)] = None,replaced_value_int64 : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[T])

}
trait InstanceNormalization extends Operator {

  def InstanceNormalization1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String,consumed_inputs : Option[(Array[Int])] = None,epsilon : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def InstanceNormalization6[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String,epsilon : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait LRN extends Operator {

  def LRN1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait LSTM extends Operator {

  def LSTM1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T], Tensor[T], Tensor[T])


  def LSTM7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T], Tensor[T], Tensor[T])

}
trait LabelEncoder extends Operator {

  def LabelEncoder1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,classes_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2])

}
trait LeakyRelu extends Operator {

  def LeakyRelu1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def LeakyRelu6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Less extends Operator {

  def Less1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])


  def Less7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])

}
trait LinearClassifier extends Operator {

  def LinearClassifier1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : (Array[Float]),intercepts : Option[(Array[Float])] = None,multi_class : Option[(Int)] = None,post_transform : Option[(String)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2], Tensor[Float])

}
trait LinearRegressor extends Operator {

  def LinearRegressor1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,coefficients : Option[(Array[Float])] = None,intercepts : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,targets : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float])

}
trait Log extends Operator {

  def Log1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Log6[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait LogSoftmax extends Operator {

  def LogSoftmax1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Loop extends Operator {

  def Loop1[@sp I : Numeric:ClassTag:Field,@sp B : Numeric:ClassTag:Field,@sp V : Numeric:ClassTag:Field](name: String,M: I, Mname: String, cond: B, condname: String,body : (Graph))
(implicit evI:(UNil TypeOr Long)#check[I],evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : (Tensor[V])

}
trait LpNormalization extends Operator {

  def LpNormalization1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None,p : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait LpPool extends Operator {

  def LpPool1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])] = None,p : Option[(Float)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def LpPool2[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),p : Option[(Int)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait MatMul extends Operator {

  def MatMul1[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Max extends Operator {

  def Max1[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Max6[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Max8[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait MaxPool extends Operator {

  def MaxPool1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def MaxPool8[@sp T : Numeric:ClassTag:Field,@sp I : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : (Tensor[T], Tensor[I])

}
trait MaxRoiPool extends Operator {

  def MaxRoiPool1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, rois: Tensor[T], roisname: String,pooled_shape : (Array[Int]),spatial_scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Mean extends Operator {

  def Mean1[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Mean6[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Mean8[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,across_channels : Option[(Int)] = None,normalize_variance : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Min extends Operator {

  def Min1[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Min6[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Min8[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Mul extends Operator {

  def Mul1[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Mul6[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Mul7[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Multinomial extends Operator {

  def Multinomial7[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Int TypeOr Long)#check[T2])    : (Tensor[T2])

}
trait Neg extends Operator {

  def Neg1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : (Tensor[T])


  def Neg6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : (Tensor[T])

}
trait Normalizer extends Operator {

  def Normalizer1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,norm : Option[(String)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float])

}
trait Not extends Operator {

  def Not1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T])    : (Tensor[T])

}
trait OneHotEncoder extends Operator {

  def OneHotEncoder1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,zeros : Option[(Int)] = None)
(implicit evT:(UNil TypeOr String TypeOr Long TypeOr Int TypeOr Float TypeOr Double)#check[T])    : (Tensor[Float])

}
trait Or extends Operator {

  def Or1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])


  def Or7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])

}
trait PRelu extends Operator {

  def PRelu1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def PRelu6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def PRelu7[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Pad extends Operator {

  def Pad1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,mode : Option[(String)] = None,paddings : (Array[Int]),value : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Pad2[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,mode : Option[(String)] = None,pads : (Array[Int]),value : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ParametricSoftplus extends Operator {

  def ParametricSoftplus1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Pow extends Operator {

  def Pow1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, Y: Tensor[T], Yname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Pow7[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, Y: Tensor[T], Yname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait RNN extends Operator {

  def RNN1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T], Tensor[T])


  def RNN7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T], Tensor[T])

}
trait RandomNormal extends Operator {

  def RandomNormal1[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait RandomNormalLike extends Operator {

  def RandomNormalLike1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : (Tensor[T2])

}
trait RandomUniform extends Operator {

  def RandomUniform1[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait RandomUniformLike extends Operator {

  def RandomUniformLike1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : (Tensor[T2])

}
trait Reciprocal extends Operator {

  def Reciprocal1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Reciprocal6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceL1 extends Operator {

  def ReduceL11[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceL2 extends Operator {

  def ReduceL21[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceLogSum extends Operator {

  def ReduceLogSum1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceMax extends Operator {

  def ReduceMax1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceMean extends Operator {

  def ReduceMean1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceMin extends Operator {

  def ReduceMin1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceProd extends Operator {

  def ReduceProd1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceSum extends Operator {

  def ReduceSum1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ReduceSumSquare extends Operator {

  def ReduceSumSquare1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Relu extends Operator {

  def Relu1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Relu6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Reshape extends Operator {

  def Reshape1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,consumed_inputs : Option[(Array[Int])] = None,shape : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])


  def Reshape5[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String, shape: Tensor[Long], shapename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait SVMClassifier extends Operator {

  def SVMClassifier1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,post_transform : Option[(String)] = None,prob_a : Option[(Array[Float])] = None,prob_b : Option[(Array[Float])] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None,vectors_per_class : Option[(Array[Int])] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2], Tensor[Float])

}
trait SVMRegressor extends Operator {

  def SVMRegressor1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,n_supports : Option[(Int)] = None,one_class : Option[(Int)] = None,post_transform : Option[(String)] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float])

}
trait Scale extends Operator {

  def Scale1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ScaledTanh extends Operator {

  def ScaledTanh1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Scaler extends Operator {

  def Scaler1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,offset : Option[(Array[Float])] = None,scaleAttr : Option[(Array[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float])

}
trait Scan extends Operator {

  def Scan8[@sp I : Numeric:ClassTag:Field,@sp V : Numeric:ClassTag:Field](name: String,sequence_lens: Option[Tensor[I]] = None,body : (Graph),directions : Option[(Array[Int])] = None,num_scan_inputs : (Int))
(implicit evI:(UNil TypeOr Long)#check[I],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : (Tensor[V])

}
trait Selu extends Operator {

  def Selu1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,gamma : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Selu6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Shape extends Operator {

  def Shape1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : (Tensor[T1])

}
trait Sigmoid extends Operator {

  def Sigmoid1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Sigmoid6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Sin extends Operator {

  def Sin7[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Size extends Operator {

  def Size1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : (Tensor[T1])

}
trait Slice extends Operator {

  def Slice1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,ends : (Array[Int]),starts : (Array[Int]))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait Softmax extends Operator {

  def Softmax1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Softplus extends Operator {

  def Softplus1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Softsign extends Operator {

  def Softsign1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait SpaceToDepth extends Operator {

  def SpaceToDepth1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,blocksize : (Int))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait Split extends Operator {

  def Split1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,split: Option[Tensor[T]] = None,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])


  def Split2[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait Sqrt extends Operator {

  def Sqrt1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Sqrt6[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Squeeze extends Operator {

  def Squeeze1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait Sub extends Operator {

  def Sub1[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Sub6[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Sub7[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Sum extends Operator {

  def Sum1[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Sum6[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Sum8[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Tan extends Operator {

  def Tan7[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Tanh extends Operator {

  def Tanh1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])


  def Tanh6[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait ThresholdedRelu extends Operator {

  def ThresholdedRelu1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T])

}
trait Tile extends Operator {

  def Tile1[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, tiles: Tensor[T], tilesname: String, axis: Tensor[T], axisname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])


  def Tile6[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, repeats: Tensor[T1], repeatsname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : (Tensor[T])

}
trait TopK extends Operator {

  def TopK1[@sp T : Numeric:ClassTag:Field,@sp I : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,axis : Option[(Int)] = None,k : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : (Tensor[T], Tensor[I])

}
trait Transpose extends Operator {

  def Transpose1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,perm : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait TreeEnsembleClassifier extends Operator {

  def TreeEnsembleClassifier1[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,base_values : Option[(Array[Float])] = None,class_ids : Option[(Array[Int])] = None,class_nodeids : Option[(Array[Int])] = None,class_treeids : Option[(Array[Int])] = None,class_weights : Option[(Array[Float])] = None,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2], Tensor[Float])

}
trait TreeEnsembleRegressor extends Operator {

  def TreeEnsembleRegressor1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,aggregate_function : Option[(String)] = None,base_values : Option[(Array[Float])] = None,n_targets : Option[(Int)] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,target_ids : Option[(Array[Int])] = None,target_nodeids : Option[(Array[Int])] = None,target_treeids : Option[(Array[Int])] = None,target_weights : Option[(Array[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float])

}
trait Unsqueeze extends Operator {

  def Unsqueeze1[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : (Array[Int]))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait Upsample extends Operator {

  def Upsample1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,height_scaleAttr : (Float),mode : Option[(String)] = None,width_scaleAttr : (Float))
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])


  def Upsample7[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,mode : Option[(String)] = None,scaleAttrs : (Array[Float]))
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T])

}
trait Xor extends Operator {

  def Xor1[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])


  def Xor7[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1])

}
trait ZipMap extends Operator {

  def ZipMap1[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[Float], Xname: String,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None)
(implicit evT:(UNil TypeOr Seq[Map[String, Float]] TypeOr Seq[Map[Long, Float]])#check[T])    : (T)

}}
