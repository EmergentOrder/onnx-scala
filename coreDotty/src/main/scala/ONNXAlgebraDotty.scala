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
  type Tensor[U, J <: XInt] = Tuple2[Seq[U], Seq[J]]
  trait Operator
trait Graph
trait DataSource {
  def inputData[T <: (Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt]: Tensor[T, J]
  def getParams[T <: (Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String): Tensor[T, J]
  def getAttributes[T <: (Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String): Tensor[T, J]
}
trait Abs extends Operator {

  def Abs1[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Abs6[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Acos extends Operator {

  def Acos7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Add extends Operator {

  def Add1[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Add6[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Add7[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Affine extends Operator {

  def Affine1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait And extends Operator {

  def And1[@sp T <: (Boolean):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def And7[@sp T <: (Boolean):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait ArgMax extends Operator {

  def ArgMax1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
    : (Tensor[Long, J])

}
trait ArgMin extends Operator {

  def ArgMin1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
    : (Tensor[Long, J])

}
trait ArrayFeatureExtractor extends Operator {

  def ArrayFeatureExtractor1[@sp T <: (Float | Double | Long | Int | String):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[Long, J], Yname: String)
    : (Tensor[T, J])

}
trait Asin extends Operator {

  def Asin7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Atan extends Operator {

  def Atan7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait AveragePool extends Operator {

  def AveragePool1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def AveragePool7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait BatchNormalization extends Operator {

  def BatchNormalization1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,consumed_inputs : (Seq[Int]),epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def BatchNormalization6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def BatchNormalization7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])

}
trait Binarizer extends Operator {

  def Binarizer1[@sp T <: (Float | Double | Long | Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,threshold : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Cast extends Operator {

  def Cast1[@sp T1 <: (Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean):Numeric:ClassTag:Field,@sp T2 <: (Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (String))
    : (Tensor[T2, J])


  def Cast6[@sp T1 <: (Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean):Numeric:ClassTag:Field,@sp T2 <: (Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (Int))
    : (Tensor[T2, J])

}
trait CastMap extends Operator {

  def CastMap1[@sp T1 <: (Map[Long, String] | Map[Long, Float]):Numeric:ClassTag:Field,@sp T2 <: (String | Float | Long):Numeric:ClassTag:Field, J <: XInt](name: String,X: T1, Xname: String,cast_to : Option[(String)] = None,map_form : Option[(String)] = None,max_map : Option[(Int)] = None)
    : (Tensor[T2, J])

}
trait CategoryMapper extends Operator {

  def CategoryMapper1[@sp T1 <: (String | Long):Numeric:ClassTag:Field,@sp T2 <: (String | Long):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,cats_int64s : Option[(Seq[Int])] = None,cats_strings : Option[(Seq[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
    : (Tensor[T2, J])

}
trait Ceil extends Operator {

  def Ceil1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Ceil6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Clip extends Operator {

  def Clip1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None)
    : (Tensor[T, J])


  def Clip6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,max : Option[(Float)] = None,min : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Concat extends Operator {

  def Concat1[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Concat4[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait Constant extends Operator {

  def Constant1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait ConstantFill extends Operator {

  def ConstantFill1[@sp T1 <: (Float | Int | Long | Boolean):Numeric:ClassTag:Field,@sp T2 <: (Float | Int | Long | Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,input: Option[Tensor[T1, J]] = None,dtype : Option[(Int)] = None,extra_shape : Option[(Seq[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Seq[Int])] = None,value : Option[(Float)] = None)
    : (Tensor[T2, J])

}
trait Conv extends Operator {

  def Conv1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Seq[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Seq[Int])] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait ConvTranspose extends Operator {

  def ConvTranspose1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Seq[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Seq[Int])] = None,output_padding : Option[(Seq[Int])] = None,output_shape : Option[(Seq[Int])] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait Cos extends Operator {

  def Cos7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Crop extends Operator {

  def Crop1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,border : Option[(Seq[Int])] = None,scaleAttr : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait DepthToSpace extends Operator {

  def DepthToSpace1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (Int))
    : (Tensor[T, J])

}
trait DictVectorizer extends Operator {

  def DictVectorizer1[@sp T1 <: (Map[String, Long] | Map[Long, String] | Map[Long, Float] | Map[Long, Double] | Map[String, Float] | Map[String, Double]):Numeric:ClassTag:Field,@sp T2 <: (Long | Float | Double | String):Numeric:ClassTag:Field, J <: XInt](name: String,X: T1, Xname: String,int64_vocabulary : Option[(Seq[Int])] = None,string_vocabulary : Option[(Seq[String])] = None)
    : (Tensor[T2, J])

}
trait Div extends Operator {

  def Div1[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Div6[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Div7[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Dropout extends Operator {

  def Dropout1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def Dropout6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def Dropout7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,ratio : Option[(Float)] = None)
    : (Tensor[T, J], Tensor[T, J])

}
trait Elu extends Operator {

  def Elu1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Elu6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Equal extends Operator {

  def Equal1[@sp T <: (Boolean | Int | Long):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Equal7[@sp T <: (Boolean | Int | Long):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait Exp extends Operator {

  def Exp1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Exp6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Expand extends Operator {

  def Expand8[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, shape: Tensor[Long, J], shapename: String)
    : (Tensor[T, J])

}
trait FeatureVectorizer extends Operator {

  def FeatureVectorizer1[J <:XInt](name: String)
    : (Tensor[Float, J])

}
trait Flatten extends Operator {

  def Flatten1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Floor extends Operator {

  def Floor1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Floor6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait GRU extends Operator {

  def GRU1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def GRU3[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def GRU7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])

}
trait GRUUnit extends Operator {

  def GRUUnit1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,hidden_prev: Tensor[T, J], hidden_prevname: String, gates: Tensor[T, J], gatesname: String, seq_lengths: Tensor[T, J], seq_lengthsname: String, t: Tensor[T, J], tname: String,drop_states : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Gather extends Operator {

  def Gather1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field,@sp Tind <: (Int | Long):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, indices: Tensor[Tind, J], indicesname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Gemm extends Operator {

  def Gemm1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
    : (Tensor[T, J])


  def Gemm6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
    : (Tensor[T, J])


  def Gemm7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait GivenTensorFill extends Operator {

  def GivenTensorFill1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,shapeInput: Option[Tensor[T, J]] = None,extra_shape : Option[(Seq[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Seq[Int])] = None,values : Option[(Seq[Float])] = None)
    : (Tensor[T, J])

}
trait GlobalAveragePool extends Operator {

  def GlobalAveragePool1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait GlobalLpPool extends Operator {

  def GlobalLpPool1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Float)] = None)
    : (Tensor[T, J])


  def GlobalLpPool2[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait GlobalMaxPool extends Operator {

  def GlobalMaxPool1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Greater extends Operator {

  def Greater1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Greater7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait HardSigmoid extends Operator {

  def HardSigmoid1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def HardSigmoid6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Hardmax extends Operator {

  def Hardmax1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Identity extends Operator {

  def Identity1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait If extends Operator {

  def If1[@sp B <: (Boolean):Numeric:ClassTag:Field,@sp V <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,cond: Tensor[B, J], condname: String,else_branch : (Graph),then_branch : (Graph))
    : (Tensor[V, J])

}
trait ImageScaler extends Operator {

  def ImageScaler1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,bias : Option[(Seq[Float])] = None,scaleAttr : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Imputer extends Operator {

  def Imputer1[@sp T <: (Float | Double | Long | Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,imputed_value_floats : Option[(Seq[Float])] = None,imputed_value_int64s : Option[(Seq[Int])] = None,replaced_value_float : Option[(Float)] = None,replaced_value_int64 : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait InstanceNormalization extends Operator {

  def InstanceNormalization1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,consumed_inputs : Option[(Seq[Int])] = None,epsilon : Option[(Float)] = None)
    : (Tensor[T, J])


  def InstanceNormalization6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,epsilon : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait LRN extends Operator {

  def LRN1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : (Int))
    : (Tensor[T, J])

}
trait LSTM extends Operator {

  def LSTM1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def LSTM7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J])

}
trait LabelEncoder extends Operator {

  def LabelEncoder1[@sp T1 <: (String | Long):Numeric:ClassTag:Field,@sp T2 <: (String | Long):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classes_strings : Option[(Seq[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
    : (Tensor[T2, J])

}
trait LeakyRelu extends Operator {

  def LeakyRelu1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def LeakyRelu6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Less extends Operator {

  def Less1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Less7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait LinearClassifier extends Operator {

  def LinearClassifier1[@sp T1 <: (Float | Double | Long | Int):Numeric:ClassTag:Field,@sp T2 <: (String | Long):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classlabels_ints : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,coefficients : (Seq[Float]),intercepts : Option[(Seq[Float])] = None,multi_class : Option[(Int)] = None,post_transform : Option[(String)] = None)
    : (Tensor[T2, J], Tensor[Float, J])

}
trait LinearRegressor extends Operator {

  def LinearRegressor1[@sp T <: (Float | Double | Long | Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,coefficients : Option[(Seq[Float])] = None,intercepts : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None,targets : Option[(Int)] = None)
    : (Tensor[Float, J])

}
trait Log extends Operator {

  def Log1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Log6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait LogSoftmax extends Operator {

  def LogSoftmax1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Loop extends Operator {

  def Loop1[@sp I <: (Long):Numeric:ClassTag:Field,@sp B <: (Boolean):Numeric:ClassTag:Field,@sp V <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,M: I, Mname: String, cond: B, condname: String,body : (Graph))
    : (Tensor[V, J])

}
trait LpNormalization extends Operator {

  def LpNormalization1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None,p : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait LpPool extends Operator {

  def LpPool1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Seq[Int])] = None,p : Option[(Float)] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def LpPool2[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),p : Option[(Int)] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait MatMul extends Operator {

  def MatMul1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Max extends Operator {

  def Max1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Max6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Max8[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait MaxPool extends Operator {

  def MaxPool1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def MaxPool8[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp I <: (Long):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J], Tensor[I, J])

}
trait MaxRoiPool extends Operator {

  def MaxRoiPool1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, rois: Tensor[T, J], roisname: String,pooled_shape : (Seq[Int]),spatial_scaleAttr : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Mean extends Operator {

  def Mean1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Mean6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Mean8[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,across_channels : Option[(Int)] = None,normalize_variance : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Min extends Operator {

  def Min1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Min6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Min8[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait Mul extends Operator {

  def Mul1[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Mul6[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Mul7[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Multinomial extends Operator {

  def Multinomial7[@sp T1 <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T2 <: (Int | Long):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None)
    : (Tensor[T2, J])

}
trait Neg extends Operator {

  def Neg1[@sp T <: (Float16 | Float | Double | Float | Int | Byte | Short | Long | Float16 | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Neg6[@sp T <: (Float16 | Float | Double | Float | Int | Byte | Short | Long | Float16 | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Normalizer extends Operator {

  def Normalizer1[@sp T <: (Float | Double | Long | Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,norm : Option[(String)] = None)
    : (Tensor[Float, J])

}
trait Not extends Operator {

  def Not1[@sp T <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait OneHotEncoder extends Operator {

  def OneHotEncoder1[@sp T <: (String | Long | Int | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,cats_int64s : Option[(Seq[Int])] = None,cats_strings : Option[(Seq[String])] = None,zeros : Option[(Int)] = None)
    : (Tensor[Float, J])

}
trait Or extends Operator {

  def Or1[@sp T <: (Boolean):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Or7[@sp T <: (Boolean):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait PRelu extends Operator {

  def PRelu1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def PRelu6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
    : (Tensor[T, J])


  def PRelu7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
    : (Tensor[T, J])

}
trait Pad extends Operator {

  def Pad1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(String)] = None,paddings : (Seq[Int]),value : Option[(Float)] = None)
    : (Tensor[T, J])


  def Pad2[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(String)] = None,pads : (Seq[Int]),value : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait ParametricSoftplus extends Operator {

  def ParametricSoftplus1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Pow extends Operator {

  def Pow1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Pow7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String)
    : (Tensor[T, J])

}
trait RNN extends Operator {

  def RNN1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def RNN7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp T1 <: (Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])

}
trait RandomNormal extends Operator {

  def RandomNormal1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait RandomNormalLike extends Operator {

  def RandomNormalLike1[@sp T1 <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field,@sp T2 <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None)
    : (Tensor[T2, J])

}
trait RandomUniform extends Operator {

  def RandomUniform1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait RandomUniformLike extends Operator {

  def RandomUniformLike1[@sp T1 <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field,@sp T2 <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None)
    : (Tensor[T2, J])

}
trait Reciprocal extends Operator {

  def Reciprocal1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Reciprocal6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait ReduceL1 extends Operator {

  def ReduceL11[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceL2 extends Operator {

  def ReduceL21[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceLogSum extends Operator {

  def ReduceLogSum1[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp1[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceMax extends Operator {

  def ReduceMax1[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceMean extends Operator {

  def ReduceMean1[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceMin extends Operator {

  def ReduceMin1[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceProd extends Operator {

  def ReduceProd1[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceSum extends Operator {

  def ReduceSum1[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait ReduceSumSquare extends Operator {

  def ReduceSumSquare1[@sp T <: (UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Relu extends Operator {

  def Relu1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Relu6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Reshape extends Operator {

  def Reshape1[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[Int])] = None,shape : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Reshape5[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, shape: Tensor[Long, J], shapename: String)
    : (Tensor[T, J])

}
trait SVMClassifier extends Operator {

  def SVMClassifier1[@sp T1 <: (Float | Double | Long | Int):Numeric:ClassTag:Field,@sp T2 <: (String | Long):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classlabels_ints : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,coefficients : Option[(Seq[Float])] = None,kernel_params : Option[(Seq[Float])] = None,kernel_type : Option[(String)] = None,post_transform : Option[(String)] = None,prob_a : Option[(Seq[Float])] = None,prob_b : Option[(Seq[Float])] = None,rho : Option[(Seq[Float])] = None,support_vectors : Option[(Seq[Float])] = None,vectors_per_class : Option[(Seq[Int])] = None)
    : (Tensor[T2, J], Tensor[Float, J])

}
trait SVMRegressor extends Operator {

  def SVMRegressor1[@sp T <: (Float | Double | Long | Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,coefficients : Option[(Seq[Float])] = None,kernel_params : Option[(Seq[Float])] = None,kernel_type : Option[(String)] = None,n_supports : Option[(Int)] = None,one_class : Option[(Int)] = None,post_transform : Option[(String)] = None,rho : Option[(Seq[Float])] = None,support_vectors : Option[(Seq[Float])] = None)
    : (Tensor[Float, J])

}
trait Scale extends Operator {

  def Scale1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,scaleAttr : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait ScaledTanh extends Operator {

  def ScaledTanh1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Scaler extends Operator {

  def Scaler1[@sp T <: (Float | Double | Long | Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,offset : Option[(Seq[Float])] = None,scaleAttr : Option[(Seq[Float])] = None)
    : (Tensor[Float, J])

}
trait Scan extends Operator {

  def Scan8[@sp I <: (Long):Numeric:ClassTag:Field,@sp V <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,sequence_lens: Option[Tensor[I, J]] = None,body : (Graph),directions : Option[(Seq[Int])] = None,num_scan_inputs : (Int))
    : (Tensor[V, J])

}
trait Selu extends Operator {

  def Selu1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None,gamma : Option[(Float)] = None)
    : (Tensor[T, J])


  def Selu6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Shape extends Operator {

  def Shape1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field,@sp T1 <: (Long):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
    : (Tensor[T1, J])

}
trait Sigmoid extends Operator {

  def Sigmoid1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Sigmoid6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Sin extends Operator {

  def Sin7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Size extends Operator {

  def Size1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field,@sp T1 <: (Long):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
    : (Tensor[T1, J])

}
trait Slice extends Operator {

  def Slice1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,ends : (Seq[Int]),starts : (Seq[Int]))
    : (Tensor[T, J])

}
trait Softmax extends Operator {

  def Softmax1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Softplus extends Operator {

  def Softplus1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Softsign extends Operator {

  def Softsign1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait SpaceToDepth extends Operator {

  def SpaceToDepth1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (Int))
    : (Tensor[T, J])

}
trait Split extends Operator {

  def Split1[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,split: Option[Tensor[T, J]] = None,axis : Option[(Int)] = None,splitAttr : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Split2[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None,splitAttr : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait Sqrt extends Operator {

  def Sqrt1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Sqrt6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Squeeze extends Operator {

  def Squeeze1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait Sub extends Operator {

  def Sub1[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Sub6[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Sub7[@sp T <: (Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Sum extends Operator {

  def Sum1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Sum6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Sum8[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait Tan extends Operator {

  def Tan7[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Tanh extends Operator {

  def Tanh1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Tanh6[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait ThresholdedRelu extends Operator {

  def ThresholdedRelu1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Tile extends Operator {

  def Tile1[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, tiles: Tensor[T, J], tilesname: String, axis: Tensor[T, J], axisname: String)
    : (Tensor[T, J])


  def Tile6[@sp T <: (Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field,@sp T1 <: (Long):Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, repeats: Tensor[T1, J], repeatsname: String)
    : (Tensor[T, J])

}
trait TopK extends Operator {

  def TopK1[@sp T <: (Float16 | Float | Double):Numeric:ClassTag:Field,@sp I <: (Long):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,axis : Option[(Int)] = None,k : (Int))
    : (Tensor[T, J], Tensor[I, J])

}
trait Transpose extends Operator {

  def Transpose1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,perm : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait TreeEnsembleClassifier extends Operator {

  def TreeEnsembleClassifier1[@sp T1 <: (Float | Double | Long | Int):Numeric:ClassTag:Field,@sp T2 <: (String | Long):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,base_values : Option[(Seq[Float])] = None,class_ids : Option[(Seq[Int])] = None,class_nodeids : Option[(Seq[Int])] = None,class_treeids : Option[(Seq[Int])] = None,class_weights : Option[(Seq[Float])] = None,classlabels_int64s : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,nodes_falsenodeids : Option[(Seq[Int])] = None,nodes_featureids : Option[(Seq[Int])] = None,nodes_hitrates : Option[(Seq[Float])] = None,nodes_missing_value_tracks_true : Option[(Seq[Int])] = None,nodes_modes : Option[(Seq[String])] = None,nodes_nodeids : Option[(Seq[Int])] = None,nodes_treeids : Option[(Seq[Int])] = None,nodes_truenodeids : Option[(Seq[Int])] = None,nodes_values : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None)
    : (Tensor[T2, J], Tensor[Float, J])

}
trait TreeEnsembleRegressor extends Operator {

  def TreeEnsembleRegressor1[@sp T <: (Float | Double | Long | Int):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,aggregate_function : Option[(String)] = None,base_values : Option[(Seq[Float])] = None,n_targets : Option[(Int)] = None,nodes_falsenodeids : Option[(Seq[Int])] = None,nodes_featureids : Option[(Seq[Int])] = None,nodes_hitrates : Option[(Seq[Float])] = None,nodes_missing_value_tracks_true : Option[(Seq[Int])] = None,nodes_modes : Option[(Seq[String])] = None,nodes_nodeids : Option[(Seq[Int])] = None,nodes_treeids : Option[(Seq[Int])] = None,nodes_truenodeids : Option[(Seq[Int])] = None,nodes_values : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None,target_ids : Option[(Seq[Int])] = None,target_nodeids : Option[(Seq[Int])] = None,target_treeids : Option[(Seq[Int])] = None,target_weights : Option[(Seq[Float])] = None)
    : (Tensor[Float, J])

}
trait Unsqueeze extends Operator {

  def Unsqueeze1[@sp T <: (UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : (Seq[Int]))
    : (Tensor[T, J])

}
trait Upsample extends Operator {

  def Upsample1[@sp T <: (Boolean | Int | Long | Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,height_scaleAttr : (Float),mode : Option[(String)] = None,width_scaleAttr : (Float))
    : (Tensor[T, J])


  def Upsample7[@sp T <: (Boolean | Int | Long | Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double]):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,mode : Option[(String)] = None,scaleAttrs : (Seq[Float]))
    : (Tensor[T, J])

}
trait Xor extends Operator {

  def Xor1[@sp T <: (Boolean):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Xor7[@sp T <: (Boolean):Numeric:ClassTag:Field,@sp T1 <: (Boolean):Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait ZipMap extends Operator {

  def ZipMap1[@sp T <: (Seq[Map[String, Float]] | Seq[Map[Long, Float]]):Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[Float, J], Xname: String,classlabels_int64s : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None)
    : (T)

}}
