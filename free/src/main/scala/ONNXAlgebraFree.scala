package org.emergentorder

import freestyle.free._
import freestyle.free.implicits._
import cats.free.Free
import cats.free.FreeApplicative
import cats.effect.IO
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
import singleton.ops._

package object onnxFree {

    import UnionType._
    @free trait DataSourceFree extends DataSource {
  def inputDataFree[T : Numeric:ClassTag:Field, J <: XInt](implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): FS[Tensor[T, J]]
  def getParamsFree[T : Numeric:ClassTag:Field, J <: XInt](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): FS[Tensor[T, J]]
  def getAttributesFree[T : Numeric:ClassTag:Field, J <: XInt](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): FS[Tensor[T, J]]
}
@free trait CeilFree extends Operator with Ceil {

  def Ceil1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Ceil6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait FeatureVectorizerFree extends Operator with FeatureVectorizer {

  def FeatureVectorizer1Free[J <:XInt](name: String)
    : FS[(Tensor[Float, J])]

}
@free trait ConstantFree extends Operator with Constant {

  def Constant1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ArgMinFree extends Operator with ArgMin {

  def ArgMin1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Long, J])]

}
@free trait RNNFree extends Operator with RNN {

  def RNN1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T, J], Tensor[T, J])]


  def RNN7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T, J], Tensor[T, J])]

}
@free trait DropoutFree extends Operator with Dropout {

  def Dropout1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J], Tensor[T, J])]


  def Dropout6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J], Tensor[T, J])]


  def Dropout7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J], Tensor[T, J])]

}
@free trait ArrayFeatureExtractorFree extends Operator with ArrayFeatureExtractor {

  def ArrayFeatureExtractor1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[Long, J], Yname: String)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int TypeOr String)#check[T])    : FS[(Tensor[T, J])]

}
@free trait DivFree extends Operator with Div {

  def Div1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Div6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Div7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait AffineFree extends Operator with Affine {

  def Affine1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait CosFree extends Operator with Cos {

  def Cos7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait AbsFree extends Operator with Abs {

  def Abs1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Abs6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait EluFree extends Operator with Elu {

  def Elu1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Elu6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReduceProdFree extends Operator with ReduceProd {

  def ReduceProd1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait LeakyReluFree extends Operator with LeakyRelu {

  def LeakyRelu1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def LeakyRelu6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait LogFree extends Operator with Log {

  def Log1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Log6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait PadFree extends Operator with Pad {

  def Pad1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(String)] = None,paddings : (Seq[Int]),value : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Pad2Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(String)] = None,pads : (Seq[Int]),value : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SplitFree extends Operator with Split {

  def Split1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,split: Option[Tensor[T, J]] = None,axis : Option[(Int)] = None,splitAttr : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Split2Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None,splitAttr : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait NegFree extends Operator with Neg {

  def Neg1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Neg6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait IfFree extends Operator with If {

  def If1Free[B : Numeric:ClassTag:Field,V : Numeric:ClassTag:Field, J <: XInt](name: String,cond: Tensor[B, J], condname: String,else_branch : (Graph),then_branch : (Graph))
(implicit evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[V])    : FS[(Tensor[V, J])]

}
@free trait TanhFree extends Operator with Tanh {

  def Tanh1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Tanh6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReduceSumFree extends Operator with ReduceSum {

  def ReduceSum1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait DictVectorizerFree extends Operator with DictVectorizer {

  def DictVectorizer1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: T1, Xname: String,int64_vocabulary : Option[(Seq[Int])] = None,string_vocabulary : Option[(Seq[String])] = None)
(implicit evT1:(UNil TypeOr Map[String, Long] TypeOr Map[Long, String] TypeOr Map[Long, Float] TypeOr Map[Long, Double] TypeOr Map[String, Float] TypeOr Map[String, Double])#check[T1],evT2:(UNil TypeOr Long TypeOr Float TypeOr Double TypeOr String)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait GRUFree extends Operator with GRU {

  def GRU1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T, J], Tensor[T, J])]


  def GRU3Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T, J], Tensor[T, J])]


  def GRU7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T, J], Tensor[T, J])]

}
@free trait HardmaxFree extends Operator with Hardmax {

  def Hardmax1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ThresholdedReluFree extends Operator with ThresholdedRelu {

  def ThresholdedRelu1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SoftplusFree extends Operator with Softplus {

  def Softplus1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait GreaterFree extends Operator with Greater {

  def Greater1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]


  def Greater7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]

}
@free trait RandomNormalFree extends Operator with RandomNormal {

  def RandomNormal1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait LSTMFree extends Operator with LSTM {

  def LSTM1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def LSTM7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J])]

}
@free trait SeluFree extends Operator with Selu {

  def Selu1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None,gamma : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Selu6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait GivenTensorFillFree extends Operator with GivenTensorFill {

  def GivenTensorFill1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,shapeInput: Option[Tensor[T, J]] = None,extra_shape : Option[(Seq[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Seq[Int])] = None,values : Option[(Seq[Float])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait NotFree extends Operator with Not {

  def Not1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReduceMinFree extends Operator with ReduceMin {

  def ReduceMin1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReduceSumSquareFree extends Operator with ReduceSumSquare {

  def ReduceSumSquare1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait MeanFree extends Operator with Mean {

  def Mean1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Mean6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait TileFree extends Operator with Tile {

  def Tile1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, tiles: Tensor[T, J], tilesname: String, axis: Tensor[T, J], axisname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Tile6Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, repeats: Tensor[T1, J], repeatsname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T, J])]

}
@free trait LogSoftmaxFree extends Operator with LogSoftmax {

  def LogSoftmax1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait LpNormalizationFree extends Operator with LpNormalization {

  def LpNormalization1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None,p : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ImageScalerFree extends Operator with ImageScaler {

  def ImageScaler1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,bias : Option[(Seq[Float])] = None,scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SinFree extends Operator with Sin {

  def Sin7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait RandomNormalLikeFree extends Operator with RandomNormalLike {

  def RandomNormalLike1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait XorFree extends Operator with Xor {

  def Xor1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]


  def Xor7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]

}
@free trait ReduceLogSumFree extends Operator with ReduceLogSum {

  def ReduceLogSum1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait CategoryMapperFree extends Operator with CategoryMapper {

  def CategoryMapper1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,cats_int64s : Option[(Seq[Int])] = None,cats_strings : Option[(Seq[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait LoopIndexTensorFree extends Operator with LoopIndexTensor {

  def LoopIndexTensor1Free[T : Numeric:ClassTag:Field,I : Numeric:ClassTag:Field, J <: XInt](name: String,T: Tensor[T, J], Tname: String, loop_idx: I, loop_idxname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T],evI:(UNil TypeOr Int)#check[I])    : FS[(Tensor[T, J])]

}
@free trait AtanFree extends Operator with Atan {

  def Atan7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait AndFree extends Operator with And {

  def And1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]


  def And7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]

}
@free trait SVMClassifierFree extends Operator with SVMClassifier {

  def SVMClassifier1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classlabels_ints : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,coefficients : Option[(Seq[Float])] = None,kernel_params : Option[(Seq[Float])] = None,kernel_type : Option[(String)] = None,post_transform : Option[(String)] = None,prob_a : Option[(Seq[Float])] = None,prob_b : Option[(Seq[Float])] = None,rho : Option[(Seq[Float])] = None,support_vectors : Option[(Seq[Float])] = None,vectors_per_class : Option[(Seq[Int])] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2, J], Tensor[Float, J])]

}
@free trait ScalerFree extends Operator with Scaler {

  def Scaler1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,offset : Option[(Seq[Float])] = None,scaleAttr : Option[(Seq[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float, J])]

}
@free trait LoopFree extends Operator with Loop {

  def Loop1Free[I : Numeric:ClassTag:Field,B : Numeric:ClassTag:Field,V : Numeric:ClassTag:Field, J <: XInt](name: String,M: I, Mname: String, cond: B, condname: String,body : (Graph))
(implicit evI:(UNil TypeOr Long)#check[I],evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[V])    : FS[(Tensor[V, J])]

}
@free trait LessFree extends Operator with Less {

  def Less1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]


  def Less7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]

}
@free trait SoftsignFree extends Operator with Softsign {

  def Softsign1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait AcosFree extends Operator with Acos {

  def Acos7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait MulFree extends Operator with Mul {

  def Mul1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Mul6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Mul7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ConstantFillFree extends Operator with ConstantFill {

  def ConstantFill1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Option[Tensor[T1, J]] = None,dtype : Option[(Int)] = None,extra_shape : Option[(Seq[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Seq[Int])] = None,value : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Int TypeOr Long TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float TypeOr Int TypeOr Long TypeOr Boolean)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait FloorFree extends Operator with Floor {

  def Floor1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Floor6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait DepthToSpaceFree extends Operator with DepthToSpace {

  def DepthToSpace1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ClipFree extends Operator with Clip {

  def Clip1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Clip6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,max : Option[(Float)] = None,min : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SqueezeFree extends Operator with Squeeze {

  def Squeeze1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : (Seq[Int]))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SizeFree extends Operator with Size {

  def Size1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr Boolean)#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T1, J])]

}
@free trait ImputerFree extends Operator with Imputer {

  def Imputer1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,imputed_value_floats : Option[(Seq[Float])] = None,imputed_value_int64s : Option[(Seq[Int])] = None,replaced_value_float : Option[(Float)] = None,replaced_value_int64 : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[T, J])]

}
@free trait EqualFree extends Operator with Equal {

  def Equal1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]


  def Equal7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]

}
@free trait TreeEnsembleRegressorFree extends Operator with TreeEnsembleRegressor {

  def TreeEnsembleRegressor1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,aggregate_function : Option[(String)] = None,base_values : Option[(Seq[Float])] = None,n_targets : Option[(Int)] = None,nodes_falsenodeids : Option[(Seq[Int])] = None,nodes_featureids : Option[(Seq[Int])] = None,nodes_hitrates : Option[(Seq[Float])] = None,nodes_missing_value_tracks_true : Option[(Seq[Int])] = None,nodes_modes : Option[(Seq[String])] = None,nodes_nodeids : Option[(Seq[Int])] = None,nodes_treeids : Option[(Seq[Int])] = None,nodes_truenodeids : Option[(Seq[Int])] = None,nodes_values : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None,target_ids : Option[(Seq[Int])] = None,target_nodeids : Option[(Seq[Int])] = None,target_treeids : Option[(Seq[Int])] = None,target_weights : Option[(Seq[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float, J])]

}
@free trait LabelEncoderFree extends Operator with LabelEncoder {

  def LabelEncoder1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classes_strings : Option[(Seq[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait ZipMapFree extends Operator with ZipMap {

  def ZipMap1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[Float, J], Xname: String,classlabels_int64s : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None)
(implicit evT:(UNil TypeOr Seq[Map[String, Float]] TypeOr Seq[Map[Long, Float]])#check[T])    : FS[(T)]

}
@free trait ReduceLogSumExpFree extends Operator with ReduceLogSumExp {

  def ReduceLogSumExp1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait AddFree extends Operator with Add {

  def Add1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Add6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Add7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait MaxPoolFree extends Operator with MaxPool {

  def MaxPool1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ConvTransposeFree extends Operator with ConvTranspose {

  def ConvTranspose1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Seq[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Seq[Int])] = None,output_padding : Option[(Seq[Int])] = None,output_shape : Option[(Seq[Int])] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait UpsampleFree extends Operator with Upsample {

  def Upsample1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,height_scaleAttr : (Float),mode : Option[(String)] = None,width_scaleAttr : (Float))
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T])    : FS[(Tensor[T, J])]


  def Upsample7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,mode : Option[(String)] = None,scaleAttrs : (Seq[Float]))
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T])    : FS[(Tensor[T, J])]

}
@free trait LinearRegressorFree extends Operator with LinearRegressor {

  def LinearRegressor1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,coefficients : Option[(Seq[Float])] = None,intercepts : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None,targets : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float, J])]

}
@free trait LinearClassifierFree extends Operator with LinearClassifier {

  def LinearClassifier1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classlabels_ints : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,coefficients : (Seq[Float]),intercepts : Option[(Seq[Float])] = None,multi_class : Option[(Int)] = None,post_transform : Option[(String)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2, J], Tensor[Float, J])]

}
@free trait SpaceToDepthFree extends Operator with SpaceToDepth {

  def SpaceToDepth1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SoftmaxFree extends Operator with Softmax {

  def Softmax1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait GlobalMaxPoolFree extends Operator with GlobalMaxPool {

  def GlobalMaxPool1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReduceL2Free extends Operator with ReduceL2 {

  def ReduceL21Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ConvFree extends Operator with Conv {

  def Conv1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Seq[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Seq[Int])] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait MatMulFree extends Operator with MatMul {

  def MatMul1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ParametricSoftplusFree extends Operator with ParametricSoftplus {

  def ParametricSoftplus1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait InstanceNormalizationFree extends Operator with InstanceNormalization {

  def InstanceNormalization1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,consumed_inputs : Option[(Seq[Int])] = None,epsilon : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def InstanceNormalization6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,epsilon : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait MaxRoiPoolFree extends Operator with MaxRoiPool {

  def MaxRoiPool1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, rois: Tensor[T, J], roisname: String,pooled_shape : (Seq[Int]),spatial_scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReduceMeanFree extends Operator with ReduceMean {

  def ReduceMean1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait NormalizerFree extends Operator with Normalizer {

  def Normalizer1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,norm : Option[(String)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float, J])]

}
@free trait BinarizerFree extends Operator with Binarizer {

  def Binarizer1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,threshold : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SigmoidFree extends Operator with Sigmoid {

  def Sigmoid1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Sigmoid6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait TransposeFree extends Operator with Transpose {

  def Transpose1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,perm : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait PowFree extends Operator with Pow {

  def Pow1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Pow7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReduceL1Free extends Operator with ReduceL1 {

  def ReduceL11Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ConcatFree extends Operator with Concat {

  def Concat1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T])    : FS[(Tensor[T, J])]


  def Concat4Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T])    : FS[(Tensor[T, J])]

}
@free trait GlobalLpPoolFree extends Operator with GlobalLpPool {

  def GlobalLpPool1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def GlobalLpPool2Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait UnsqueezeFree extends Operator with Unsqueeze {

  def Unsqueeze1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : (Seq[Int]))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T])    : FS[(Tensor[T, J])]

}
@free trait MultinomialFree extends Operator with Multinomial {

  def Multinomial7Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Int TypeOr Long)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait SqrtFree extends Operator with Sqrt {

  def Sqrt1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Sqrt6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait GRUUnitFree extends Operator with GRUUnit {

  def GRUUnit1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,hidden_prev: Tensor[T, J], hidden_prevname: String, gates: Tensor[T, J], gatesname: String, seq_lengths: Tensor[T, J], seq_lengthsname: String, t: Tensor[T, J], tname: String,drop_states : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReduceMaxFree extends Operator with ReduceMax {

  def ReduceMax1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait MinFree extends Operator with Min {

  def Min1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Min6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait TanFree extends Operator with Tan {

  def Tan7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait MeanVarianceNormalizationFree extends Operator with MeanVarianceNormalization {

  def MeanVarianceNormalization1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,across_channels : Option[(Int)] = None,normalize_variance : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait CropFree extends Operator with Crop {

  def Crop1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,border : Option[(Seq[Int])] = None,scaleAttr : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait GemmFree extends Operator with Gemm {

  def Gemm1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Gemm6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Gemm7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait RandomUniformLikeFree extends Operator with RandomUniformLike {

  def RandomUniformLike1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait TreeEnsembleClassifierFree extends Operator with TreeEnsembleClassifier {

  def TreeEnsembleClassifier1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,base_values : Option[(Seq[Float])] = None,class_ids : Option[(Seq[Int])] = None,class_nodeids : Option[(Seq[Int])] = None,class_treeids : Option[(Seq[Int])] = None,class_weights : Option[(Seq[Float])] = None,classlabels_int64s : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,nodes_falsenodeids : Option[(Seq[Int])] = None,nodes_featureids : Option[(Seq[Int])] = None,nodes_hitrates : Option[(Seq[Float])] = None,nodes_missing_value_tracks_true : Option[(Seq[Int])] = None,nodes_modes : Option[(Seq[String])] = None,nodes_nodeids : Option[(Seq[Int])] = None,nodes_treeids : Option[(Seq[Int])] = None,nodes_truenodeids : Option[(Seq[Int])] = None,nodes_values : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2, J], Tensor[Float, J])]

}
@free trait SliceFree extends Operator with Slice {

  def Slice1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,ends : (Seq[Int]),starts : (Seq[Int]))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ShapeFree extends Operator with Shape {

  def Shape1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr Boolean)#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T1, J])]

}
@free trait SumFree extends Operator with Sum {

  def Sum1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Sum6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ExpFree extends Operator with Exp {

  def Exp1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Exp6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait PReluFree extends Operator with PRelu {

  def PRelu1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def PRelu6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def PRelu7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ScaleFree extends Operator with Scale {

  def Scale1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait LRNFree extends Operator with LRN {

  def LRN1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait OneHotEncoderFree extends Operator with OneHotEncoder {

  def OneHotEncoder1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,cats_int64s : Option[(Seq[Int])] = None,cats_strings : Option[(Seq[String])] = None,zeros : Option[(Int)] = None)
(implicit evT:(UNil TypeOr String TypeOr Long TypeOr Int TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Float, J])]

}
@free trait RandomUniformFree extends Operator with RandomUniform {

  def RandomUniform1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait BatchNormalizationFree extends Operator with BatchNormalization {

  def BatchNormalization1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,consumed_inputs : (Seq[Int]),epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def BatchNormalization6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def BatchNormalization7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]

}
@free trait AsinFree extends Operator with Asin {

  def Asin7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReshapeFree extends Operator with Reshape {

  def Reshape1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[Int])] = None,shape : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Reshape5Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, shape: Tensor[Long, J], shapename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait LpPoolFree extends Operator with LpPool {

  def LpPool1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Seq[Int])] = None,p : Option[(Float)] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def LpPool2Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),p : Option[(Int)] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait IdentityFree extends Operator with Identity {

  def Identity1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T])    : FS[(Tensor[T, J])]

}
@free trait GlobalAveragePoolFree extends Operator with GlobalAveragePool {

  def GlobalAveragePool1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait CastFree extends Operator with Cast {

  def Cast1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (String))
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : FS[(Tensor[T2, J])]


  def Cast6Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (Int))
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait OrFree extends Operator with Or {

  def Or1Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]


  def Or7Free[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1, J])]

}
@free trait FlattenFree extends Operator with Flatten {

  def Flatten1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SubFree extends Operator with Sub {

  def Sub1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Sub6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Sub7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReciprocalFree extends Operator with Reciprocal {

  def Reciprocal1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Reciprocal6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ReluFree extends Operator with Relu {

  def Relu1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Relu6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait ScaledTanhFree extends Operator with ScaledTanh {

  def ScaledTanh1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait AveragePoolFree extends Operator with AveragePool {

  def AveragePool1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def AveragePool7Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait CastMapFree extends Operator with CastMap {

  def CastMap1Free[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: T1, Xname: String,cast_to : Option[(String)] = None,map_form : Option[(String)] = None,max_map : Option[(Int)] = None)
(implicit evT1:(UNil TypeOr Map[Long, String] TypeOr Map[Long, Float])#check[T1],evT2:(UNil TypeOr String TypeOr Float TypeOr Long)#check[T2])    : FS[(Tensor[T2, J])]

}
@free trait ArgMaxFree extends Operator with ArgMax {

  def ArgMax1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Long, J])]

}
@free trait GatherFree extends Operator with Gather {

  def Gather1Free[T : Numeric:ClassTag:Field,Tind : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, indices: Tensor[Tind, J], indicesname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean)#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : FS[(Tensor[T, J])]

}
@free trait MaxFree extends Operator with Max {

  def Max1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def Max6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait SVMRegressorFree extends Operator with SVMRegressor {

  def SVMRegressor1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,coefficients : Option[(Seq[Float])] = None,kernel_params : Option[(Seq[Float])] = None,kernel_type : Option[(String)] = None,n_supports : Option[(Int)] = None,one_class : Option[(Int)] = None,post_transform : Option[(String)] = None,rho : Option[(Seq[Float])] = None,support_vectors : Option[(Seq[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float, J])]

}
@free trait HardSigmoidFree extends Operator with HardSigmoid {

  def HardSigmoid1Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]


  def HardSigmoid6Free[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T, J])]

}
@free trait TopKFree extends Operator with TopK {

  def TopK1Free[T : Numeric:ClassTag:Field,I : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,axis : Option[(Int)] = None,k : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : FS[(Tensor[T, J], Tensor[I, J])]

}}
