package org.emergentorder

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
import singleton.ops._

package object onnx {
type |:[+A1, +A2] = Either[A1, A2]
  type Tensor[U, J <: XInt] = Tuple2[Vector[U], Seq[J]]
type F[B] = IO[B]
type Par[F[_], A] = FreeApplicative[F, A]
final type FS[A] = Par[F, A]
  trait Operator
trait Graph
trait DataSource {
  def inputData[T : Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Complex[Float] | Complex[Double]:Numeric:ClassTag:Field, J <: XInt]: Tensor[T, J]
  def getParams[T : Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Complex[Float] | Complex[Double]:Numeric:ClassTag:Field, J <: XInt](name: String): Tensor[T, J]
  def getAttributes[T : Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Complex[Float] | Complex[Double]:Numeric:ClassTag:Field, J <: XInt](name: String): Tensor[T, J]
}
trait Acos extends Operator {

  def Acos7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait ReduceLogSum extends Operator {

  def ReduceLogSum1[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Atan extends Operator {

  def Atan7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait ReduceProd extends Operator {

  def ReduceProd1[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Elu extends Operator {

  def Elu1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Elu6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait LpNormalization extends Operator {

  def LpNormalization1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None,p : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Min extends Operator {

  def Min1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Min6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Min8[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait Selu extends Operator {

  def Selu1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None,gamma : Option[(Float)] = None)
    : (Tensor[T, J])


  def Selu6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait ReduceMean extends Operator {

  def ReduceMean1[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Softmax extends Operator {

  def Softmax1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Pow extends Operator {

  def Pow1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Pow7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String)
    : (Tensor[T, J])

}
trait Loop extends Operator {

  def Loop1[I : Long : Numeric:ClassTag:Field,B : Boolean : Numeric:ClassTag:Field,V : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,M: I, Mname: String, cond: B, condname: String,body : (Graph))
    : (Tensor[V, J])

}
trait Gemm extends Operator {

  def Gemm1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
    : (Tensor[T, J])


  def Gemm6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
    : (Tensor[T, J])


  def Gemm7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Relu extends Operator {

  def Relu1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Relu6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait TopK extends Operator {

  def TopK1[T : Float16 | Float | Double : Numeric:ClassTag:Field,I : Long : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,axis : Option[(Int)] = None,k : (Int))
    : (Tensor[T, J], Tensor[I, J])

}
trait ParametricSoftplus extends Operator {

  def ParametricSoftplus1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait RandomUniformLike extends Operator {

  def RandomUniformLike1[T1 : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T2 : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None)
    : (Tensor[T2, J])

}
trait Pad extends Operator {

  def Pad1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(String)] = None,paddings : (Seq[Int]),value : Option[(Float)] = None)
    : (Tensor[T, J])


  def Pad2[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(String)] = None,pads : (Seq[Int]),value : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Max extends Operator {

  def Max1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Max6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Max8[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait ConstantFill extends Operator {

  def ConstantFill1[T1 : Float | Int | Long | Boolean : Numeric:ClassTag:Field,T2 : Float | Int | Long | Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Option[Tensor[T1, J]] = None,dtype : Option[(Int)] = None,extra_shape : Option[(Seq[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Seq[Int])] = None,value : Option[(Float)] = None)
    : (Tensor[T2, J])

}
trait MaxRoiPool extends Operator {

  def MaxRoiPool1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, rois: Tensor[T, J], roisname: String,pooled_shape : (Seq[Int]),spatial_scaleAttr : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Reshape extends Operator {

  def Reshape1[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[Int])] = None,shape : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Reshape5[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, shape: Tensor[Long, J], shapename: String)
    : (Tensor[T, J])

}
trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp1[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Cos extends Operator {

  def Cos7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Log extends Operator {

  def Log1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Log6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,across_channels : Option[(Int)] = None,normalize_variance : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Constant extends Operator {

  def Constant1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait Concat extends Operator {

  def Concat1[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Concat4[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait Size extends Operator {

  def Size1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T1 : Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
    : (Tensor[T1, J])

}
trait Cast extends Operator {

  def Cast1[T1 : Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag:Field,T2 : Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (String))
    : (Tensor[T2, J])


  def Cast6[T1 : Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag:Field,T2 : Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (Int))
    : (Tensor[T2, J])

}
trait GlobalAveragePool extends Operator {

  def GlobalAveragePool1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Tan extends Operator {

  def Tan7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Less extends Operator {

  def Less1[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Less7[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait LRN extends Operator {

  def LRN1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : (Int))
    : (Tensor[T, J])

}
trait Crop extends Operator {

  def Crop1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,border : Option[(Seq[Int])] = None,scaleAttr : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait Exp extends Operator {

  def Exp1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Exp6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait HardSigmoid extends Operator {

  def HardSigmoid1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def HardSigmoid6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait AveragePool extends Operator {

  def AveragePool1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def AveragePool7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait ImageScaler extends Operator {

  def ImageScaler1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,bias : Option[(Seq[Float])] = None,scaleAttr : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Floor extends Operator {

  def Floor1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Floor6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait GlobalMaxPool extends Operator {

  def GlobalMaxPool1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait RandomUniform extends Operator {

  def RandomUniform1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait Unsqueeze extends Operator {

  def Unsqueeze1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : (Seq[Int]))
    : (Tensor[T, J])

}
trait Multinomial extends Operator {

  def Multinomial7[T1 : Float16 | Float | Double : Numeric:ClassTag:Field,T2 : Int | Long : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None)
    : (Tensor[T2, J])

}
trait MaxPool extends Operator {

  def MaxPool1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def MaxPool8[T : Float16 | Float | Double : Numeric:ClassTag:Field,I : Long : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J], Tensor[I, J])

}
trait LpPool extends Operator {

  def LpPool1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Seq[Int])] = None,p : Option[(Float)] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def LpPool2[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),p : Option[(Int)] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait Identity extends Operator {

  def Identity1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Tile extends Operator {

  def Tile1[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, tiles: Tensor[T, J], tilesname: String, axis: Tensor[T, J], axisname: String)
    : (Tensor[T, J])


  def Tile6[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T1 : Long : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, repeats: Tensor[T1, J], repeatsname: String)
    : (Tensor[T, J])

}
trait ScaledTanh extends Operator {

  def ScaledTanh1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait RNN extends Operator {

  def RNN1[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def RNN7[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])

}
trait Slice extends Operator {

  def Slice1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,ends : (Seq[Int]),starts : (Seq[Int]))
    : (Tensor[T, J])

}
trait ReduceSumSquare extends Operator {

  def ReduceSumSquare1[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Greater extends Operator {

  def Greater1[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Greater7[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait And extends Operator {

  def And1[T : Boolean : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def And7[T : Boolean : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait Shape extends Operator {

  def Shape1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T1 : Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
    : (Tensor[T1, J])

}
trait Sum extends Operator {

  def Sum1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Sum6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Sum8[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait LeakyRelu extends Operator {

  def LeakyRelu1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def LeakyRelu6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Ceil extends Operator {

  def Ceil1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Ceil6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait ReduceMax extends Operator {

  def ReduceMax1[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Scale extends Operator {

  def Scale1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,scaleAttr : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Or extends Operator {

  def Or1[T : Boolean : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Or7[T : Boolean : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait Clip extends Operator {

  def Clip1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None)
    : (Tensor[T, J])


  def Clip6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,max : Option[(Float)] = None,min : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Gather extends Operator {

  def Gather1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,Tind : Int | Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, indices: Tensor[Tind, J], indicesname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Flatten extends Operator {

  def Flatten1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Split extends Operator {

  def Split1[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,split: Option[Tensor[T, J]] = None,axis : Option[(Int)] = None,splitAttr : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Split2[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None,splitAttr : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait InstanceNormalization extends Operator {

  def InstanceNormalization1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,consumed_inputs : Option[(Seq[Int])] = None,epsilon : Option[(Float)] = None)
    : (Tensor[T, J])


  def InstanceNormalization6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,epsilon : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Div extends Operator {

  def Div1[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Div6[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Div7[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Mul extends Operator {

  def Mul1[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Mul6[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Mul7[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Sqrt extends Operator {

  def Sqrt1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Sqrt6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Neg extends Operator {

  def Neg1[T : Float16 | Float | Double | Float | Int | Byte | Short | Long | Float16 | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Neg6[T : Float16 | Float | Double | Float | Int | Byte | Short | Long | Float16 | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait ArgMin extends Operator {

  def ArgMin1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
    : (Tensor[Long, J])

}
trait Upsample extends Operator {

  def Upsample1[T : Boolean | Int | Long | Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,height_scaleAttr : (Float),mode : Option[(String)] = None,width_scaleAttr : (Float))
    : (Tensor[T, J])


  def Upsample7[T : Boolean | Int | Long | Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,mode : Option[(String)] = None,scaleAttrs : (Seq[Float]))
    : (Tensor[T, J])

}
trait Mean extends Operator {

  def Mean1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Mean6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])


  def Mean8[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait Squeeze extends Operator {

  def Squeeze1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait Softsign extends Operator {

  def Softsign1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait Sigmoid extends Operator {

  def Sigmoid1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Sigmoid6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait ThresholdedRelu extends Operator {

  def ThresholdedRelu1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait LSTM extends Operator {

  def LSTM1[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def LSTM7[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J])

}
trait Sub extends Operator {

  def Sub1[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Sub6[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Sub7[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Affine extends Operator {

  def Affine1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
    : (Tensor[T, J])

}
trait Softplus extends Operator {

  def Softplus1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait Tanh extends Operator {

  def Tanh1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Tanh6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait ReduceMin extends Operator {

  def ReduceMin1[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Transpose extends Operator {

  def Transpose1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,perm : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait RandomNormalLike extends Operator {

  def RandomNormalLike1[T1 : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T2 : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None)
    : (Tensor[T2, J])

}
trait GlobalLpPool extends Operator {

  def GlobalLpPool1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Float)] = None)
    : (Tensor[T, J])


  def GlobalLpPool2[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Asin extends Operator {

  def Asin7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}
trait PRelu extends Operator {

  def PRelu1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def PRelu6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
    : (Tensor[T, J])


  def PRelu7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
    : (Tensor[T, J])

}
trait MatMul extends Operator {

  def MatMul1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait GivenTensorFill extends Operator {

  def GivenTensorFill1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,shapeInput: Option[Tensor[T, J]] = None,extra_shape : Option[(Seq[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Seq[Int])] = None,values : Option[(Seq[Float])] = None)
    : (Tensor[T, J])

}
trait Conv extends Operator {

  def Conv1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Seq[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Seq[Int])] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait Add extends Operator {

  def Add1[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Add6[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T, J])


  def Add7[T : Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T, J])

}
trait Abs extends Operator {

  def Abs1[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Abs6[T : Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait ReduceL1 extends Operator {

  def ReduceL11[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Scan extends Operator {

  def Scan8[I : Long : Numeric:ClassTag:Field,V : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,sequence_lens: Option[Tensor[I, J]] = None,body : (Graph),directions : Option[(Seq[Int])] = None,num_scan_inputs : (Int))
    : (Tensor[V, J])

}
trait BatchNormalization extends Operator {

  def BatchNormalization1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,consumed_inputs : (Seq[Int]),epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def BatchNormalization6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def BatchNormalization7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])

}
trait Xor extends Operator {

  def Xor1[T : Boolean : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Xor7[T : Boolean : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait Equal extends Operator {

  def Equal1[T : Boolean | Int | Long : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
    : (Tensor[T1, J])


  def Equal7[T : Boolean | Int | Long : Numeric:ClassTag:Field,T1 : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : (Tensor[T1, J])

}
trait GRU extends Operator {

  def GRU1[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def GRU3[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def GRU7[T : Float16 | Float | Double : Numeric:ClassTag:Field,T1 : Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None)
    : (Tensor[T, J], Tensor[T, J])

}
trait Expand extends Operator {

  def Expand8[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, shape: Tensor[Long, J], shapename: String)
    : (Tensor[T, J])

}
trait ArgMax extends Operator {

  def ArgMax1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
    : (Tensor[Long, J])

}
trait Dropout extends Operator {

  def Dropout1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def Dropout6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
    : (Tensor[T, J], Tensor[T, J])


  def Dropout7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,ratio : Option[(Float)] = None)
    : (Tensor[T, J], Tensor[T, J])

}
trait Reciprocal extends Operator {

  def Reciprocal1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
    : (Tensor[T, J])


  def Reciprocal6[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait RandomNormal extends Operator {

  def RandomNormal1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : (Tensor[T, J])

}
trait ReduceL2 extends Operator {

  def ReduceL21[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait GRUUnit extends Operator {

  def GRUUnit1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,hidden_prev: Tensor[T, J], hidden_prevname: String, gates: Tensor[T, J], gatesname: String, seq_lengths: Tensor[T, J], seq_lengthsname: String, t: Tensor[T, J], tname: String,drop_states : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Hardmax extends Operator {

  def Hardmax1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait SpaceToDepth extends Operator {

  def SpaceToDepth1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (Int))
    : (Tensor[T, J])

}
trait ReduceSum extends Operator {

  def ReduceSum1[T : UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait DepthToSpace extends Operator {

  def DepthToSpace1[T : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (Int))
    : (Tensor[T, J])

}
trait Not extends Operator {

  def Not1[T : Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : (Tensor[T, J])

}
trait If extends Operator {

  def If1[B : Boolean : Numeric:ClassTag:Field,V : UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,cond: Tensor[B, J], condname: String,else_branch : (Graph),then_branch : (Graph))
    : (Tensor[V, J])

}
trait ConvTranspose extends Operator {

  def ConvTranspose1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Seq[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Seq[Int])] = None,output_padding : Option[(Seq[Int])] = None,output_shape : Option[(Seq[Int])] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
    : (Tensor[T, J])

}
trait LogSoftmax extends Operator {

  def LogSoftmax1[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
    : (Tensor[T, J])

}
trait Sin extends Operator {

  def Sin7[T : Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : (Tensor[T, J])

}}
