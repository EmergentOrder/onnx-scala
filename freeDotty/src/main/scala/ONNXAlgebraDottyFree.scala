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

package onnx {
 trait DataSourceFree extends DataSource {
  def inputDataFree[T <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | Complex[Float] | Complex[Double]:Numeric:ClassTag:Field, J <: XInt]: FS[Tensor[T, J]]
  def getParamsFree[T <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | Complex[Float] | Complex[Double]:Numeric:ClassTag:Field, J <: XInt](name: String): FS[Tensor[T, J]]
  def getAttributesFree[T <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | Complex[Float] | Complex[Double]:Numeric:ClassTag:Field, J <: XInt](name: String): FS[Tensor[T, J]]
}
trait ReduceMinFree extends Operator with ReduceMin {

  def ReduceMin1Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait LessFree extends Operator with Less {

  def Less1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Less7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
trait ArgMinFree extends Operator with ArgMin {

  def ArgMin1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long, J])]

}
trait RandomUniformLikeFree extends Operator with RandomUniformLike {

  def RandomUniformLike1Free[T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T2 <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2, J])]

}
trait UpsampleFree extends Operator with Upsample {

  def Upsample1Free[T <: Boolean | Int | Long | Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,height_scaleAttr : (Int),mode : Option[(Tensor[T, J])] = None,width_scaleAttr : (Int))
    : FS[(Tensor[T, J])]


  def Upsample7Free[T <: Boolean | Int | Long | Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,mode : Option[(Tensor[T, J])] = None,scaleAttrs : (Seq[Int]))
    : FS[(Tensor[T, J])]

}
trait CosFree extends Operator with Cos {

  def Cos7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait ScaleFree extends Operator with Scale {

  def Scale1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait ClipFree extends Operator with Clip {

  def Clip1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def Clip6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait SplitFree extends Operator with Split {

  def Split1Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,split: Option[Tensor[T, J]] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Split2Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
trait CastFree extends Operator with Cast {

  def Cast1Free[T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag:Field,T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (Tensor[T1, J]))
    : FS[(Tensor[T2, J])]


  def Cast6Free[T1 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag:Field,T2 <: Float16 | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (String))
    : FS[(Tensor[T2, J])]

}
trait AndFree extends Operator with And {

  def And1Free[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def And7Free[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
trait AtanFree extends Operator with Atan {

  def Atan7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait ConcatFree extends Operator with Concat {

  def Concat1Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Concat4Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
trait SqueezeFree extends Operator with Squeeze {

  def Squeeze1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
trait ReduceL2Free extends Operator with ReduceL2 {

  def ReduceL21Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait GivenTensorFillFree extends Operator with GivenTensorFill {

  def GivenTensorFill1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,shapeInput: Option[Tensor[T, J]] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : FS[(Tensor[T, J])]

}
trait ExpandFree extends Operator with Expand {

  def Expand8Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, shape: Tensor[Long, J], shapename: String)
    : FS[(Tensor[T, J])]

}
trait ExpFree extends Operator with Exp {

  def Exp1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Exp6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait ThresholdedReluFree extends Operator with ThresholdedRelu {

  def ThresholdedRelu1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait PadFree extends Operator with Pad {

  def Pad1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(Tensor[T, J])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def Pad2Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(Tensor[T, J])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait AffineFree extends Operator with Affine {

  def Affine1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait ImageScalerFree extends Operator with ImageScaler {

  def ImageScaler1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait SoftplusFree extends Operator with Softplus {

  def Softplus1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait GlobalLpPoolFree extends Operator with GlobalLpPool {

  def GlobalLpPool1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def GlobalLpPool2Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait SinFree extends Operator with Sin {

  def Sin7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait FloorFree extends Operator with Floor {

  def Floor1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Floor6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait TileFree extends Operator with Tile {

  def Tile1Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, tiles: Tensor[T, J], tilesname: String, axis: Tensor[T, J], axisname: String)
    : FS[(Tensor[T, J])]


  def Tile6Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T1 <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, repeats: Tensor[T1, J], repeatsname: String)
    : FS[(Tensor[T, J])]

}
trait AddFree extends Operator with Add {

  def Add1Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Add6Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Add7Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
trait SigmoidFree extends Operator with Sigmoid {

  def Sigmoid1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Sigmoid6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait NegFree extends Operator with Neg {

  def Neg1Free[T <: Float16 | Float | Double | Float | Int | Byte | Short | Long | Float16 | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Neg6Free[T <: Float16 | Float | Double | Float | Int | Byte | Short | Long | Float16 | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait DepthToSpaceFree extends Operator with DepthToSpace {

  def DepthToSpace1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (String))
    : FS[(Tensor[T, J])]

}
trait SqrtFree extends Operator with Sqrt {

  def Sqrt1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Sqrt6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait IdentityFree extends Operator with Identity {

  def Identity1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait RNNFree extends Operator with RNN {

  def RNN1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[T, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[T, J])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def RNN7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[T, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[T, J])] = None,hidden_size : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]

}
trait HardmaxFree extends Operator with Hardmax {

  def Hardmax1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait SliceFree extends Operator with Slice {

  def Slice1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : FS[(Tensor[T, J])]

}
trait DropoutFree extends Operator with Dropout {

  def Dropout1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def Dropout6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def Dropout7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,ratio : Option[(Int)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]

}
trait AveragePoolFree extends Operator with AveragePool {

  def AveragePool1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[T, J])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def AveragePool7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[T, J])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
trait BatchNormalizationFree extends Operator with BatchNormalization {

  def BatchNormalization1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def BatchNormalization6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def BatchNormalization7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]

}
trait GRUFree extends Operator with GRU {

  def GRU1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[T, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[T, J])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def GRU3Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[T, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[T, J])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def GRU7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[T, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[T, J])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]

}
trait ParametricSoftplusFree extends Operator with ParametricSoftplus {

  def ParametricSoftplus1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait HardSigmoidFree extends Operator with HardSigmoid {

  def HardSigmoid1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def HardSigmoid6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait ReduceSumFree extends Operator with ReduceSum {

  def ReduceSum1Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait RandomNormalLikeFree extends Operator with RandomNormalLike {

  def RandomNormalLike1Free[T1 <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T2 <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2, J])]

}
trait GatherFree extends Operator with Gather {

  def Gather1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,Tind <: Int | Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, indices: Tensor[Tind, J], indicesname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait ReduceMeanFree extends Operator with ReduceMean {

  def ReduceMean1Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait AsinFree extends Operator with Asin {

  def Asin7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait ConvTransposeFree extends Operator with ConvTranspose {

  def ConvTranspose1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(Tensor[T, J])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
trait ReduceMaxFree extends Operator with ReduceMax {

  def ReduceMax1Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait RandomUniformFree extends Operator with RandomUniform {

  def RandomUniform1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
trait DivFree extends Operator with Div {

  def Div1Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Div6Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Div7Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
trait ShapeFree extends Operator with Shape {

  def Shape1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T1 <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
    : FS[(Tensor[T1, J])]

}
trait ArgMaxFree extends Operator with ArgMax {

  def ArgMax1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long, J])]

}
trait MeanFree extends Operator with Mean {

  def Mean1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Mean6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Mean8Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
trait ReluFree extends Operator with Relu {

  def Relu1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Relu6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait LogSoftmaxFree extends Operator with LogSoftmax {

  def LogSoftmax1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait SpaceToDepthFree extends Operator with SpaceToDepth {

  def SpaceToDepth1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (String))
    : FS[(Tensor[T, J])]

}
trait FlattenFree extends Operator with Flatten {

  def Flatten1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait EluFree extends Operator with Elu {

  def Elu1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Elu6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait AbsFree extends Operator with Abs {

  def Abs1Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Abs6Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait ScaledTanhFree extends Operator with ScaledTanh {

  def ScaledTanh1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait ReduceL1Free extends Operator with ReduceL1 {

  def ReduceL11Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait LRNFree extends Operator with LRN {

  def LRN1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : FS[(Tensor[T, J])]

}
trait GemmFree extends Operator with Gemm {

  def Gemm1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Gemm6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Gemm7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait TopKFree extends Operator with TopK {

  def TopK1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,I <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,axis : Option[(String)] = None,k : (String))
    : FS[(Tensor[T, J], Tensor[I, J])]

}
trait GRUUnitFree extends Operator with GRUUnit {

  def GRUUnit1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,hidden_prev: Tensor[T, J], hidden_prevname: String, gates: Tensor[T, J], gatesname: String, seq_lengths: Tensor[T, J], seq_lengthsname: String, t: Tensor[T, J], tname: String,drop_states : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait MaxRoiPoolFree extends Operator with MaxRoiPool {

  def MaxRoiPool1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, rois: Tensor[T, J], roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait EqualFree extends Operator with Equal {

  def Equal1Free[T <: Boolean | Int | Long : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Equal7Free[T <: Boolean | Int | Long : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
trait TanFree extends Operator with Tan {

  def Tan7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait SubFree extends Operator with Sub {

  def Sub1Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Sub6Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Sub7Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
trait GreaterFree extends Operator with Greater {

  def Greater1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Greater7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
trait XorFree extends Operator with Xor {

  def Xor1Free[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Xor7Free[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
trait LpPoolFree extends Operator with LpPool {

  def LpPool1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[T, J])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def LpPool2Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[T, J])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
trait ReciprocalFree extends Operator with Reciprocal {

  def Reciprocal1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Reciprocal6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait ReduceLogSumFree extends Operator with ReduceLogSum {

  def ReduceLogSum1Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait RandomNormalFree extends Operator with RandomNormal {

  def RandomNormal1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
trait LogFree extends Operator with Log {

  def Log1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Log6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait SoftmaxFree extends Operator with Softmax {

  def Softmax1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait AcosFree extends Operator with Acos {

  def Acos7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait PowFree extends Operator with Pow {

  def Pow1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Pow7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String)
    : FS[(Tensor[T, J])]

}
trait InstanceNormalizationFree extends Operator with InstanceNormalization {

  def InstanceNormalization1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def InstanceNormalization6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,epsilon : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait SumFree extends Operator with Sum {

  def Sum1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Sum6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Sum8Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
trait IfFree extends Operator with If {

  def If1Free[B <: Boolean : Numeric:ClassTag:Field,V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,cond: Tensor[B, J], condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : FS[(Tensor[V, J])]

}
trait SeluFree extends Operator with Selu {

  def Selu1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def Selu6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait LpNormalizationFree extends Operator with LpNormalization {

  def LpNormalization1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait LeakyReluFree extends Operator with LeakyRelu {

  def LeakyRelu1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def LeakyRelu6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
trait ReduceSumSquareFree extends Operator with ReduceSumSquare {

  def ReduceSumSquare1Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait UnsqueezeFree extends Operator with Unsqueeze {

  def Unsqueeze1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : (Seq[String]))
    : FS[(Tensor[T, J])]

}
trait CeilFree extends Operator with Ceil {

  def Ceil1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Ceil6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait SizeFree extends Operator with Size {

  def Size1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field,T1 <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
    : FS[(Tensor[T1, J])]

}
trait MaxFree extends Operator with Max {

  def Max1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Max6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Max8Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
trait ReduceLogSumExpFree extends Operator with ReduceLogSumExp {

  def ReduceLogSumExp1Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait ScanFree extends Operator with Scan {

  def Scan8Free[I <: Long : Numeric:ClassTag:Field,V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,sequence_lens: Option[Tensor[I, J]] = None,body : (Seq[Float]),directions : Option[(Seq[String])] = None,num_scan_inputs : (String))
    : FS[(Tensor[V, J])]

}
trait MultinomialFree extends Operator with Multinomial {

  def Multinomial7Free[T1 <: Float16 | Float | Double : Numeric:ClassTag:Field,T2 <: Int | Long : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2, J])]

}
trait CropFree extends Operator with Crop {

  def Crop1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
trait NotFree extends Operator with Not {

  def Not1Free[T <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait MaxPoolFree extends Operator with MaxPool {

  def MaxPool1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[T, J])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def MaxPool8Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,I <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[T, J])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,storage_order : Option[(String)] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J], Tensor[I, J])]

}
trait MatMulFree extends Operator with MatMul {

  def MatMul1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
trait MinFree extends Operator with Min {

  def Min1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Min6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Min8Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
trait ReduceProdFree extends Operator with ReduceProd {

  def ReduceProd1Free[T <: UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait MeanVarianceNormalizationFree extends Operator with MeanVarianceNormalization {

  def MeanVarianceNormalization1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
trait TransposeFree extends Operator with Transpose {

  def Transpose1Free[T <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,perm : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
trait GlobalMaxPoolFree extends Operator with GlobalMaxPool {

  def GlobalMaxPool1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait PReluFree extends Operator with PRelu {

  def PRelu1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def PRelu6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
    : FS[(Tensor[T, J])]


  def PRelu7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
    : FS[(Tensor[T, J])]

}
trait OrFree extends Operator with Or {

  def Or1Free[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Or7Free[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
trait SoftsignFree extends Operator with Softsign {

  def Softsign1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait ConvFree extends Operator with Conv {

  def Conv1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(Tensor[T, J])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
trait LSTMFree extends Operator with LSTM {

  def LSTM1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[T, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[T, J])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def LSTM7Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[T, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[T, J])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J])]

}
trait TanhFree extends Operator with Tanh {

  def Tanh1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Tanh6Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
trait GlobalAveragePoolFree extends Operator with GlobalAveragePool {

  def GlobalAveragePool1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
trait ReshapeFree extends Operator with Reshape {

  def Reshape1Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Reshape5Free[T <: Float16 | Float | Double | UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, shape: Tensor[Long, J], shapename: String)
    : FS[(Tensor[T, J])]

}
trait ConstantFillFree extends Operator with ConstantFill {

  def ConstantFill1Free[T1 <: Float | Int | Long | Boolean : Numeric:ClassTag:Field,T2 <: Float | Int | Long | Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Option[Tensor[T1, J]] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : FS[(Tensor[T2, J])]

}
trait MulFree extends Operator with Mul {

  def Mul1Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Mul6Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Mul7Free[T <: Float16 | Float | Double | UInt | ULong | Int | Long | Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
trait ConstantFree extends Operator with Constant {

  def Constant1Free[T <: Float16 | Float | Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
trait LoopFree extends Operator with Loop {

  def Loop1Free[I <: Long : Numeric:ClassTag:Field,B <: Boolean : Numeric:ClassTag:Field,V <: UByte | UShort | UInt | ULong | Byte | Short | Int | Long | Float16 | Float | Double | String | Boolean | Complex[Float] | Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,M: I, Mname: String, cond: B, condname: String,body : (Seq[Float]))
    : FS[(Tensor[V, J])]

}}
