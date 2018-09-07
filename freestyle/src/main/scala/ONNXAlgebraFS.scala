package org.emergentorder

import freestyle.free._
import freestyle.free.implicits._
import scala.{specialized => sp}
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import scala.reflect.ClassTag

package onnx {
@free trait DataSourceFS extends DataSource {
  def inputData[VV:Numeric:ClassTag]: FS[Tensor[VV]]
  def getParams[VV:Numeric:ClassTag](name: String): FS[Tensor[VV]]
  def getAttributes[VV:Numeric:ClassTag](name: String): FS[Tensor[VV]]
}
@free trait MatMulFS extends Operator with MatMul {

  def MatMul1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T])]

}
@free trait AtanFS extends Operator with Atan {

  def Atan7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait ReduceLogSumFS extends Operator with ReduceLogSum {

  def ReduceLogSum1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait IfFS extends Operator with If {

  def If1[B <: Boolean : Numeric:ClassTag,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,cond: Tensor[B], condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : FS[(Tensor[V])]

}
@free trait ScaleFS extends Operator with Scale {

  def Scale1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait CeilFS extends Operator with Ceil {

  def Ceil1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Ceil6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait LoopFS extends Operator with Loop {

  def Loop1[I <: Long : Numeric:ClassTag,B <: Boolean : Numeric:ClassTag,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,M: I, Mname: String, cond: B, condname: String,body : (Seq[Float]))
    : FS[(Tensor[V])]

}
@free trait GlobalLpPoolFS extends Operator with GlobalLpPool {

  def GlobalLpPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,p : Option[(Int)] = None)
    : FS[(Tensor[T])]


  def GlobalLpPool2[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,p : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait RandomNormalLikeFS extends Operator with RandomNormalLike {

  def RandomNormalLike1[T1 <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T2 <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2])]

}
@free trait MeanFS extends Operator with Mean {

  def Mean1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Mean6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Mean8[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]

}
@free trait OrFS extends Operator with Or {

  def Or1[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1])]


  def Or7[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T1])]

}
@free trait RandomNormalFS extends Operator with RandomNormal {

  def RandomNormal1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]

}
@free trait ConstantFS extends Operator with Constant {

  def Constant1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]

}
@free trait GatherFS extends Operator with Gather {

  def Gather1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,Tind <: Int |: Long : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String, indices: Tensor[Tind], indicesname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait InstanceNormalizationFS extends Operator with InstanceNormalization {

  def InstanceNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : FS[(Tensor[T])]


  def InstanceNormalization6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String,epsilon : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait SigmoidFS extends Operator with Sigmoid {

  def Sigmoid1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Sigmoid6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait ReduceLogSumExpFS extends Operator with ReduceLogSumExp {

  def ReduceLogSumExp1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait XorFS extends Operator with Xor {

  def Xor1[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1])]


  def Xor7[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T1])]

}
@free trait FlattenFS extends Operator with Flatten {

  def Flatten1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait ClipFS extends Operator with Clip {

  def Clip1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(Tensor[T])]


  def Clip6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait PadFS extends Operator with Pad {

  def Pad1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,mode : Option[(Tensor[_])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : FS[(Tensor[T])]


  def Pad2[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,mode : Option[(Tensor[_])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait ImageScalerFS extends Operator with ImageScaler {

  def ImageScaler1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait TransposeFS extends Operator with Transpose {

  def Transpose1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,perm : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]

}
@free trait ThresholdedReluFS extends Operator with ThresholdedRelu {

  def ThresholdedRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait IdentityFS extends Operator with Identity {

  def Identity1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait LogFS extends Operator with Log {

  def Log1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Log6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait SumFS extends Operator with Sum {

  def Sum1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Sum6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Sum8[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]

}
@free trait AddFS extends Operator with Add {

  def Add1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Add6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T])]


  def Add7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T])]

}
@free trait PowFS extends Operator with Pow {

  def Pow1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, Y: Tensor[T], Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T])]


  def Pow7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, Y: Tensor[T], Yname: String)
    : FS[(Tensor[T])]

}
@free trait TanhFS extends Operator with Tanh {

  def Tanh1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Tanh6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait LpPoolFS extends Operator with LpPool {

  def LpPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def LpPool2[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]

}
@free trait ReciprocalFS extends Operator with Reciprocal {

  def Reciprocal1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Reciprocal6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait SoftsignFS extends Operator with Softsign {

  def Softsign1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait LpNormalizationFS extends Operator with LpNormalization {

  def LpNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait NegFS extends Operator with Neg {

  def Neg1[T <: Float16 |: Float |: Double |: Float |: Int |: Byte |: Short |: Long |: Float16 |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Neg6[T <: Float16 |: Float |: Double |: Float |: Int |: Byte |: Short |: Long |: Float16 |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait GRUFS extends Operator with GRU {

  def GRU1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T])]


  def GRU3[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T])]


  def GRU7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T])]

}
@free trait DepthToSpaceFS extends Operator with DepthToSpace {

  def DepthToSpace1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,blocksize : (String))
    : FS[(Tensor[T])]

}
@free trait ShapeFS extends Operator with Shape {

  def Shape1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T1 <: Long : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String)
    : FS[(Tensor[T1])]

}
@free trait BatchNormalizationFS extends Operator with BatchNormalization {

  def BatchNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]

}
@free trait MultinomialFS extends Operator with Multinomial {

  def Multinomial7[T1 <: Float16 |: Float |: Double : Numeric:ClassTag,T2 <: Int |: Long : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2])]

}
@free trait NotFS extends Operator with Not {

  def Not1[T <: Boolean : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait PReluFS extends Operator with PRelu {

  def PRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def PRelu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String)
    : FS[(Tensor[T])]


  def PRelu7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String)
    : FS[(Tensor[T])]

}
@free trait MinFS extends Operator with Min {

  def Min1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Min6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Min8[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]

}
@free trait AbsFS extends Operator with Abs {

  def Abs1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Abs6[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait DropoutFS extends Operator with Dropout {

  def Dropout1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(Tensor[T], Tensor[T])]


  def Dropout6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(Tensor[T], Tensor[T])]


  def Dropout7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,ratio : Option[(Int)] = None)
    : FS[(Tensor[T], Tensor[T])]

}
@free trait UpsampleFS extends Operator with Upsample {

  def Upsample1[T <: Boolean |: Int |: Long |: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,height_scaleAttr : (Int),mode : Option[(Tensor[_])] = None,width_scaleAttr : (Int))
    : FS[(Tensor[T])]


  def Upsample7[T <: Boolean |: Int |: Long |: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,mode : Option[(Tensor[_])] = None,scaleAttrs : (Seq[Int]))
    : FS[(Tensor[T])]

}
@free trait UnsqueezeFS extends Operator with Unsqueeze {

  def Unsqueeze1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : (Seq[String]))
    : FS[(Tensor[T])]

}
@free trait SqrtFS extends Operator with Sqrt {

  def Sqrt1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Sqrt6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait SoftmaxFS extends Operator with Softmax {

  def Softmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait ReduceSumSquareFS extends Operator with ReduceSumSquare {

  def ReduceSumSquare1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait ArgMinFS extends Operator with ArgMin {

  def ArgMin1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long])]

}
@free trait LogSoftmaxFS extends Operator with LogSoftmax {

  def LogSoftmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait AcosFS extends Operator with Acos {

  def Acos7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait SubFS extends Operator with Sub {

  def Sub1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Sub6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T])]


  def Sub7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T])]

}
@free trait EluFS extends Operator with Elu {

  def Elu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Elu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait HardSigmoidFS extends Operator with HardSigmoid {

  def HardSigmoid1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def HardSigmoid6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait SliceFS extends Operator with Slice {

  def Slice1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : FS[(Tensor[T])]

}
@free trait TanFS extends Operator with Tan {

  def Tan7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait SqueezeFS extends Operator with Squeeze {

  def Squeeze1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]

}
@free trait GlobalAveragePoolFS extends Operator with GlobalAveragePool {

  def GlobalAveragePool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait LeakyReluFS extends Operator with LeakyRelu {

  def LeakyRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def LeakyRelu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait TileFS extends Operator with Tile {

  def Tile1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, tiles: Tensor[T], tilesname: String, axis: Tensor[T], axisname: String)
    : FS[(Tensor[T])]


  def Tile6[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T1 <: Long : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, repeats: Tensor[T1], repeatsname: String)
    : FS[(Tensor[T])]

}
@free trait GRUUnitFS extends Operator with GRUUnit {

  def GRUUnit1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,hidden_prev: Tensor[T], hidden_prevname: String, gates: Tensor[T], gatesname: String, seq_lengths: Tensor[T], seq_lengthsname: String, t: Tensor[T], tname: String,drop_states : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait GlobalMaxPoolFS extends Operator with GlobalMaxPool {

  def GlobalMaxPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait CastFS extends Operator with Cast {

  def Cast1[T1 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag,T2 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,to : (Tensor[_]))
    : FS[(Tensor[T2])]


  def Cast6[T1 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag,T2 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,to : (String))
    : FS[(Tensor[T2])]

}
@free trait ReduceMaxFS extends Operator with ReduceMax {

  def ReduceMax1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait SoftplusFS extends Operator with Softplus {

  def Softplus1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait ReduceMeanFS extends Operator with ReduceMean {

  def ReduceMean1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait ReduceSumFS extends Operator with ReduceSum {

  def ReduceSum1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait GivenTensorFillFS extends Operator with GivenTensorFill {

  def GivenTensorFill1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,shapeInput: Option[Tensor[T]] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : FS[(Tensor[T])]

}
@free trait SplitFS extends Operator with Split {

  def Split1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,split: Option[Tensor[T]] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Split2[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]

}
@free trait ConvFS extends Operator with Conv {

  def Conv1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String,B: Option[Tensor[T]] = None,auto_pad : Option[(Tensor[_])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]

}
@free trait RNNFS extends Operator with RNN {

  def RNN1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T])]


  def RNN7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T])]

}
@free trait GreaterFS extends Operator with Greater {

  def Greater1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1])]


  def Greater7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T1])]

}
@free trait ConvTransposeFS extends Operator with ConvTranspose {

  def ConvTranspose1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String,B: Option[Tensor[T]] = None,auto_pad : Option[(Tensor[_])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]

}
@free trait GemmFS extends Operator with Gemm {

  def Gemm1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T])]


  def Gemm6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T])]


  def Gemm7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait SizeFS extends Operator with Size {

  def Size1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T1 <: Long : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String)
    : FS[(Tensor[T1])]

}
@free trait ReduceProdFS extends Operator with ReduceProd {

  def ReduceProd1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait LRNFS extends Operator with LRN {

  def LRN1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : FS[(Tensor[T])]

}
@free trait SpaceToDepthFS extends Operator with SpaceToDepth {

  def SpaceToDepth1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,blocksize : (String))
    : FS[(Tensor[T])]

}
@free trait AffineFS extends Operator with Affine {

  def Affine1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait RandomUniformFS extends Operator with RandomUniform {

  def RandomUniform1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]

}
@free trait MaxPoolFS extends Operator with MaxPool {

  def MaxPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def MaxPool8[T <: Float16 |: Float |: Double : Numeric:ClassTag,I <: Long : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,storage_order : Option[(String)] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T], Tensor[I])]

}
@free trait RandomUniformLikeFS extends Operator with RandomUniformLike {

  def RandomUniformLike1[T1 <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T2 <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2])]

}
@free trait CosFS extends Operator with Cos {

  def Cos7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait ReshapeFS extends Operator with Reshape {

  def Reshape1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Reshape5[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String, shape: Tensor[Long], shapename: String)
    : FS[(Tensor[T])]

}
@free trait MulFS extends Operator with Mul {

  def Mul1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Mul6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T])]


  def Mul7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T])]

}
@free trait MeanVarianceNormalizationFS extends Operator with MeanVarianceNormalization {

  def MeanVarianceNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait MaxFS extends Operator with Max {

  def Max1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Max6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Max8[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]

}
@free trait ScanFS extends Operator with Scan {

  def Scan8[I <: Long : Numeric:ClassTag,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,sequence_lens: Option[Tensor[I]] = None,body : (Seq[Float]),directions : Option[(Seq[String])] = None,num_scan_inputs : (String))
    : FS[(Tensor[V])]

}
@free trait ReduceL2FS extends Operator with ReduceL2 {

  def ReduceL21[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait MaxRoiPoolFS extends Operator with MaxRoiPool {

  def MaxRoiPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, rois: Tensor[T], roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait ConcatFS extends Operator with Concat {

  def Concat1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]


  def Concat4[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String)
    : FS[(Tensor[T])]

}
@free trait LessFS extends Operator with Less {

  def Less1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1])]


  def Less7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T1])]

}
@free trait DivFS extends Operator with Div {

  def Div1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Div6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T])]


  def Div7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T])]

}
@free trait ConstantFillFS extends Operator with ConstantFill {

  def ConstantFill1[T1 <: Float |: Int |: Long |: Boolean : Numeric:ClassTag,T2 <: Float |: Int |: Long |: Boolean : Numeric:ClassTag](name: String,input: Option[Tensor[T1]] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : FS[(Tensor[T2])]

}
@free trait CropFS extends Operator with Crop {

  def Crop1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]

}
@free trait EqualFS extends Operator with Equal {

  def Equal1[T <: Boolean |: Int |: Long : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1])]


  def Equal7[T <: Boolean |: Int |: Long : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T1])]

}
@free trait FloorFS extends Operator with Floor {

  def Floor1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Floor6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait ExpandFS extends Operator with Expand {

  def Expand8[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, shape: Tensor[Long], shapename: String)
    : FS[(Tensor[T])]

}
@free trait HardmaxFS extends Operator with Hardmax {

  def Hardmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait LSTMFS extends Operator with LSTM {

  def LSTM1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T], Tensor[T])]


  def LSTM7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : FS[(Tensor[T], Tensor[T], Tensor[T])]

}
@free trait AndFS extends Operator with And {

  def And1[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1])]


  def And7[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : FS[(Tensor[T1])]

}
@free trait AveragePoolFS extends Operator with AveragePool {

  def AveragePool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def AveragePool7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]

}
@free trait AsinFS extends Operator with Asin {

  def Asin7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait ReduceL1FS extends Operator with ReduceL1 {

  def ReduceL11[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait TopKFS extends Operator with TopK {

  def TopK1[T <: Float16 |: Float |: Double : Numeric:ClassTag,I <: Long : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,axis : Option[(String)] = None,k : (String))
    : FS[(Tensor[T], Tensor[I])]

}
@free trait SinFS extends Operator with Sin {

  def Sin7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait ReduceMinFS extends Operator with ReduceMin {

  def ReduceMin1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T])]

}
@free trait ReluFS extends Operator with Relu {

  def Relu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Relu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : FS[(Tensor[T])]

}
@free trait ParametricSoftplusFS extends Operator with ParametricSoftplus {

  def ParametricSoftplus1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait SeluFS extends Operator with Selu {

  def Selu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : FS[(Tensor[T])]


  def Selu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait ScaledTanhFS extends Operator with ScaledTanh {

  def ScaledTanh1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T])]

}
@free trait ExpFS extends Operator with Exp {

  def Exp1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T])]


  def Exp6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : FS[(Tensor[T])]

}
@free trait ArgMaxFS extends Operator with ArgMax {

  def ArgMax1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long])]

}}
