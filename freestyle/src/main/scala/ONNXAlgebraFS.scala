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
import spire.implicits._
import spire.algebra.Field
import scala.reflect.ClassTag
import singleton.ops._

package onnx {
@free trait DataSourceFS extends DataSource {
  def inputData[T <: Float16 |: Float |: Double :Numeric:ClassTag:Field, J <: XInt]: FS[Tensor[T, J]]
  def getParams[T <: Float16 |: Float |: Double :Numeric:ClassTag:Field, J <: XInt](name: String): FS[Tensor[T, J]]
  def getAttributes[T <: Float16 |: Float |: Double :Numeric:ClassTag:Field, J <: XInt](name: String): FS[Tensor[T, J]]
}
@free trait ParametricSoftplusFS extends Operator with ParametricSoftplus {

  def ParametricSoftplus1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait AcosFS extends Operator with Acos {

  def Acos7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait LRNFS extends Operator with LRN {

  def LRN1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : FS[(Tensor[T, J])]

}
@free trait FlattenFS extends Operator with Flatten {

  def Flatten1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait SubFS extends Operator with Sub {

  def Sub1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Sub6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Sub7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
@free trait SqueezeFS extends Operator with Squeeze {

  def Squeeze1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
@free trait ArgMinFS extends Operator with ArgMin {

  def ArgMin1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long, J])]

}
@free trait LogSoftmaxFS extends Operator with LogSoftmax {

  def LogSoftmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait FloorFS extends Operator with Floor {

  def Floor1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Floor6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait AbsFS extends Operator with Abs {

  def Abs1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Abs6[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait SliceFS extends Operator with Slice {

  def Slice1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : FS[(Tensor[T, J])]

}
@free trait MatMulFS extends Operator with MatMul {

  def MatMul1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
@free trait GRUUnitFS extends Operator with GRUUnit {

  def GRUUnit1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,hidden_prev: Tensor[T, J], hidden_prevname: String, gates: Tensor[T, J], gatesname: String, seq_lengths: Tensor[T, J], seq_lengthsname: String, t: Tensor[T, J], tname: String,drop_states : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait RandomNormalLikeFS extends Operator with RandomNormalLike {

  def RandomNormalLike1[T1 <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field,T2 <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2, J])]

}
@free trait PowFS extends Operator with Pow {

  def Pow1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Pow7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String)
    : FS[(Tensor[T, J])]

}
@free trait UpsampleFS extends Operator with Upsample {

  def Upsample1[T <: Boolean |: Int |: Long |: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,height_scaleAttr : (Int),mode : Option[(Tensor[_, J])] = None,width_scaleAttr : (Int))
    : FS[(Tensor[T, J])]


  def Upsample7[T <: Boolean |: Int |: Long |: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,mode : Option[(Tensor[_, J])] = None,scaleAttrs : (Seq[Int]))
    : FS[(Tensor[T, J])]

}
@free trait RandomNormalFS extends Operator with RandomNormal {

  def RandomNormal1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
@free trait MaxFS extends Operator with Max {

  def Max1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Max6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Max8[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
@free trait UnsqueezeFS extends Operator with Unsqueeze {

  def Unsqueeze1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : (Seq[String]))
    : FS[(Tensor[T, J])]

}
@free trait MaxPoolFS extends Operator with MaxPool {

  def MaxPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[_, J])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def MaxPool8[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,I <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[_, J])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,storage_order : Option[(String)] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J], Tensor[I, J])]

}
@free trait ReduceSumFS extends Operator with ReduceSum {

  def ReduceSum1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait ReluFS extends Operator with Relu {

  def Relu1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Relu6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait SizeFS extends Operator with Size {

  def Size1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field,T1 <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
    : FS[(Tensor[T1, J])]

}
@free trait ClipFS extends Operator with Clip {

  def Clip1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def Clip6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait CeilFS extends Operator with Ceil {

  def Ceil1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Ceil6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait EluFS extends Operator with Elu {

  def Elu1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Elu6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait ReciprocalFS extends Operator with Reciprocal {

  def Reciprocal1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Reciprocal6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait EqualFS extends Operator with Equal {

  def Equal1[T <: Boolean |: Int |: Long : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Equal7[T <: Boolean |: Int |: Long : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
@free trait ExpandFS extends Operator with Expand {

  def Expand8[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, shape: Tensor[Long, J], shapename: String)
    : FS[(Tensor[T, J])]

}
@free trait NotFS extends Operator with Not {

  def Not1[T <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait SoftmaxFS extends Operator with Softmax {

  def Softmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait CastFS extends Operator with Cast {

  def Cast1[T1 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag:Field,T2 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (Tensor[_, J]))
    : FS[(Tensor[T2, J])]


  def Cast6[T1 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag:Field,T2 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (String))
    : FS[(Tensor[T2, J])]

}
@free trait SoftplusFS extends Operator with Softplus {

  def Softplus1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait GreaterFS extends Operator with Greater {

  def Greater1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Greater7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
@free trait ConvTransposeFS extends Operator with ConvTranspose {

  def ConvTranspose1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(Tensor[_, J])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
@free trait GatherFS extends Operator with Gather {

  def Gather1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field,Tind <: Int |: Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, indices: Tensor[Tind, J], indicesname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait AffineFS extends Operator with Affine {

  def Affine1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait IdentityFS extends Operator with Identity {

  def Identity1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait HardSigmoidFS extends Operator with HardSigmoid {

  def HardSigmoid1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def HardSigmoid6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait DropoutFS extends Operator with Dropout {

  def Dropout1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def Dropout6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def Dropout7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,ratio : Option[(Int)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]

}
@free trait ShapeFS extends Operator with Shape {

  def Shape1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field,T1 <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
    : FS[(Tensor[T1, J])]

}
@free trait ThresholdedReluFS extends Operator with ThresholdedRelu {

  def ThresholdedRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait ArgMaxFS extends Operator with ArgMax {

  def ArgMax1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long, J])]

}
@free trait MeanVarianceNormalizationFS extends Operator with MeanVarianceNormalization {

  def MeanVarianceNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait AddFS extends Operator with Add {

  def Add1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Add6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Add7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
@free trait InstanceNormalizationFS extends Operator with InstanceNormalization {

  def InstanceNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def InstanceNormalization6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,epsilon : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait LeakyReluFS extends Operator with LeakyRelu {

  def LeakyRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def LeakyRelu6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait MultinomialFS extends Operator with Multinomial {

  def Multinomial7[T1 <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T2 <: Int |: Long : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2, J])]

}
@free trait AndFS extends Operator with And {

  def And1[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def And7[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
@free trait ConstantFillFS extends Operator with ConstantFill {

  def ConstantFill1[T1 <: Float |: Int |: Long |: Boolean : Numeric:ClassTag:Field,T2 <: Float |: Int |: Long |: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,input: Option[Tensor[T1, J]] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : FS[(Tensor[T2, J])]

}
@free trait MeanFS extends Operator with Mean {

  def Mean1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Mean6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Mean8[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
@free trait GRUFS extends Operator with GRU {

  def GRU1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_, J])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def GRU3[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_, J])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def GRU7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_, J])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]

}
@free trait ImageScalerFS extends Operator with ImageScaler {

  def ImageScaler1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait LpPoolFS extends Operator with LpPool {

  def LpPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[_, J])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def LpPool2[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[_, J])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
@free trait IfFS extends Operator with If {

  def If1[B <: Boolean : Numeric:ClassTag:Field,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,cond: Tensor[B, J], condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : FS[(Tensor[V, J])]

}
@free trait BatchNormalizationFS extends Operator with BatchNormalization {

  def BatchNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def BatchNormalization6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def BatchNormalization7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])]

}
@free trait GivenTensorFillFS extends Operator with GivenTensorFill {

  def GivenTensorFill1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,shapeInput: Option[Tensor[T, J]] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : FS[(Tensor[T, J])]

}
@free trait DepthToSpaceFS extends Operator with DepthToSpace {

  def DepthToSpace1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (String))
    : FS[(Tensor[T, J])]

}
@free trait AveragePoolFS extends Operator with AveragePool {

  def AveragePool1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[_, J])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def AveragePool7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(Tensor[_, J])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
@free trait TopKFS extends Operator with TopK {

  def TopK1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,I <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,axis : Option[(String)] = None,k : (String))
    : FS[(Tensor[T, J], Tensor[I, J])]

}
@free trait ReduceSumSquareFS extends Operator with ReduceSumSquare {

  def ReduceSumSquare1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait XorFS extends Operator with Xor {

  def Xor1[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Xor7[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
@free trait CropFS extends Operator with Crop {

  def Crop1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
@free trait ConvFS extends Operator with Conv {

  def Conv1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(Tensor[_, J])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
@free trait ScaledTanhFS extends Operator with ScaledTanh {

  def ScaledTanh1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait SplitFS extends Operator with Split {

  def Split1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,split: Option[Tensor[T, J]] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Split2[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
@free trait GlobalLpPoolFS extends Operator with GlobalLpPool {

  def GlobalLpPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def GlobalLpPool2[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait GemmFS extends Operator with Gemm {

  def Gemm1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Gemm6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Gemm7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait ReduceMeanFS extends Operator with ReduceMean {

  def ReduceMean1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait SinFS extends Operator with Sin {

  def Sin7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait GlobalAveragePoolFS extends Operator with GlobalAveragePool {

  def GlobalAveragePool1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait GlobalMaxPoolFS extends Operator with GlobalMaxPool {

  def GlobalMaxPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait ScanFS extends Operator with Scan {

  def Scan8[I <: Long : Numeric:ClassTag:Field,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,sequence_lens: Option[Tensor[I, J]] = None,body : (Seq[Float]),directions : Option[(Seq[String])] = None,num_scan_inputs : (String))
    : FS[(Tensor[V, J])]

}
@free trait LoopFS extends Operator with Loop {

  def Loop1[I <: Long : Numeric:ClassTag:Field,B <: Boolean : Numeric:ClassTag:Field,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,M: I, Mname: String, cond: B, condname: String,body : (Seq[Float]))
    : FS[(Tensor[V, J])]

}
@free trait ReduceMinFS extends Operator with ReduceMin {

  def ReduceMin1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait TanFS extends Operator with Tan {

  def Tan7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait RandomUniformFS extends Operator with RandomUniform {

  def RandomUniform1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
@free trait SigmoidFS extends Operator with Sigmoid {

  def Sigmoid1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Sigmoid6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait MulFS extends Operator with Mul {

  def Mul1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Mul6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Mul7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
@free trait ReduceLogSumFS extends Operator with ReduceLogSum {

  def ReduceLogSum1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait LogFS extends Operator with Log {

  def Log1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Log6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait SqrtFS extends Operator with Sqrt {

  def Sqrt1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Sqrt6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait RNNFS extends Operator with RNN {

  def RNN1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_, J])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]


  def RNN7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_, J])] = None,hidden_size : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J])]

}
@free trait SpaceToDepthFS extends Operator with SpaceToDepth {

  def SpaceToDepth1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (String))
    : FS[(Tensor[T, J])]

}
@free trait AsinFS extends Operator with Asin {

  def Asin7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait LpNormalizationFS extends Operator with LpNormalization {

  def LpNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait ReduceL2FS extends Operator with ReduceL2 {

  def ReduceL21[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait ConstantFS extends Operator with Constant {

  def Constant1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
@free trait ScaleFS extends Operator with Scale {

  def Scale1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait ReduceProdFS extends Operator with ReduceProd {

  def ReduceProd1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait DivFS extends Operator with Div {

  def Div1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Div6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T, J])]


  def Div7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T, J])]

}
@free trait TanhFS extends Operator with Tanh {

  def Tanh1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Tanh6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait AtanFS extends Operator with Atan {

  def Atan7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait HardmaxFS extends Operator with Hardmax {

  def Hardmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait LessFS extends Operator with Less {

  def Less1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Less7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
@free trait PadFS extends Operator with Pad {

  def Pad1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(Tensor[_, J])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def Pad2[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(Tensor[_, J])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait CosFS extends Operator with Cos {

  def Cos7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait SoftsignFS extends Operator with Softsign {

  def Softsign1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}
@free trait TileFS extends Operator with Tile {

  def Tile1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, tiles: Tensor[T, J], tilesname: String, axis: Tensor[T, J], axisname: String)
    : FS[(Tensor[T, J])]


  def Tile6[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field,T1 <: Long : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, repeats: Tensor[T1, J], repeatsname: String)
    : FS[(Tensor[T, J])]

}
@free trait MinFS extends Operator with Min {

  def Min1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Min6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Min8[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
@free trait ReduceMaxFS extends Operator with ReduceMax {

  def ReduceMax1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait ReduceLogSumExpFS extends Operator with ReduceLogSumExp {

  def ReduceLogSumExp1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait SeluFS extends Operator with Selu {

  def Selu1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : FS[(Tensor[T, J])]


  def Selu6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait SumFS extends Operator with Sum {

  def Sum1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Sum6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Sum8[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
@free trait ReshapeFS extends Operator with Reshape {

  def Reshape1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Reshape5[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, shape: Tensor[Long, J], shapename: String)
    : FS[(Tensor[T, J])]

}
@free trait LSTMFS extends Operator with LSTM {

  def LSTM1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_, J])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J])]


  def LSTM7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field,T1 <: Int : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_, J]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_, J])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : FS[(Tensor[T, J], Tensor[T, J], Tensor[T, J])]

}
@free trait ReduceL1FS extends Operator with ReduceL1 {

  def ReduceL11[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[T, J])]

}
@free trait TransposeFS extends Operator with Transpose {

  def Transpose1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,perm : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]

}
@free trait RandomUniformLikeFS extends Operator with RandomUniformLike {

  def RandomUniformLike1[T1 <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field,T2 <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[T2, J])]

}
@free trait NegFS extends Operator with Neg {

  def Neg1[T <: Float16 |: Float |: Double |: Float |: Int |: Byte |: Short |: Long |: Float16 |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Neg6[T <: Float16 |: Float |: Double |: Float |: Int |: Byte |: Short |: Long |: Float16 |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
    : FS[(Tensor[T, J])]

}
@free trait PReluFS extends Operator with PRelu {

  def PRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def PRelu6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
    : FS[(Tensor[T, J])]


  def PRelu7[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
    : FS[(Tensor[T, J])]

}
@free trait MaxRoiPoolFS extends Operator with MaxRoiPool {

  def MaxRoiPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, rois: Tensor[T, J], roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[T, J])]

}
@free trait OrFS extends Operator with Or {

  def Or1[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[T1, J])]


  def Or7[T <: Boolean : Numeric:ClassTag:Field,T1 <: Boolean : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
    : FS[(Tensor[T1, J])]

}
@free trait ConcatFS extends Operator with Concat {

  def Concat1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]


  def Concat4[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag:Field, J <: XInt](name: String)
    : FS[(Tensor[T, J])]

}
@free trait ExpFS extends Operator with Exp {

  def Exp1[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[T, J])]


  def Exp6[T <: Float16 |: Float |: Double : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
    : FS[(Tensor[T, J])]

}}
