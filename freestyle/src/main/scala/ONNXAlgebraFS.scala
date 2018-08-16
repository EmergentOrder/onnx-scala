package org.emergentorder

import freestyle.free._
import freestyle.free.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Numeric
import scala.reflect.ClassTag

package onnx {

@free trait DataSourceFS extends DataSource {
  def inputData[VV:Numeric:ClassTag]: FS[Tensor[VV]]
  def getParams[VV:Numeric:ClassTag](name: String): FS[Tensor[VV]]
  def getAttributes[VV:Numeric:ClassTag](name: String): FS[Tensor[VV]]
}
@free trait IdentityFS extends Operator with Identity {

  def Identity1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait ExpFS extends Operator with Exp {

  def Exp1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Exp6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait ClipFS extends Operator with Clip {

  def Clip1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(T[VV])]


  def Clip6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait TanFS extends Operator with Tan {

  def Tan7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait GlobalAveragePoolFS extends Operator with GlobalAveragePool {

  def GlobalAveragePool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait TransposeFS extends Operator with Transpose {

  def Transpose1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,perm : Option[(Seq[String])] = None)
    : FS[(T[VV])]

}
@free trait AsinFS extends Operator with Asin {

  def Asin7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait HardSigmoidFS extends Operator with HardSigmoid {

  def HardSigmoid1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def HardSigmoid6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait ReduceMaxFS extends Operator with ReduceMax {

  def ReduceMax1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait ArgMinFS extends Operator with ArgMin {

  def ArgMin1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long])]

}
@free trait LpNormalizationFS extends Operator with LpNormalization {

  def LpNormalization1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait EluFS extends Operator with Elu {

  def Elu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Elu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait CeilFS extends Operator with Ceil {

  def Ceil1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Ceil6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait ReduceL1FS extends Operator with ReduceL1 {

  def ReduceL11[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait UnsqueezeFS extends Operator with Unsqueeze {

  def Unsqueeze1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : (Seq[String]))
    : FS[(T[VV])]

}
@free trait GlobalMaxPoolFS extends Operator with GlobalMaxPool {

  def GlobalMaxPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait SoftmaxFS extends Operator with Softmax {

  def Softmax1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait ScaleFS extends Operator with Scale {

  def Scale1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,scaleAttr : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait CropFS extends Operator with Crop {

  def Crop1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : FS[(T[VV])]

}
@free trait MeanFS extends Operator with Mean {

  def Mean1[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]


  def Mean6[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]

}
@free trait ReduceLogSumExpFS extends Operator with ReduceLogSumExp {

  def ReduceLogSumExp1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait PadFS extends Operator with Pad {

  def Pad1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,mode : Option[(Tensor[VV])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : FS[(T[VV])]


  def Pad2[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,mode : Option[(Tensor[VV])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait LoopIndexTensorFS extends Operator with LoopIndexTensor {

  def LoopIndexTensor1[VV : Numeric:ClassTag](name: String,T: T[VV], Tname: String, loop_idx: T[VV], loop_idxname: String,axis : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait RandomUniformLikeFS extends Operator with RandomUniformLike {

  def RandomUniformLike1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait PReluFS extends Operator with PRelu {

  def PRelu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, slope: T[VV], slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def PRelu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, slope: T[VV], slopename: String)
    : FS[(T[VV])]


  def PRelu7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, slope: T[VV], slopename: String)
    : FS[(T[VV])]

}
@free trait IfFS extends Operator with If {

  def If1[VV : Numeric:ClassTag](name: String,cond: T[VV], condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : FS[(T[VV])]

}
@free trait ReduceMinFS extends Operator with ReduceMin {

  def ReduceMin1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait SoftsignFS extends Operator with Softsign {

  def Softsign1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait ConstantFillFS extends Operator with ConstantFill {

  def ConstantFill1[VV : Numeric:ClassTag](name: String,input: Option[T[VV]] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait LogSoftmaxFS extends Operator with LogSoftmax {

  def LogSoftmax1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait LpPoolFS extends Operator with LpPool {

  def LpPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def LpPool2[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(T[VV])]

}
@free trait SinFS extends Operator with Sin {

  def Sin7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait ReluFS extends Operator with Relu {

  def Relu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Relu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait SeluFS extends Operator with Selu {

  def Selu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : FS[(T[VV])]


  def Selu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait NotFS extends Operator with Not {

  def Not1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait AndFS extends Operator with And {

  def And1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def And7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait SizeFS extends Operator with Size {

  def Size1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String)
    : FS[(T[VV])]

}
@free trait FlattenFS extends Operator with Flatten {

  def Flatten1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait EqualFS extends Operator with Equal {

  def Equal1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Equal7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait DropoutFS extends Operator with Dropout {

  def Dropout1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(T[VV], T[VV])]


  def Dropout6[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(T[VV], T[VV])]


  def Dropout7[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,ratio : Option[(Int)] = None)
    : FS[(T[VV], T[VV])]

}
@free trait RandomNormalLikeFS extends Operator with RandomNormalLike {

  def RandomNormalLike1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait ScaledTanhFS extends Operator with ScaledTanh {

  def ScaledTanh1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait ParametricSoftplusFS extends Operator with ParametricSoftplus {

  def ParametricSoftplus1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait MaxFS extends Operator with Max {

  def Max1[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]


  def Max6[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]

}
@free trait SliceFS extends Operator with Slice {

  def Slice1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : FS[(T[VV])]

}
@free trait MinFS extends Operator with Min {

  def Min1[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]


  def Min6[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]

}
@free trait GRUUnitFS extends Operator with GRUUnit {

  def GRUUnit1[VV : Numeric:ClassTag](name: String,hidden_prev: T[VV], hidden_prevname: String, gates: T[VV], gatesname: String, seq_lengths: T[VV], seq_lengthsname: String, t: T[VV], tname: String,drop_states : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait SqrtFS extends Operator with Sqrt {

  def Sqrt1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Sqrt6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait MatMulFS extends Operator with MatMul {

  def MatMul1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait OrFS extends Operator with Or {

  def Or1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Or7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait ThresholdedReluFS extends Operator with ThresholdedRelu {

  def ThresholdedRelu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait AtanFS extends Operator with Atan {

  def Atan7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait NegFS extends Operator with Neg {

  def Neg1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Neg6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait LRNFS extends Operator with LRN {

  def LRN1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : FS[(T[VV])]

}
@free trait GRUFS extends Operator with GRU {

  def GRU1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[T[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(T[VV], T[VV])]


  def GRU3[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[T[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(T[VV], T[VV])]


  def GRU7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[T[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : FS[(T[VV], T[VV])]

}
@free trait LeakyReluFS extends Operator with LeakyRelu {

  def LeakyRelu1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def LeakyRelu6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait MaxRoiPoolFS extends Operator with MaxRoiPool {

  def MaxRoiPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, rois: T[VV], roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait HardmaxFS extends Operator with Hardmax {

  def Hardmax1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait ArgMaxFS extends Operator with ArgMax {

  def ArgMax1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long])]

}
@free trait RandomUniformFS extends Operator with RandomUniform {

  def RandomUniform1[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]

}
@free trait ReduceL2FS extends Operator with ReduceL2 {

  def ReduceL21[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait SigmoidFS extends Operator with Sigmoid {

  def Sigmoid1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Sigmoid6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait SoftplusFS extends Operator with Softplus {

  def Softplus1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait SqueezeFS extends Operator with Squeeze {

  def Squeeze1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : (Seq[String]))
    : FS[(T[VV])]

}
@free trait SubFS extends Operator with Sub {

  def Sub1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Sub6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Sub7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait SpaceToDepthFS extends Operator with SpaceToDepth {

  def SpaceToDepth1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,blocksize : (String))
    : FS[(T[VV])]

}
@free trait ConvTransposeFS extends Operator with ConvTranspose {

  def ConvTranspose1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String,B: Option[T[VV]] = None,auto_pad : Option[(Tensor[VV])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(T[VV])]

}
@free trait AddFS extends Operator with Add {

  def Add1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Add6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Add7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait TileFS extends Operator with Tile {

  def Tile1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String, tiles: T[VV], tilesname: String, axis: T[VV], axisname: String)
    : FS[(T[VV])]


  def Tile6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String, repeats: T[VV], repeatsname: String)
    : FS[(T[VV])]

}
@free trait TanhFS extends Operator with Tanh {

  def Tanh1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Tanh6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait GatherFS extends Operator with Gather {

  def Gather1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String, indices: T[VV], indicesname: String,axis : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait AveragePoolFS extends Operator with AveragePool {

  def AveragePool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def AveragePool7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(T[VV])]

}
@free trait ReciprocalFS extends Operator with Reciprocal {

  def Reciprocal1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Reciprocal6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait AcosFS extends Operator with Acos {

  def Acos7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait CosFS extends Operator with Cos {

  def Cos7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait InstanceNormalizationFS extends Operator with InstanceNormalization {

  def InstanceNormalization1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : FS[(T[VV])]


  def InstanceNormalization6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String,epsilon : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait ReduceSumFS extends Operator with ReduceSum {

  def ReduceSum1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait CastFS extends Operator with Cast {

  def Cast1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,to : (Tensor[VV]))
    : FS[(T[VV])]


  def Cast6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,to : (String))
    : FS[(T[VV])]

}
@free trait DepthToSpaceFS extends Operator with DepthToSpace {

  def DepthToSpace1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,blocksize : (String))
    : FS[(T[VV])]

}
@free trait ReduceLogSumFS extends Operator with ReduceLogSum {

  def ReduceLogSum1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait AbsFS extends Operator with Abs {

  def Abs1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Abs6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait ConcatFS extends Operator with Concat {

  def Concat1[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]


  def Concat4[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]

}
@free trait ReshapeFS extends Operator with Reshape {

  def Reshape1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Reshape5[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String, shape: Tensor[Long], shapename: String)
    : FS[(T[VV])]

}
@free trait ReduceSumSquareFS extends Operator with ReduceSumSquare {

  def ReduceSumSquare1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait DivFS extends Operator with Div {

  def Div1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Div6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Div7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait GreaterFS extends Operator with Greater {

  def Greater1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Greater7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait UpsampleFS extends Operator with Upsample {

  def Upsample1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,height_scaleAttr : (Int),mode : Option[(Tensor[VV])] = None,width_scaleAttr : (Int))
    : FS[(T[VV])]


  def Upsample7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,mode : Option[(Tensor[VV])] = None,scaleAttrs : (Seq[Int]))
    : FS[(T[VV])]

}
@free trait ReduceMeanFS extends Operator with ReduceMean {

  def ReduceMean1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait PowFS extends Operator with Pow {

  def Pow1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, Y: T[VV], Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Pow7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, Y: T[VV], Yname: String)
    : FS[(T[VV])]

}
@free trait BatchNormalizationFS extends Operator with BatchNormalization {

  def BatchNormalization1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String, mean: T[VV], meanname: String, someVar: T[VV], varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(T[VV], T[VV], T[VV], T[VV], T[VV])]


  def BatchNormalization6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String, mean: T[VV], meanname: String, someVar: T[VV], varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(T[VV], T[VV], T[VV], T[VV], T[VV])]


  def BatchNormalization7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, scale: T[VV], scalename: String, B: T[VV], Bname: String, mean: T[VV], meanname: String, someVar: T[VV], varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(T[VV], T[VV], T[VV], T[VV], T[VV])]

}
@free trait FloorFS extends Operator with Floor {

  def Floor1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Floor6[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String)
    : FS[(T[VV])]

}
@free trait ConstantFS extends Operator with Constant {

  def Constant1[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]

}
@free trait MeanVarianceNormalizationFS extends Operator with MeanVarianceNormalization {

  def MeanVarianceNormalization1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait LSTMFS extends Operator with LSTM {

  def LSTM1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[T[VV]] = None, initial_h: Option[T[VV]] = None, initial_c: Option[T[VV]] = None, P: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(T[VV], T[VV], T[VV])]


  def LSTM7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[T[VV]] = None, initial_h: Option[T[VV]] = None, initial_c: Option[T[VV]] = None, P: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : FS[(T[VV], T[VV], T[VV])]

}
@free trait LogFS extends Operator with Log {

  def Log1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Log6[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String)
    : FS[(T[VV])]

}
@free trait GivenTensorFillFS extends Operator with GivenTensorFill {

  def GivenTensorFill1[VV : Numeric:ClassTag](name: String,shapeInput: Option[T[VV]] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : FS[(T[VV])]

}
@free trait ReduceProdFS extends Operator with ReduceProd {

  def ReduceProd1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait MaxPoolFS extends Operator with MaxPool {

  def MaxPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,auto_pad : Option[(Tensor[VV])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(T[VV])]

}
@free trait ShapeFS extends Operator with Shape {

  def Shape1[VV : Numeric:ClassTag](name: String,data: T[VV], dataname: String)
    : FS[(T[VV])]

}
@free trait MulFS extends Operator with Mul {

  def Mul1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Mul6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Mul7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait AffineFS extends Operator with Affine {

  def Affine1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait GemmFS extends Operator with Gemm {

  def Gemm1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String, C: T[VV], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(T[VV])]


  def Gemm6[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String, C: T[VV], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(T[VV])]


  def Gemm7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String, C: T[VV], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait ImageScalerFS extends Operator with ImageScaler {

  def ImageScaler1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait ConvFS extends Operator with Conv {

  def Conv1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String,B: Option[T[VV]] = None,auto_pad : Option[(Tensor[VV])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(T[VV])]

}
@free trait RandomNormalFS extends Operator with RandomNormal {

  def RandomNormal1[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]

}
@free trait LoopFS extends Operator with Loop {

  def Loop1[VV : Numeric:ClassTag](name: String,M: T[VV], Mname: String, cond: T[VV], condname: String,body : (Seq[Float]))
    : FS[(T[VV])]

}
@free trait MultinomialFS extends Operator with Multinomial {

  def Multinomial7[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : FS[(T[VV])]

}
@free trait RNNFS extends Operator with RNN {

  def RNN1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[T[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(T[VV], T[VV])]


  def RNN7[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String, W: T[VV], Wname: String, R: T[VV], Rname: String,B: Option[T[VV]] = None, sequence_lens: Option[T[VV]] = None, initial_h: Option[T[VV]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[VV]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[VV])] = None,hidden_size : Option[(String)] = None)
    : FS[(T[VV], T[VV])]

}
@free trait SumFS extends Operator with Sum {

  def Sum1[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]


  def Sum6[VV : Numeric:ClassTag](name: String)
    : FS[(T[VV])]

}
@free trait SplitFS extends Operator with Split {

  def Split1[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,split: Option[T[VV]] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(T[VV])]


  def Split2[VV : Numeric:ClassTag](name: String,input: T[VV], inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(T[VV])]

}
@free trait LessFS extends Operator with Less {

  def Less1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Less7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}
@free trait GlobalLpPoolFS extends Operator with GlobalLpPool {

  def GlobalLpPool1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,p : Option[(Int)] = None)
    : FS[(T[VV])]


  def GlobalLpPool2[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,p : Option[(String)] = None)
    : FS[(T[VV])]

}
@free trait TopKFS extends Operator with TopK {

  def TopK1[VV : Numeric:ClassTag](name: String,X: T[VV], Xname: String,axis : Option[(String)] = None,k : (String))
    : FS[(T[VV], T[VV])]

}
@free trait XorFS extends Operator with Xor {

  def Xor1[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(T[VV])]


  def Xor7[VV : Numeric:ClassTag](name: String,A: T[VV], Aname: String, B: T[VV], Bname: String)
    : FS[(T[VV])]

}}
