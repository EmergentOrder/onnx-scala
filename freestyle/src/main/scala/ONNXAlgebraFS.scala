package org.emergentorder

import freestyle.free._
import freestyle.free.implicits._
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
@free trait TanhFS extends Operator with Tanh {

  def Tanh1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Tanh6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait TileFS extends Operator with Tile {

  def Tile1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String, tiles: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], tilesname: String, axis: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], axisname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Tile6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String, repeats: Tensor[Long], repeatsname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SpaceToDepthFS extends Operator with SpaceToDepth {

  def SpaceToDepth1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,blocksize : (String))
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ImageScalerFS extends Operator with ImageScaler {

  def ImageScaler1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait TanFS extends Operator with Tan {

  def Tan7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ThresholdedReluFS extends Operator with ThresholdedRelu {

  def ThresholdedRelu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SqrtFS extends Operator with Sqrt {

  def Sqrt1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Sqrt6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ReshapeFS extends Operator with Reshape {

  def Reshape1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Reshape5(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String, shape: Tensor[Long], shapename: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LogSoftmaxFS extends Operator with LogSoftmax {

  def LogSoftmax1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait CropFS extends Operator with Crop {

  def Crop1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SoftplusFS extends Operator with Softplus {

  def Softplus1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SoftsignFS extends Operator with Softsign {

  def Softsign1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait CeilFS extends Operator with Ceil {

  def Ceil1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Ceil6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ZipMapFS extends Operator with ZipMap {

  def ZipMap1(name: String,X: Tensor[Float], Xname: String,classlabels_int64s : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[_]])] = None)
    : FS[(Seq[Map[String, Float]] |: Seq[Map[Long, Float]])]

}
@free trait ImputerFS extends Operator with Imputer {

  def Imputer1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,imputed_value_floats : Option[(Seq[Int])] = None,imputed_value_int64s : Option[(Seq[String])] = None,replaced_value_float : Option[(Int)] = None,replaced_value_int64 : Option[(String)] = None)
    : FS[(Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int])]

}
@free trait ReduceSumFS extends Operator with ReduceSum {

  def ReduceSum1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait AveragePoolFS extends Operator with AveragePool {

  def AveragePool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def AveragePool7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait RNNFS extends Operator with RNN {

  def RNN1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def RNN7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait HardmaxFS extends Operator with Hardmax {

  def Hardmax1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SqueezeFS extends Operator with Squeeze {

  def Squeeze1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], dataname: String,axes : (Seq[String]))
    : FS[(Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait CastMapFS extends Operator with CastMap {

  def CastMap1(name: String,X: Map[Long, String] |: Map[Long, Float], Xname: String,cast_to : Option[(Tensor[_])] = None,map_form : Option[(Tensor[_])] = None,max_map : Option[(String)] = None)
    : FS[(Tensor[String] |: Tensor[Float] |: Tensor[Long])]

}
@free trait ClipFS extends Operator with Clip {

  def Clip1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Clip6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LpNormalizationFS extends Operator with LpNormalization {

  def LpNormalization1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait CosFS extends Operator with Cos {

  def Cos7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ArrayFeatureExtractorFS extends Operator with ArrayFeatureExtractor {

  def ArrayFeatureExtractor1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int] |: Tensor[String], Xname: String, Y: Tensor[Long], Yname: String)
    : FS[(Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int] |: Tensor[String])]

}
@free trait ParametricSoftplusFS extends Operator with ParametricSoftplus {

  def ParametricSoftplus1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SizeFS extends Operator with Size {

  def Size1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[Boolean], dataname: String)
    : FS[(Tensor[Long])]

}
@free trait AndFS extends Operator with And {

  def And1(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Boolean])]


  def And7(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String)
    : FS[(Tensor[Boolean])]

}
@free trait GemmFS extends Operator with Gemm {

  def Gemm1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, C: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Gemm6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, C: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Gemm7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, C: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait GRUUnitFS extends Operator with GRUUnit {

  def GRUUnit1(name: String,hidden_prev: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], hidden_prevname: String, gates: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], gatesname: String, seq_lengths: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], seq_lengthsname: String, t: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], tname: String,drop_states : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ReduceSumSquareFS extends Operator with ReduceSumSquare {

  def ReduceSumSquare1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait RandomUniformLikeFS extends Operator with RandomUniformLike {

  def RandomUniformLike1(name: String,input: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait MultinomialFS extends Operator with Multinomial {

  def Multinomial7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[Int] |: Tensor[Long])]

}
@free trait GlobalAveragePoolFS extends Operator with GlobalAveragePool {

  def GlobalAveragePool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait AffineFS extends Operator with Affine {

  def Affine1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait NegFS extends Operator with Neg {

  def Neg1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Float] |: Tensor[Int] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Float] |: Tensor[Int] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Double])]


  def Neg6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Float] |: Tensor[Int] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Float] |: Tensor[Int] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Double])]

}
@free trait FeatureVectorizerFS extends Operator with FeatureVectorizer {

  def FeatureVectorizer1(name: String)
    : FS[(Tensor[Float])]

}
@free trait TopKFS extends Operator with TopK {

  def TopK1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,axis : Option[(String)] = None,k : (String))
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Long])]

}
@free trait OrFS extends Operator with Or {

  def Or1(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Boolean])]


  def Or7(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String)
    : FS[(Tensor[Boolean])]

}
@free trait GatherFS extends Operator with Gather {

  def Gather1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], dataname: String, indices: Tensor[Int] |: Tensor[Long], indicesname: String,axis : Option[(String)] = None)
    : FS[(Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait BatchNormalizationFS extends Operator with BatchNormalization {

  def BatchNormalization1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, mean: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], meanname: String, someVar: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def BatchNormalization6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, mean: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], meanname: String, someVar: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def BatchNormalization7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, mean: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], meanname: String, someVar: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait TransposeFS extends Operator with Transpose {

  def Transpose1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,perm : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait GlobalLpPoolFS extends Operator with GlobalLpPool {

  def GlobalLpPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,p : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def GlobalLpPool2(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,p : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait IfFS extends Operator with If {

  def If1(name: String,cond: Tensor[Boolean], condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : FS[(Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait SoftmaxFS extends Operator with Softmax {

  def Softmax1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SinFS extends Operator with Sin {

  def Sin7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait TreeEnsembleRegressorFS extends Operator with TreeEnsembleRegressor {

  def TreeEnsembleRegressor1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,aggregate_function : Option[(Tensor[_])] = None,base_values : Option[(Seq[Int])] = None,n_targets : Option[(String)] = None,nodes_falsenodeids : Option[(Seq[String])] = None,nodes_featureids : Option[(Seq[String])] = None,nodes_hitrates : Option[(Seq[Int])] = None,nodes_missing_value_tracks_true : Option[(Seq[String])] = None,nodes_modes : Option[(Seq[Tensor[_]])] = None,nodes_nodeids : Option[(Seq[String])] = None,nodes_treeids : Option[(Seq[String])] = None,nodes_truenodeids : Option[(Seq[String])] = None,nodes_values : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[_])] = None,target_ids : Option[(Seq[String])] = None,target_nodeids : Option[(Seq[String])] = None,target_treeids : Option[(Seq[String])] = None,target_weights : Option[(Seq[Int])] = None)
    : FS[(Tensor[Float])]

}
@free trait ConvFS extends Operator with Conv {

  def Conv1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,auto_pad : Option[(Tensor[_])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SVMClassifierFS extends Operator with SVMClassifier {

  def SVMClassifier1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,classlabels_ints : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[_]])] = None,coefficients : Option[(Seq[Int])] = None,kernel_params : Option[(Seq[Int])] = None,kernel_type : Option[(Tensor[_])] = None,post_transform : Option[(Tensor[_])] = None,prob_a : Option[(Seq[Int])] = None,prob_b : Option[(Seq[Int])] = None,rho : Option[(Seq[Int])] = None,support_vectors : Option[(Seq[Int])] = None,vectors_per_class : Option[(Seq[String])] = None)
    : FS[(Tensor[String] |: Tensor[Long], Tensor[Float])]

}
@free trait AbsFS extends Operator with Abs {

  def Abs1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Abs6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait MaxPoolFS extends Operator with MaxPool {

  def MaxPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SumFS extends Operator with Sum {

  def Sum1(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Sum6(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait DropoutFS extends Operator with Dropout {

  def Dropout1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Dropout6(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Dropout7(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,ratio : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LabelEncoderFS extends Operator with LabelEncoder {

  def LabelEncoder1(name: String,X: Tensor[String] |: Tensor[Long], Xname: String,classes_strings : Option[(Seq[Tensor[_]])] = None,default_int64 : Option[(String)] = None,default_string : Option[(Tensor[_])] = None)
    : FS[(Tensor[String] |: Tensor[Long])]

}
@free trait CastFS extends Operator with Cast {

  def Cast1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Boolean], inputname: String,to : (Tensor[_]))
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Boolean])]


  def Cast6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Boolean], inputname: String,to : (String))
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Boolean])]

}
@free trait ReduceMaxFS extends Operator with ReduceMax {

  def ReduceMax1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LeakyReluFS extends Operator with LeakyRelu {

  def LeakyRelu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def LeakyRelu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ReduceMeanFS extends Operator with ReduceMean {

  def ReduceMean1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait NormalizerFS extends Operator with Normalizer {

  def Normalizer1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,norm : Option[(Tensor[_])] = None)
    : FS[(Tensor[Float])]

}
@free trait ScalerFS extends Operator with Scaler {

  def Scaler1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,offset : Option[(Seq[Int])] = None,scaleAttr : Option[(Seq[Int])] = None)
    : FS[(Tensor[Float])]

}
@free trait IdentityFS extends Operator with Identity {

  def Identity1(name: String,input: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], inputname: String)
    : FS[(Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait ReduceLogSumExpFS extends Operator with ReduceLogSumExp {

  def ReduceLogSumExp1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ExpFS extends Operator with Exp {

  def Exp1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Exp6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ReciprocalFS extends Operator with Reciprocal {

  def Reciprocal1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Reciprocal6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait InstanceNormalizationFS extends Operator with InstanceNormalization {

  def InstanceNormalization1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def InstanceNormalization6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,epsilon : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait DivFS extends Operator with Div {

  def Div1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Div6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Div7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait EqualFS extends Operator with Equal {

  def Equal1(name: String,A: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long], Aname: String, B: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Boolean])]


  def Equal7(name: String,A: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long], Aname: String, B: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long], Bname: String)
    : FS[(Tensor[Boolean])]

}
@free trait ConstantFS extends Operator with Constant {

  def Constant1(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait BinarizerFS extends Operator with Binarizer {

  def Binarizer1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,threshold : Option[(Int)] = None)
    : FS[(Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int])]

}
@free trait ReduceMinFS extends Operator with ReduceMin {

  def ReduceMin1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ArgMaxFS extends Operator with ArgMax {

  def ArgMax1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long])]

}
@free trait LRNFS extends Operator with LRN {

  def LRN1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait GreaterFS extends Operator with Greater {

  def Greater1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Boolean])]


  def Greater7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : FS[(Tensor[Boolean])]

}
@free trait UnsqueezeFS extends Operator with Unsqueeze {

  def Unsqueeze1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], dataname: String,axes : (Seq[String]))
    : FS[(Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait CategoryMapperFS extends Operator with CategoryMapper {

  def CategoryMapper1(name: String,X: Tensor[String] |: Tensor[Long], Xname: String,cats_int64s : Option[(Seq[String])] = None,cats_strings : Option[(Seq[Tensor[_]])] = None,default_int64 : Option[(String)] = None,default_string : Option[(Tensor[_])] = None)
    : FS[(Tensor[String] |: Tensor[Long])]

}
@free trait TreeEnsembleClassifierFS extends Operator with TreeEnsembleClassifier {

  def TreeEnsembleClassifier1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,base_values : Option[(Seq[Int])] = None,class_ids : Option[(Seq[String])] = None,class_nodeids : Option[(Seq[String])] = None,class_treeids : Option[(Seq[String])] = None,class_weights : Option[(Seq[Int])] = None,classlabels_int64s : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[_]])] = None,nodes_falsenodeids : Option[(Seq[String])] = None,nodes_featureids : Option[(Seq[String])] = None,nodes_hitrates : Option[(Seq[Int])] = None,nodes_missing_value_tracks_true : Option[(Seq[String])] = None,nodes_modes : Option[(Seq[Tensor[_]])] = None,nodes_nodeids : Option[(Seq[String])] = None,nodes_treeids : Option[(Seq[String])] = None,nodes_truenodeids : Option[(Seq[String])] = None,nodes_values : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[_])] = None)
    : FS[(Tensor[String] |: Tensor[Long], Tensor[Float])]

}
@free trait RandomNormalLikeFS extends Operator with RandomNormalLike {

  def RandomNormalLike1(name: String,input: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SigmoidFS extends Operator with Sigmoid {

  def Sigmoid1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Sigmoid6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LSTMFS extends Operator with LSTM {

  def LSTM1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, initial_c: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, P: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def LSTM7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, initial_c: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, P: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait MeanFS extends Operator with Mean {

  def Mean1(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Mean6(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ScaledTanhFS extends Operator with ScaledTanh {

  def ScaledTanh1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ReduceL1FS extends Operator with ReduceL1 {

  def ReduceL11(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait AtanFS extends Operator with Atan {

  def Atan7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LinearRegressorFS extends Operator with LinearRegressor {

  def LinearRegressor1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,coefficients : Option[(Seq[Int])] = None,intercepts : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[_])] = None,targets : Option[(String)] = None)
    : FS[(Tensor[Float])]

}
@free trait NotFS extends Operator with Not {

  def Not1(name: String,X: Tensor[Boolean], Xname: String)
    : FS[(Tensor[Boolean])]

}
@free trait LogFS extends Operator with Log {

  def Log1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Log6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ReluFS extends Operator with Relu {

  def Relu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Relu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LoopIndexTensorFS extends Operator with LoopIndexTensor {

  def LoopIndexTensor1(name: String,T: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], Tname: String, loop_idx: Int, loop_idxname: String,axis : Option[(String)] = None)
    : FS[(Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait ReduceProdFS extends Operator with ReduceProd {

  def ReduceProd1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait GRUFS extends Operator with GRU {

  def GRU1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def GRU3(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def GRU7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LessFS extends Operator with Less {

  def Less1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Boolean])]


  def Less7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : FS[(Tensor[Boolean])]

}
@free trait MeanVarianceNormalizationFS extends Operator with MeanVarianceNormalization {

  def MeanVarianceNormalization1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ConstantFillFS extends Operator with ConstantFill {

  def ConstantFill1(name: String,input: Option[Tensor[Float] |: Tensor[Int] |: Tensor[Long] |: Tensor[Boolean]] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : FS[(Tensor[Float] |: Tensor[Int] |: Tensor[Long] |: Tensor[Boolean])]

}
@free trait SubFS extends Operator with Sub {

  def Sub1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Sub6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Sub7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ScaleFS extends Operator with Scale {

  def Scale1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait AddFS extends Operator with Add {

  def Add1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Add6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Add7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait RandomNormalFS extends Operator with RandomNormal {

  def RandomNormal1(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait PadFS extends Operator with Pad {

  def Pad1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,mode : Option[(Tensor[_])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Pad2(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,mode : Option[(Tensor[_])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ReduceL2FS extends Operator with ReduceL2 {

  def ReduceL21(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait MinFS extends Operator with Min {

  def Min1(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Min6(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait FloorFS extends Operator with Floor {

  def Floor1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Floor6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ShapeFS extends Operator with Shape {

  def Shape1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[Boolean], dataname: String)
    : FS[(Tensor[Long])]

}
@free trait SliceFS extends Operator with Slice {

  def Slice1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait DepthToSpaceFS extends Operator with DepthToSpace {

  def DepthToSpace1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,blocksize : (String))
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait PowFS extends Operator with Pow {

  def Pow1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, Y: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Pow7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, Y: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Yname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait DictVectorizerFS extends Operator with DictVectorizer {

  def DictVectorizer1(name: String,X: Map[String, Long] |: Map[Long, String] |: Map[Long, Float] |: Map[Long, Double] |: Map[String, Float] |: Map[String, Double], Xname: String,int64_vocabulary : Option[(Seq[String])] = None,string_vocabulary : Option[(Seq[Tensor[_]])] = None)
    : FS[(Tensor[Long] |: Tensor[Float] |: Tensor[Double] |: Tensor[String])]

}
@free trait ReduceLogSumFS extends Operator with ReduceLogSum {

  def ReduceLogSum1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait XorFS extends Operator with Xor {

  def Xor1(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Boolean])]


  def Xor7(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String)
    : FS[(Tensor[Boolean])]

}
@free trait UpsampleFS extends Operator with Upsample {

  def Upsample1(name: String,X: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], Xname: String,height_scaleAttr : (Int),mode : Option[(Tensor[_])] = None,width_scaleAttr : (Int))
    : FS[(Tensor[Boolean] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]


  def Upsample7(name: String,X: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], Xname: String,mode : Option[(Tensor[_])] = None,scaleAttrs : (Seq[Int]))
    : FS[(Tensor[Boolean] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait OneHotEncoderFS extends Operator with OneHotEncoder {

  def OneHotEncoder1(name: String,X: Tensor[String] |: Tensor[Long] |: Tensor[Int] |: Tensor[Float] |: Tensor[Double], Xname: String,cats_int64s : Option[(Seq[String])] = None,cats_strings : Option[(Seq[Tensor[_]])] = None,zeros : Option[(String)] = None)
    : FS[(Tensor[Float])]

}
@free trait GivenTensorFillFS extends Operator with GivenTensorFill {

  def GivenTensorFill1(name: String,shapeInput: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait MatMulFS extends Operator with MatMul {

  def MatMul1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait MulFS extends Operator with Mul {

  def Mul1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Mul6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Mul7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ArgMinFS extends Operator with ArgMin {

  def ArgMin1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : FS[(Tensor[Long])]

}
@free trait SVMRegressorFS extends Operator with SVMRegressor {

  def SVMRegressor1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,coefficients : Option[(Seq[Int])] = None,kernel_params : Option[(Seq[Int])] = None,kernel_type : Option[(Tensor[_])] = None,n_supports : Option[(String)] = None,one_class : Option[(String)] = None,post_transform : Option[(Tensor[_])] = None,rho : Option[(Seq[Int])] = None,support_vectors : Option[(Seq[Int])] = None)
    : FS[(Tensor[Float])]

}
@free trait MaxRoiPoolFS extends Operator with MaxRoiPool {

  def MaxRoiPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, rois: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait ConvTransposeFS extends Operator with ConvTranspose {

  def ConvTranspose1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,auto_pad : Option[(Tensor[_])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait RandomUniformFS extends Operator with RandomUniform {

  def RandomUniform1(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LoopFS extends Operator with Loop {

  def Loop1(name: String,M: Long, Mname: String, cond: Boolean, condname: String,body : (Seq[Float]))
    : FS[(Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait ConcatFS extends Operator with Concat {

  def Concat1(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]


  def Concat4(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])]

}
@free trait PReluFS extends Operator with PRelu {

  def PRelu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, slope: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def PRelu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, slope: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], slopename: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def PRelu7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, slope: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], slopename: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LpPoolFS extends Operator with LpPool {

  def LpPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def LpPool2(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait AsinFS extends Operator with Asin {

  def Asin7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait AcosFS extends Operator with Acos {

  def Acos7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SplitFS extends Operator with Split {

  def Split1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,split: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Split2(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait MaxFS extends Operator with Max {

  def Max1(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Max6(name: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait FlattenFS extends Operator with Flatten {

  def Flatten1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait LinearClassifierFS extends Operator with LinearClassifier {

  def LinearClassifier1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,classlabels_ints : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[_]])] = None,coefficients : (Seq[Int]),intercepts : Option[(Seq[Int])] = None,multi_class : Option[(String)] = None,post_transform : Option[(Tensor[_])] = None)
    : FS[(Tensor[String] |: Tensor[Long], Tensor[Float])]

}
@free trait GlobalMaxPoolFS extends Operator with GlobalMaxPool {

  def GlobalMaxPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait SeluFS extends Operator with Selu {

  def Selu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Selu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait EluFS extends Operator with Elu {

  def Elu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def Elu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}
@free trait HardSigmoidFS extends Operator with HardSigmoid {

  def HardSigmoid1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]


  def HardSigmoid6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : FS[(Tensor[Float16] |: Tensor[Float] |: Tensor[Double])]

}}
