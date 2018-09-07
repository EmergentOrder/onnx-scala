package org.emergentorder

import scala.{specialized => sp}
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.math.Numeric
import scala.reflect.ClassTag

package object onnx {
type |:[+A1, +A2] = Either[A1, A2]
  type Tensor[U] = Tuple2[Vector[U], Seq[Int]]
  trait Operator
trait DataSource {
  def inputData[VV:Numeric:ClassTag]: Tensor[VV]
  def getParams[VV:Numeric:ClassTag](name: String): Tensor[VV]
  def getAttributes[VV:Numeric:ClassTag](name: String): Tensor[VV]
}
trait ConvTranspose extends Operator {

  def ConvTranspose1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String,B: Option[Tensor[T]] = None,auto_pad : Option[(Tensor[_])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[T])

}
trait ReduceMin extends Operator {

  def ReduceMin1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait Hardmax extends Operator {

  def Hardmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None)
    : (Tensor[T])

}
trait Pad extends Operator {

  def Pad1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,mode : Option[(Tensor[_])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : (Tensor[T])


  def Pad2[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,mode : Option[(Tensor[_])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : (Tensor[T])

}
trait ReduceProd extends Operator {

  def ReduceProd1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait Clip extends Operator {

  def Clip1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : (Tensor[T])


  def Clip6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : (Tensor[T])

}
trait Identity extends Operator {

  def Identity1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait GivenTensorFill extends Operator {

  def GivenTensorFill1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,shapeInput: Option[Tensor[T]] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : (Tensor[T])

}
trait Neg extends Operator {

  def Neg1[T <: Float16 |: Float |: Double |: Float |: Int |: Byte |: Short |: Long |: Float16 |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Neg6[T <: Float16 |: Float |: Double |: Float |: Int |: Byte |: Short |: Long |: Float16 |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait Not extends Operator {

  def Not1[T <: Boolean : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait DepthToSpace extends Operator {

  def DepthToSpace1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,blocksize : (String))
    : (Tensor[T])

}
trait RandomNormal extends Operator {

  def RandomNormal1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])

}
trait ParametricSoftplus extends Operator {

  def ParametricSoftplus1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (Tensor[T])

}
trait BatchNormalization extends Operator {

  def BatchNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])


  def BatchNormalization6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])


  def BatchNormalization7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])

}
trait RandomUniformLike extends Operator {

  def RandomUniformLike1[T1 <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T2 <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : (Tensor[T2])

}
trait Split extends Operator {

  def Split1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,split: Option[Tensor[T]] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Split2[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : (Tensor[T])

}
trait ReduceSumSquare extends Operator {

  def ReduceSumSquare1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait Expand extends Operator {

  def Expand8[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, shape: Tensor[Long], shapename: String)
    : (Tensor[T])

}
trait TopK extends Operator {

  def TopK1[T <: Float16 |: Float |: Double : Numeric:ClassTag,I <: Long : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,axis : Option[(String)] = None,k : (String))
    : (Tensor[T], Tensor[I])

}
trait InstanceNormalization extends Operator {

  def InstanceNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : (Tensor[T])


  def InstanceNormalization6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String,epsilon : Option[(Int)] = None)
    : (Tensor[T])

}
trait LogSoftmax extends Operator {

  def LogSoftmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None)
    : (Tensor[T])

}
trait Tan extends Operator {

  def Tan7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait Dropout extends Operator {

  def Dropout1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : (Tensor[T], Tensor[T])


  def Dropout6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : (Tensor[T], Tensor[T])


  def Dropout7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,ratio : Option[(Int)] = None)
    : (Tensor[T], Tensor[T])

}
trait SpaceToDepth extends Operator {

  def SpaceToDepth1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,blocksize : (String))
    : (Tensor[T])

}
trait Cos extends Operator {

  def Cos7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait Upsample extends Operator {

  def Upsample1[T <: Boolean |: Int |: Long |: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,height_scaleAttr : (Int),mode : Option[(Tensor[_])] = None,width_scaleAttr : (Int))
    : (Tensor[T])


  def Upsample7[T <: Boolean |: Int |: Long |: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,mode : Option[(Tensor[_])] = None,scaleAttrs : (Seq[Int]))
    : (Tensor[T])

}
trait GlobalLpPool extends Operator {

  def GlobalLpPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,p : Option[(Int)] = None)
    : (Tensor[T])


  def GlobalLpPool2[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,p : Option[(String)] = None)
    : (Tensor[T])

}
trait Elu extends Operator {

  def Elu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Elu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None)
    : (Tensor[T])

}
trait Conv extends Operator {

  def Conv1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String,B: Option[Tensor[T]] = None,auto_pad : Option[(Tensor[_])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[T])

}
trait ArgMax extends Operator {

  def ArgMax1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : (Tensor[Long])

}
trait Loop extends Operator {

  def Loop1[I <: Long : Numeric:ClassTag,B <: Boolean : Numeric:ClassTag,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,M: I, Mname: String, cond: B, condname: String,body : (Seq[Float]))
    : (Tensor[V])

}
trait And extends Operator {

  def And1[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T1])


  def And7[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T1])

}
trait Greater extends Operator {

  def Greater1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T1])


  def Greater7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T1])

}
trait Max extends Operator {

  def Max1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Max6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Max8[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])

}
trait Floor extends Operator {

  def Floor1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Floor6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait Gemm extends Operator {

  def Gemm1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (Tensor[T])


  def Gemm6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (Tensor[T])


  def Gemm7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (Tensor[T])

}
trait Div extends Operator {

  def Div1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Div6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T])


  def Div7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T])

}
trait Multinomial extends Operator {

  def Multinomial7[T1 <: Float16 |: Float |: Double : Numeric:ClassTag,T2 <: Int |: Long : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : (Tensor[T2])

}
trait Xor extends Operator {

  def Xor1[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T1])


  def Xor7[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T1])

}
trait ThresholdedRelu extends Operator {

  def ThresholdedRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None)
    : (Tensor[T])

}
trait Sqrt extends Operator {

  def Sqrt1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Sqrt6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait Log extends Operator {

  def Log1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Log6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait Mul extends Operator {

  def Mul1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Mul6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T])


  def Mul7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T])

}
trait ArgMin extends Operator {

  def ArgMin1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : (Tensor[Long])

}
trait Constant extends Operator {

  def Constant1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])

}
trait LpPool extends Operator {

  def LpPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[T])


  def LpPool2[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[T])

}
trait Transpose extends Operator {

  def Transpose1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,perm : Option[(Seq[String])] = None)
    : (Tensor[T])

}
trait ReduceMean extends Operator {

  def ReduceMean1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait GlobalMaxPool extends Operator {

  def GlobalMaxPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait PRelu extends Operator {

  def PRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def PRelu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String)
    : (Tensor[T])


  def PRelu7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String)
    : (Tensor[T])

}
trait Selu extends Operator {

  def Selu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : (Tensor[T])


  def Selu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : (Tensor[T])

}
trait Softmax extends Operator {

  def Softmax1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None)
    : (Tensor[T])

}
trait Slice extends Operator {

  def Slice1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : (Tensor[T])

}
trait Pow extends Operator {

  def Pow1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, Y: Tensor[T], Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T])


  def Pow7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, Y: Tensor[T], Yname: String)
    : (Tensor[T])

}
trait Softplus extends Operator {

  def Softplus1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait Tanh extends Operator {

  def Tanh1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Tanh6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait MatMul extends Operator {

  def MatMul1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T])

}
trait RNN extends Operator {

  def RNN1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (Tensor[T], Tensor[T])


  def RNN7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None)
    : (Tensor[T], Tensor[T])

}
trait MaxRoiPool extends Operator {

  def MaxRoiPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, rois: Tensor[T], roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : (Tensor[T])

}
trait Size extends Operator {

  def Size1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T1 <: Long : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String)
    : (Tensor[T1])

}
trait Cast extends Operator {

  def Cast1[T1 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag,T2 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,to : (Tensor[_]))
    : (Tensor[T2])


  def Cast6[T1 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag,T2 <: Float16 |: Float |: Double |: Byte |: Short |: Int |: Long |: UByte |: UShort |: UInt |: ULong |: Boolean : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,to : (String))
    : (Tensor[T2])

}
trait Softsign extends Operator {

  def Softsign1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait Concat extends Operator {

  def Concat1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Concat4[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String)
    : (Tensor[T])

}
trait Exp extends Operator {

  def Exp1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Exp6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait Crop extends Operator {

  def Crop1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : (Tensor[T])

}
trait Sigmoid extends Operator {

  def Sigmoid1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Sigmoid6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : (Tensor[T])

}
trait GlobalAveragePool extends Operator {

  def GlobalAveragePool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait ConstantFill extends Operator {

  def ConstantFill1[T1 <: Float |: Int |: Long |: Boolean : Numeric:ClassTag,T2 <: Float |: Int |: Long |: Boolean : Numeric:ClassTag](name: String,input: Option[Tensor[T1]] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : (Tensor[T2])

}
trait Add extends Operator {

  def Add1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Add6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T])


  def Add7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T])

}
trait GRU extends Operator {

  def GRU1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (Tensor[T], Tensor[T])


  def GRU3[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (Tensor[T], Tensor[T])


  def GRU7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : (Tensor[T], Tensor[T])

}
trait ReduceL2 extends Operator {

  def ReduceL21[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait Sum extends Operator {

  def Sum1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Sum6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Sum8[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])

}
trait ReduceL1 extends Operator {

  def ReduceL11[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait If extends Operator {

  def If1[B <: Boolean : Numeric:ClassTag,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,cond: Tensor[B], condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : (Tensor[V])

}
trait Reshape extends Operator {

  def Reshape1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Reshape5[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String, shape: Tensor[Long], shapename: String)
    : (Tensor[T])

}
trait MaxPool extends Operator {

  def MaxPool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[T])


  def MaxPool8[T <: Float16 |: Float |: Double : Numeric:ClassTag,I <: Long : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,storage_order : Option[(String)] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[T], Tensor[I])

}
trait LSTM extends Operator {

  def LSTM1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (Tensor[T], Tensor[T], Tensor[T])


  def LSTM7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Int : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : (Tensor[T], Tensor[T], Tensor[T])

}
trait LeakyRelu extends Operator {

  def LeakyRelu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def LeakyRelu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None)
    : (Tensor[T])

}
trait ReduceSum extends Operator {

  def ReduceSum1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait LpNormalization extends Operator {

  def LpNormalization1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : (Tensor[T])

}
trait Flatten extends Operator {

  def Flatten1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,axis : Option[(String)] = None)
    : (Tensor[T])

}
trait Scan extends Operator {

  def Scan8[I <: Long : Numeric:ClassTag,V <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,sequence_lens: Option[Tensor[I]] = None,body : (Seq[Float]),directions : Option[(Seq[String])] = None,num_scan_inputs : (String))
    : (Tensor[V])

}
trait Tile extends Operator {

  def Tile1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, tiles: Tensor[T], tilesname: String, axis: Tensor[T], axisname: String)
    : (Tensor[T])


  def Tile6[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T1 <: Long : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String, repeats: Tensor[T1], repeatsname: String)
    : (Tensor[T])

}
trait GRUUnit extends Operator {

  def GRUUnit1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,hidden_prev: Tensor[T], hidden_prevname: String, gates: Tensor[T], gatesname: String, seq_lengths: Tensor[T], seq_lengthsname: String, t: Tensor[T], tname: String,drop_states : Option[(String)] = None)
    : (Tensor[T])

}
trait Min extends Operator {

  def Min1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Min6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Min8[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])

}
trait ImageScaler extends Operator {

  def ImageScaler1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : (Tensor[T])

}
trait AveragePool extends Operator {

  def AveragePool1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[T])


  def AveragePool7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(Tensor[_])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[T])

}
trait ReduceLogSum extends Operator {

  def ReduceLogSum1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait ReduceMax extends Operator {

  def ReduceMax1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait Mean extends Operator {

  def Mean1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Mean6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])


  def Mean8[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])

}
trait Abs extends Operator {

  def Abs1[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Abs6[T <: Float16 |: Float |: Double |: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait Gather extends Operator {

  def Gather1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,Tind <: Int |: Long : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String, indices: Tensor[Tind], indicesname: String,axis : Option[(String)] = None)
    : (Tensor[T])

}
trait Sub extends Operator {

  def Sub1[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Sub6[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T])


  def Sub7[T <: Float16 |: Float |: Double |: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T])

}
trait Less extends Operator {

  def Less1[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T1])


  def Less7[T <: Float16 |: Float |: Double : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T1])

}
trait Reciprocal extends Operator {

  def Reciprocal1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Reciprocal6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait Atan extends Operator {

  def Atan7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait RandomNormalLike extends Operator {

  def RandomNormalLike1[T1 <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T2 <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T1], inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : (Tensor[T2])

}
trait Relu extends Operator {

  def Relu1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Relu6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait RandomUniform extends Operator {

  def RandomUniform1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String)
    : (Tensor[T])

}
trait ScaledTanh extends Operator {

  def ScaledTanh1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (Tensor[T])

}
trait Ceil extends Operator {

  def Ceil1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def Ceil6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String)
    : (Tensor[T])

}
trait Squeeze extends Operator {

  def Squeeze1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None)
    : (Tensor[T])

}
trait Scale extends Operator {

  def Scale1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String,scaleAttr : Option[(Int)] = None)
    : (Tensor[T])

}
trait Unsqueeze extends Operator {

  def Unsqueeze1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : (Seq[String]))
    : (Tensor[T])

}
trait Affine extends Operator {

  def Affine1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (Tensor[T])

}
trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp1[T <: UInt |: ULong |: Int |: Long |: Float16 |: Float |: Double : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[T])

}
trait Sin extends Operator {

  def Sin7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait Equal extends Operator {

  def Equal1[T <: Boolean |: Int |: Long : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T1])


  def Equal7[T <: Boolean |: Int |: Long : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T1])

}
trait HardSigmoid extends Operator {

  def HardSigmoid1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[T])


  def HardSigmoid6[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (Tensor[T])

}
trait LRN extends Operator {

  def LRN1[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,X: Tensor[T], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : (Tensor[T])

}
trait Asin extends Operator {

  def Asin7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait Acos extends Operator {

  def Acos7[T <: Float16 |: Float |: Double : Numeric:ClassTag](name: String,input: Tensor[T], inputname: String)
    : (Tensor[T])

}
trait Or extends Operator {

  def Or1[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[T1])


  def Or7[T <: Boolean : Numeric:ClassTag,T1 <: Boolean : Numeric:ClassTag](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
    : (Tensor[T1])

}
trait Shape extends Operator {

  def Shape1[T <: UByte |: UShort |: UInt |: ULong |: Byte |: Short |: Int |: Long |: Float16 |: Float |: Double |: String |: Boolean |: Complex[Float] |: Complex[Double] : Numeric:ClassTag,T1 <: Long : Numeric:ClassTag](name: String,data: Tensor[T], dataname: String)
    : (Tensor[T1])

}}
