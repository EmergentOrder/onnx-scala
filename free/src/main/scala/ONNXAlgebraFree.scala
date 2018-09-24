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
  def inputDataFree[T : Numeric:ClassTag:Field](implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): FS[Tensor[T]]
  def getParamsFree[T : Numeric:ClassTag:Field](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): FS[Tensor[T]]
  def getAttributesFree[T : Numeric:ClassTag:Field](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): FS[Tensor[T]]
}
@free trait AbsFree extends Operator with Abs {

  def Abs1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Abs6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AcosFree extends Operator with Acos {

  def Acos7Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AddFree extends Operator with Add {

  def Add1Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Add6Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Add7Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AffineFree extends Operator with Affine {

  def Affine1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AndFree extends Operator with And {

  def And1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def And7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait ArgMaxFree extends Operator with ArgMax {

  def ArgMax1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Long])]

}
@free trait ArgMinFree extends Operator with ArgMin {

  def ArgMin1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Long])]

}
@free trait ArrayFeatureExtractorFree extends Operator with ArrayFeatureExtractor {

  def ArrayFeatureExtractor1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, Y: Tensor[Long], Yname: String)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int TypeOr String)#check[T])    : FS[(Tensor[T])]

}
@free trait AsinFree extends Operator with Asin {

  def Asin7Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AtanFree extends Operator with Atan {

  def Atan7Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AveragePoolFree extends Operator with AveragePool {

  def AveragePool1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def AveragePool7Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait BatchNormalizationFree extends Operator with BatchNormalization {

  def BatchNormalization1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,consumed_inputs : (Array[Int]),epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization7Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String, mean: Tensor[T], meanname: String, someVar: Tensor[T], varname: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]

}
@free trait BinarizerFree extends Operator with Binarizer {

  def Binarizer1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,threshold : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[T])]

}
@free trait CastFree extends Operator with Cast {

  def Cast1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,to : (String))
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : FS[(Tensor[T2])]


  def Cast6Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,to : (Int))
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : FS[(Tensor[T2])]

}
@free trait CastMapFree extends Operator with CastMap {

  def CastMap1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: T1, Xname: String,cast_to : Option[(String)] = None,map_form : Option[(String)] = None,max_map : Option[(Int)] = None)
(implicit evT1:(UNil TypeOr Map[Long, String] TypeOr Map[Long, Float])#check[T1],evT2:(UNil TypeOr String TypeOr Float TypeOr Long)#check[T2])    : FS[(Tensor[T2])]

}
@free trait CategoryMapperFree extends Operator with CategoryMapper {

  def CategoryMapper1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2])]

}
@free trait CeilFree extends Operator with Ceil {

  def Ceil1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Ceil6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ClipFree extends Operator with Clip {

  def Clip1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Array[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Clip6Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,max : Option[(Float)] = None,min : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ConcatFree extends Operator with Concat {

  def Concat1Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Concat4Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait ConstantFillFree extends Operator with ConstantFill {

  def ConstantFill1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Option[Tensor[T1]] = None,dtype : Option[(Int)] = None,extra_shape : Option[(Array[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Array[Int])] = None,value : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Int TypeOr Long TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float TypeOr Int TypeOr Long TypeOr Boolean)#check[T2])    : FS[(Tensor[T2])]

}
@free trait ConstantFree extends Operator with Constant {

  def Constant1Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ConvFree extends Operator with Conv {

  def Conv1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String,B: Option[Tensor[T]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ConvTransposeFree extends Operator with ConvTranspose {

  def ConvTranspose1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String,B: Option[Tensor[T]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,output_padding : Option[(Array[Int])] = None,output_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait CosFree extends Operator with Cos {

  def Cos7Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait CropFree extends Operator with Crop {

  def Crop1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,border : Option[(Array[Int])] = None,scaleAttr : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait DepthToSpaceFree extends Operator with DepthToSpace {

  def DepthToSpace1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,blocksize : (Int))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait DictVectorizerFree extends Operator with DictVectorizer {

  def DictVectorizer1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: T1, Xname: String,int64_vocabulary : Option[(Array[Int])] = None,string_vocabulary : Option[(Array[String])] = None)
(implicit evT1:(UNil TypeOr Map[String, Long] TypeOr Map[Long, String] TypeOr Map[Long, Float] TypeOr Map[Long, Double] TypeOr Map[String, Float] TypeOr Map[String, Double])#check[T1],evT2:(UNil TypeOr Long TypeOr Float TypeOr Double TypeOr String)#check[T2])    : FS[(Tensor[T2])]

}
@free trait DivFree extends Operator with Div {

  def Div1Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Div6Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Div7Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait DropoutFree extends Operator with Dropout {

  def Dropout1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,consumed_inputs : Option[(Array[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T])]


  def Dropout6Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T])]


  def Dropout7Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T])]

}
@free trait EluFree extends Operator with Elu {

  def Elu1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Elu6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait EqualFree extends Operator with Equal {

  def Equal1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Equal7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait ExpFree extends Operator with Exp {

  def Exp1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Exp6Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ExpandFree extends Operator with Expand {

  def Expand8Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, shape: Tensor[Long], shapename: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait FeatureVectorizerFree extends Operator with FeatureVectorizer {

  def FeatureVectorizer1Free(name: String)
    : FS[(Tensor[Float])]

}
@free trait FlattenFree extends Operator with Flatten {

  def Flatten1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait FloorFree extends Operator with Floor {

  def Floor1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Floor6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GRUFree extends Operator with GRU {

  def GRU1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]


  def GRU3Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]


  def GRU7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]

}
@free trait GRUUnitFree extends Operator with GRUUnit {

  def GRUUnit1Free[@sp T : Numeric:ClassTag:Field](name: String,hidden_prev: Tensor[T], hidden_prevname: String, gates: Tensor[T], gatesname: String, seq_lengths: Tensor[T], seq_lengthsname: String, t: Tensor[T], tname: String,drop_states : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GatherFree extends Operator with Gather {

  def Gather1Free[@sp T : Numeric:ClassTag:Field,@sp Tind : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String, indices: Tensor[Tind], indicesname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : FS[(Tensor[T])]

}
@free trait GemmFree extends Operator with Gemm {

  def Gemm1Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Gemm6Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Gemm7Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String, C: Tensor[T], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GivenTensorFillFree extends Operator with GivenTensorFill {

  def GivenTensorFill1Free[@sp T : Numeric:ClassTag:Field](name: String,shapeInput: Option[Tensor[T]] = None,extra_shape : Option[(Array[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Array[Int])] = None,values : Option[(Array[Float])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GlobalAveragePoolFree extends Operator with GlobalAveragePool {

  def GlobalAveragePool1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GlobalLpPoolFree extends Operator with GlobalLpPool {

  def GlobalLpPool1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,p : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def GlobalLpPool2Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,p : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GlobalMaxPoolFree extends Operator with GlobalMaxPool {

  def GlobalMaxPool1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GreaterFree extends Operator with Greater {

  def Greater1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Greater7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait HardSigmoidFree extends Operator with HardSigmoid {

  def HardSigmoid1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def HardSigmoid6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait HardmaxFree extends Operator with Hardmax {

  def Hardmax1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait IdentityFree extends Operator with Identity {

  def Identity1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait IfFree extends Operator with If {

  def If1Free[@sp B : Numeric:ClassTag:Field,@sp V : Numeric:ClassTag:Field](name: String,cond: Tensor[B], condname: String,else_branch : (Graph),then_branch : (Graph))
(implicit evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : FS[(Tensor[V])]

}
@free trait ImageScalerFree extends Operator with ImageScaler {

  def ImageScaler1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,bias : Option[(Array[Float])] = None,scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ImputerFree extends Operator with Imputer {

  def Imputer1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,imputed_value_floats : Option[(Array[Float])] = None,imputed_value_int64s : Option[(Array[Int])] = None,replaced_value_float : Option[(Float)] = None,replaced_value_int64 : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[T])]

}
@free trait InstanceNormalizationFree extends Operator with InstanceNormalization {

  def InstanceNormalization1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String,consumed_inputs : Option[(Array[Int])] = None,epsilon : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def InstanceNormalization6Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, scale: Tensor[T], scalename: String, B: Tensor[T], Bname: String,epsilon : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LRNFree extends Operator with LRN {

  def LRN1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LSTMFree extends Operator with LSTM {

  def LSTM1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T], Tensor[T])]


  def LSTM7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T], Tensor[T])]

}
@free trait LabelEncoderFree extends Operator with LabelEncoder {

  def LabelEncoder1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,classes_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2])]

}
@free trait LeakyReluFree extends Operator with LeakyRelu {

  def LeakyRelu1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def LeakyRelu6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LessFree extends Operator with Less {

  def Less1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Less7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait LinearClassifierFree extends Operator with LinearClassifier {

  def LinearClassifier1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : (Array[Float]),intercepts : Option[(Array[Float])] = None,multi_class : Option[(Int)] = None,post_transform : Option[(String)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2], Tensor[Float])]

}
@free trait LinearRegressorFree extends Operator with LinearRegressor {

  def LinearRegressor1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,coefficients : Option[(Array[Float])] = None,intercepts : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,targets : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait LogFree extends Operator with Log {

  def Log1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Log6Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LogSoftmaxFree extends Operator with LogSoftmax {

  def LogSoftmax1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LoopFree extends Operator with Loop {

  def Loop1Free[@sp I : Numeric:ClassTag:Field,@sp B : Numeric:ClassTag:Field,@sp V : Numeric:ClassTag:Field](name: String,M: I, Mname: String, cond: B, condname: String,body : (Graph))
(implicit evI:(UNil TypeOr Long)#check[I],evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : FS[(Tensor[V])]

}
@free trait LpNormalizationFree extends Operator with LpNormalization {

  def LpNormalization1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None,p : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LpPoolFree extends Operator with LpPool {

  def LpPool1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])] = None,p : Option[(Float)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def LpPool2Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),p : Option[(Int)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MatMulFree extends Operator with MatMul {

  def MatMul1Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MaxFree extends Operator with Max {

  def Max1Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Max6Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Max8Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MaxPoolFree extends Operator with MaxPool {

  def MaxPool1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def MaxPool8Free[@sp T : Numeric:ClassTag:Field,@sp I : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Array[Int]),pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : FS[(Tensor[T], Tensor[I])]

}
@free trait MaxRoiPoolFree extends Operator with MaxRoiPool {

  def MaxRoiPool1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, rois: Tensor[T], roisname: String,pooled_shape : (Array[Int]),spatial_scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MeanFree extends Operator with Mean {

  def Mean1Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Mean6Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Mean8Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MeanVarianceNormalizationFree extends Operator with MeanVarianceNormalization {

  def MeanVarianceNormalization1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,across_channels : Option[(Int)] = None,normalize_variance : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MinFree extends Operator with Min {

  def Min1Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Min6Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Min8Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MulFree extends Operator with Mul {

  def Mul1Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Mul6Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Mul7Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MultinomialFree extends Operator with Multinomial {

  def Multinomial7Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Int TypeOr Long)#check[T2])    : FS[(Tensor[T2])]

}
@free trait NegFree extends Operator with Neg {

  def Neg1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Neg6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait NormalizerFree extends Operator with Normalizer {

  def Normalizer1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,norm : Option[(String)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait NotFree extends Operator with Not {

  def Not1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T])    : FS[(Tensor[T])]

}
@free trait OneHotEncoderFree extends Operator with OneHotEncoder {

  def OneHotEncoder1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,zeros : Option[(Int)] = None)
(implicit evT:(UNil TypeOr String TypeOr Long TypeOr Int TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Float])]

}
@free trait OrFree extends Operator with Or {

  def Or1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Or7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait PReluFree extends Operator with PRelu {

  def PRelu1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def PRelu6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def PRelu7Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, slope: Tensor[T], slopename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait PadFree extends Operator with Pad {

  def Pad1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,mode : Option[(String)] = None,paddings : (Array[Int]),value : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Pad2Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,mode : Option[(String)] = None,pads : (Array[Int]),value : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ParametricSoftplusFree extends Operator with ParametricSoftplus {

  def ParametricSoftplus1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait PowFree extends Operator with Pow {

  def Pow1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, Y: Tensor[T], Yname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Pow7Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, Y: Tensor[T], Yname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait RNNFree extends Operator with RNN {

  def RNN1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]


  def RNN7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String, W: Tensor[T], Wname: String, R: Tensor[T], Rname: String,B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]

}
@free trait RandomNormalFree extends Operator with RandomNormal {

  def RandomNormal1Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait RandomNormalLikeFree extends Operator with RandomNormalLike {

  def RandomNormalLike1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : FS[(Tensor[T2])]

}
@free trait RandomUniformFree extends Operator with RandomUniform {

  def RandomUniform1Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait RandomUniformLikeFree extends Operator with RandomUniformLike {

  def RandomUniformLike1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,input: Tensor[T1], inputname: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : FS[(Tensor[T2])]

}
@free trait ReciprocalFree extends Operator with Reciprocal {

  def Reciprocal1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Reciprocal6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceL1Free extends Operator with ReduceL1 {

  def ReduceL11Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceL2Free extends Operator with ReduceL2 {

  def ReduceL21Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceLogSumExpFree extends Operator with ReduceLogSumExp {

  def ReduceLogSumExp1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceLogSumFree extends Operator with ReduceLogSum {

  def ReduceLogSum1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceMaxFree extends Operator with ReduceMax {

  def ReduceMax1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceMeanFree extends Operator with ReduceMean {

  def ReduceMean1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceMinFree extends Operator with ReduceMin {

  def ReduceMin1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceProdFree extends Operator with ReduceProd {

  def ReduceProd1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceSumFree extends Operator with ReduceSum {

  def ReduceSum1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceSumSquareFree extends Operator with ReduceSumSquare {

  def ReduceSumSquare1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReluFree extends Operator with Relu {

  def Relu1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Relu6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReshapeFree extends Operator with Reshape {

  def Reshape1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,consumed_inputs : Option[(Array[Int])] = None,shape : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Reshape5Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String, shape: Tensor[Long], shapename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SVMClassifierFree extends Operator with SVMClassifier {

  def SVMClassifier1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,post_transform : Option[(String)] = None,prob_a : Option[(Array[Float])] = None,prob_b : Option[(Array[Float])] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None,vectors_per_class : Option[(Array[Int])] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2], Tensor[Float])]

}
@free trait SVMRegressorFree extends Operator with SVMRegressor {

  def SVMRegressor1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,n_supports : Option[(Int)] = None,one_class : Option[(Int)] = None,post_transform : Option[(String)] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait ScaleFree extends Operator with Scale {

  def Scale1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ScaledTanhFree extends Operator with ScaledTanh {

  def ScaledTanh1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ScalerFree extends Operator with Scaler {

  def Scaler1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,offset : Option[(Array[Float])] = None,scaleAttr : Option[(Array[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait ScanFree extends Operator with Scan {

  def Scan8Free[@sp I : Numeric:ClassTag:Field,@sp V : Numeric:ClassTag:Field](name: String,sequence_lens: Option[Tensor[I]] = None,body : (Graph),directions : Option[(Array[Int])] = None,num_scan_inputs : (Int))
(implicit evI:(UNil TypeOr Long)#check[I],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : FS[(Tensor[V])]

}
@free trait SeluFree extends Operator with Selu {

  def Selu1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,gamma : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Selu6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ShapeFree extends Operator with Shape {

  def Shape1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T1])]

}
@free trait SigmoidFree extends Operator with Sigmoid {

  def Sigmoid1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sigmoid6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SinFree extends Operator with Sin {

  def Sin7Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SizeFree extends Operator with Size {

  def Size1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T1])]

}
@free trait SliceFree extends Operator with Slice {

  def Slice1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None,ends : (Array[Int]),starts : (Array[Int]))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SoftmaxFree extends Operator with Softmax {

  def Softmax1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SoftplusFree extends Operator with Softplus {

  def Softplus1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SoftsignFree extends Operator with Softsign {

  def Softsign1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SpaceToDepthFree extends Operator with SpaceToDepth {

  def SpaceToDepth1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,blocksize : (Int))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SplitFree extends Operator with Split {

  def Split1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,split: Option[Tensor[T]] = None,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Split2Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SqrtFree extends Operator with Sqrt {

  def Sqrt1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sqrt6Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SqueezeFree extends Operator with Squeeze {

  def Squeeze1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SubFree extends Operator with Sub {

  def Sub1Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sub6Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sub7Free[@sp T : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SumFree extends Operator with Sum {

  def Sum1Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sum6Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sum8Free[@sp T : Numeric:ClassTag:Field](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait TanFree extends Operator with Tan {

  def Tan7Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait TanhFree extends Operator with Tanh {

  def Tanh1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String,consumed_inputs : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Tanh6Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ThresholdedReluFree extends Operator with ThresholdedRelu {

  def ThresholdedRelu1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait TileFree extends Operator with Tile {

  def Tile1Free[@sp T : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, tiles: Tensor[T], tilesname: String, axis: Tensor[T], axisname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Tile6Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,input: Tensor[T], inputname: String, repeats: Tensor[T1], repeatsname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T])]

}
@free trait TopKFree extends Operator with TopK {

  def TopK1Free[@sp T : Numeric:ClassTag:Field,@sp I : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,axis : Option[(Int)] = None,k : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : FS[(Tensor[T], Tensor[I])]

}
@free trait TransposeFree extends Operator with Transpose {

  def Transpose1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,perm : Option[(Array[Int])] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait TreeEnsembleClassifierFree extends Operator with TreeEnsembleClassifier {

  def TreeEnsembleClassifier1Free[@sp T1 : Numeric:ClassTag:Field,@sp T2 : Numeric:ClassTag:Field](name: String,X: Tensor[T1], Xname: String,base_values : Option[(Array[Float])] = None,class_ids : Option[(Array[Int])] = None,class_nodeids : Option[(Array[Int])] = None,class_treeids : Option[(Array[Int])] = None,class_weights : Option[(Array[Float])] = None,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2], Tensor[Float])]

}
@free trait TreeEnsembleRegressorFree extends Operator with TreeEnsembleRegressor {

  def TreeEnsembleRegressor1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,aggregate_function : Option[(String)] = None,base_values : Option[(Array[Float])] = None,n_targets : Option[(Int)] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,target_ids : Option[(Array[Int])] = None,target_nodeids : Option[(Array[Int])] = None,target_treeids : Option[(Array[Int])] = None,target_weights : Option[(Array[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait UnsqueezeFree extends Operator with Unsqueeze {

  def Unsqueeze1Free[@sp T : Numeric:ClassTag:Field](name: String,data: Tensor[T], dataname: String,axes : (Array[Int]))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait UpsampleFree extends Operator with Upsample {

  def Upsample1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,height_scaleAttr : (Float),mode : Option[(String)] = None,width_scaleAttr : (Float))
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Upsample7Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[T], Xname: String,mode : Option[(String)] = None,scaleAttrs : (Array[Float]))
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait XorFree extends Operator with Xor {

  def Xor1Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Xor7Free[@sp T : Numeric:ClassTag:Field,@sp T1 : Numeric:ClassTag:Field](name: String,A: Tensor[T], Aname: String, B: Tensor[T], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait ZipMapFree extends Operator with ZipMap {

  def ZipMap1Free[@sp T : Numeric:ClassTag:Field](name: String,X: Tensor[Float], Xname: String,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None)
(implicit evT:(UNil TypeOr Seq[Map[String, Float]] TypeOr Seq[Map[Long, Float]])#check[T])    : FS[(T)]

}}
