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
  def inputDataFree[T : Numeric:ClassTag]
//  (implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
  : FS[Tensor[T]]
//  def getParamsFree[T : Numeric:ClassTag](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): FS[Tensor[T]]
//  def getAttributesFree[T : Numeric:ClassTag](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): FS[Tensor[T]]
}
@free trait AbsFree extends Operator with Abs {

  def Abs1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Abs6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AcosFree extends Operator with Acos {

  def Acos7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AcoshFree extends Operator with Acosh {

  def Acosh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AddFree extends Operator with Add {

  def Add1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Add6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Add7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AffineFree extends Operator with Affine {

  def Affine1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AndFree extends Operator with And {

  def And1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def And7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait ArgMaxFree extends Operator with ArgMax {

  def ArgMax1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Long])]

}
@free trait ArgMinFree extends Operator with ArgMin {

  def ArgMin1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Long])]

}
@free trait ArrayFeatureExtractorFree extends Operator with ArrayFeatureExtractor {

  def ArrayFeatureExtractor1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], Y: Option[Tensor[Long]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int TypeOr String)#check[T])    : FS[(Tensor[T])]

}
@free trait AsinFree extends Operator with Asin {

  def Asin7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AsinhFree extends Operator with Asinh {

  def Asinh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AtanFree extends Operator with Atan {

  def Atan7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AtanhFree extends Operator with Atanh {

  def Atanh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait AveragePoolFree extends Operator with AveragePool {

  def AveragePool1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def AveragePool7Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait BatchNormalizationFree extends Operator with BatchNormalization {

  def BatchNormalization1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])],epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]], mean: Option[Tensor[T]], someVar: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization6Free[@sp T : Numeric:ClassTag](name: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]], mean: Option[Tensor[T]], someVar: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization7Free[@sp T : Numeric:ClassTag](name: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]], mean: Option[Tensor[T]], someVar: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization9Free[@sp T : Numeric:ClassTag](name: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,X: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]], mean: Option[Tensor[T]], someVar: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]

}
@free trait BinarizerFree extends Operator with Binarizer {

  def Binarizer1Free[@sp T : Numeric:ClassTag](name: String,threshold : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[T])]

}
@free trait CastFree extends Operator with Cast {

  def Cast1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,to : Option[(String)],input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T2])    : FS[(Tensor[T2])]


  def Cast6Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,to : Option[(Int)],input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T2])    : FS[(Tensor[T2])]


  def Cast9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,to : Option[(Int)],input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T2])    : FS[(Tensor[T2])]

}
@free trait CastMapFree extends Operator with CastMap {

  def CastMap1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,cast_to : Option[(String)] = None,map_form : Option[(String)] = None,max_map : Option[(Int)] = None,X: Option[T1])
(implicit evT1:(UNil TypeOr Map[Long, String] TypeOr Map[Long, Float])#check[T1],evT2:(UNil TypeOr String TypeOr Float TypeOr Long)#check[T2])    : FS[(Tensor[T2])]

}
@free trait CategoryMapperFree extends Operator with CategoryMapper {

  def CategoryMapper1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2])]

}
@free trait CeilFree extends Operator with Ceil {

  def Ceil1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Ceil6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ClipFree extends Operator with Clip {

  def Clip1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Clip6Free[@sp T : Numeric:ClassTag](name: String,max : Option[(Float)] = None,min : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait CompressFree extends Operator with Compress {

  def Compress9Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]], condition: Option[Tensor[T1]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T])]

}
@free trait ConcatFree extends Operator with Concat {

  def Concat4Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)],inputs: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait ConstantFree extends Operator with Constant {

  def Constant1Free[@sp T : Numeric:ClassTag](name: String,value : Option[(Tensor[T])])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Constant9Free[@sp T : Numeric:ClassTag](name: String,value : Option[(Tensor[T])])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait ConstantOfShapeFree extends Operator with ConstantOfShape {

  def ConstantOfShape9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,value : Option[(Tensor[T2])] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Long)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : FS[(Tensor[T2])]

}
@free trait ConvFree extends Operator with Conv {

  def Conv1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]], W: Option[Tensor[T]],B: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ConvTransposeFree extends Operator with ConvTranspose {

  def ConvTranspose1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,output_padding : Option[(Array[Int])] = None,output_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]], W: Option[Tensor[T]],B: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait CosFree extends Operator with Cos {

  def Cos7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait CoshFree extends Operator with Cosh {

  def Cosh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait CropFree extends Operator with Crop {

  def Crop1Free[@sp T : Numeric:ClassTag](name: String,border : Option[(Array[Int])] = None,scaleAttr : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait DepthToSpaceFree extends Operator with DepthToSpace {

  def DepthToSpace1Free[@sp T : Numeric:ClassTag](name: String,blocksize : Option[(Int)],input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait DictVectorizerFree extends Operator with DictVectorizer {

  def DictVectorizer1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,int64_vocabulary : Option[(Array[Int])] = None,string_vocabulary : Option[(Array[String])] = None,X: Option[T1])
(implicit evT1:(UNil TypeOr Map[String, Long] TypeOr Map[Long, String] TypeOr Map[Long, Float] TypeOr Map[Long, Double] TypeOr Map[String, Float] TypeOr Map[String, Double])#check[T1],evT2:(UNil TypeOr Long TypeOr Float TypeOr Double TypeOr String)#check[T2])    : FS[(Tensor[T2])]

}
@free trait DivFree extends Operator with Div {

  def Div1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Div6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Div7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait DropoutFree extends Operator with Dropout {

  /*
  def Dropout1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T])]


  def Dropout6Free[@sp T : Numeric:ClassTag](name: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T], Tensor[T])]

*/
  def Dropout7Free[@sp T : Numeric:ClassTag](name: String,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
//(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])   
: FS[(Tensor[T], Tensor[T])]

}
@free trait DynamicSliceFree extends Operator with DynamicSlice {

  def DynamicSlice1Free[@sp T : Numeric:ClassTag,@sp Tind : Numeric:ClassTag](name: String,data: Option[Tensor[T]], starts: Option[Tensor[Tind]], ends: Option[Tensor[Tind]],axes: Option[Tensor[Tind]] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : FS[(Tensor[T])]

}
@free trait EluFree extends Operator with Elu {

  def Elu1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Elu6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait EqualFree extends Operator with Equal {

  def Equal1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Equal7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait ErfFree extends Operator with Erf {

  def Erf9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ExpFree extends Operator with Exp {

  def Exp1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Exp6Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ExpandFree extends Operator with Expand {

  def Expand8Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]], shape: Option[Tensor[Long]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait EyeLikeFree extends Operator with EyeLike {

  def EyeLike9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,k : Option[(Int)] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : FS[(Tensor[T2])]

}
@free trait FlattenFree extends Operator with Flatten {

  def Flatten1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Flatten9Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait FloorFree extends Operator with Floor {

  def Floor1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Floor6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GRUFree extends Operator with GRU {

  def GRU1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]


  def GRU3Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]


  def GRU7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]

}
@free trait GRUUnitFree extends Operator with GRUUnit {

  def GRUUnit1Free[@sp T : Numeric:ClassTag](name: String,drop_states : Option[(Int)] = None,hidden_prev: Option[Tensor[T]], gates: Option[Tensor[T]], seq_lengths: Option[Tensor[T]], t: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GatherFree extends Operator with Gather {

  def Gather1Free[@sp T : Numeric:ClassTag,@sp Tind : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,data: Option[Tensor[T]], indices: Option[Tensor[Tind]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : FS[(Tensor[T])]

}
@free trait GemmFree extends Operator with Gemm {

  def Gemm1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]


  def Gemm6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]


  def Gemm7Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]


  def Gemm9Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]

}
@free trait GivenTensorFillFree extends Operator with GivenTensorFill {

  def GivenTensorFill1Free[@sp T : Numeric:ClassTag](name: String,extra_shape : Option[(Array[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Array[Int])] = None,values : Option[(Array[Float])] = None,shapeInput: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GlobalAveragePoolFree extends Operator with GlobalAveragePool {

  def GlobalAveragePool1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GlobalLpPoolFree extends Operator with GlobalLpPool {

  def GlobalLpPool1Free[@sp T : Numeric:ClassTag](name: String,p : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def GlobalLpPool2Free[@sp T : Numeric:ClassTag](name: String,p : Option[(Int)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GlobalMaxPoolFree extends Operator with GlobalMaxPool {

  def GlobalMaxPool1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait GreaterFree extends Operator with Greater {

  def Greater1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Greater7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Greater9Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait HardSigmoidFree extends Operator with HardSigmoid {

  def HardSigmoid1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def HardSigmoid6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait HardmaxFree extends Operator with Hardmax {

  def Hardmax1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait IdentityFree extends Operator with Identity {

  def Identity1Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait IfFree extends Operator with If {

  def If1Free[@sp B : Numeric:ClassTag,@sp V : Numeric:ClassTag](name: String,else_branch : Option[(Graph)],then_branch : Option[(Graph)],cond: Option[Tensor[B]])
(implicit evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : FS[(Tensor[V])]

}
@free trait ImageScalerFree extends Operator with ImageScaler {

  def ImageScaler1Free[@sp T : Numeric:ClassTag](name: String,bias : Option[(Array[Float])] = None,scaleAttr : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ImputerFree extends Operator with Imputer {

  def Imputer1Free[@sp T : Numeric:ClassTag](name: String,imputed_value_floats : Option[(Array[Float])] = None,imputed_value_int64s : Option[(Array[Int])] = None,replaced_value_float : Option[(Float)] = None,replaced_value_int64 : Option[(Int)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[T])]

}
@free trait InstanceNormalizationFree extends Operator with InstanceNormalization {

  def InstanceNormalization1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,epsilon : Option[(Float)] = None,input: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def InstanceNormalization6Free[@sp T : Numeric:ClassTag](name: String,epsilon : Option[(Float)] = None,input: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait IsNaNFree extends Operator with IsNaN {

  def IsNaN9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Boolean)#check[T2])    : FS[(Tensor[T2])]

}
@free trait LRNFree extends Operator with LRN {

  def LRN1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : Option[(Int)],X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LSTMFree extends Operator with LSTM {

  def LSTM1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T], Tensor[T])]


  def LSTM7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T], Tensor[T])]

}
@free trait LabelEncoderFree extends Operator with LabelEncoder {

  def LabelEncoder1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,classes_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T1],evT2:(UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T2])    : FS[(Tensor[T2])]


  def LabelEncoder2Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,default_float : Option[(Float)] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,keys_floats : Option[(Array[Float])] = None,keys_int64s : Option[(Array[Int])] = None,keys_strings : Option[(Array[String])] = None,values_floats : Option[(Array[Float])] = None,values_int64s : Option[(Array[Int])] = None,values_strings : Option[(Array[String])] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T1],evT2:(UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T2])    : FS[(Tensor[T2])]

}
@free trait LeakyReluFree extends Operator with LeakyRelu {

  def LeakyRelu1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def LeakyRelu6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LessFree extends Operator with Less {

  def Less1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Less7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Less9Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait LinearClassifierFree extends Operator with LinearClassifier {

  def LinearClassifier1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : Option[(Array[Float])],intercepts : Option[(Array[Float])] = None,multi_class : Option[(Int)] = None,post_transform : Option[(String)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2], Tensor[Float])]

}
@free trait LinearRegressorFree extends Operator with LinearRegressor {

  def LinearRegressor1Free[@sp T : Numeric:ClassTag](name: String,coefficients : Option[(Array[Float])] = None,intercepts : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,targets : Option[(Int)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait LogFree extends Operator with Log {

  def Log1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Log6Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LogSoftmaxFree extends Operator with LogSoftmax {

  def LogSoftmax1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LoopFree extends Operator with Loop {

  def Loop1Free[@sp I : Numeric:ClassTag,@sp B : Numeric:ClassTag,@sp V : Numeric:ClassTag](name: String,body : Option[(Graph)],M: Option[Tensor[I]] = None, cond: Option[Tensor[B]] = None,v_initial: Seq[Option[Tensor[V]]])
(implicit evI:(UNil TypeOr Long)#check[I],evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : FS[(Tensor[V])]

}
@free trait LpNormalizationFree extends Operator with LpNormalization {

  def LpNormalization1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,p : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait LpPoolFree extends Operator with LpPool {

  def LpPool1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])] = None,p : Option[(Float)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def LpPool2Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])],p : Option[(Int)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MatMulFree extends Operator with MatMul {

  def MatMul1Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]


  def MatMul9Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]

}
@free trait MaxFree extends Operator with Max {

  def Max6Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Max8Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MaxPoolFree extends Operator with MaxPool {

  def MaxPool1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def MaxPool8Free[@sp T : Numeric:ClassTag,@sp I : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : FS[(Tensor[T], Tensor[I])]

}
@free trait MaxRoiPoolFree extends Operator with MaxRoiPool {

  def MaxRoiPool1Free[@sp T : Numeric:ClassTag](name: String,pooled_shape : Option[(Array[Int])],spatial_scaleAttr : Option[(Float)] = None,X: Option[Tensor[T]], rois: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MaxUnpoolFree extends Operator with MaxUnpool {

  def MaxUnpool9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T1]], I: Option[Tensor[T2]],output_shapeInput: Option[Tensor[T2]] = None)
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Long)#check[T2])    : FS[(Tensor[T1])]

}
@free trait MeanFree extends Operator with Mean {

  def Mean6Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Mean8Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MinFree extends Operator with Min {

  def Min6Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Min8Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MulFree extends Operator with Mul {

  def Mul1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Mul6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Mul7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait MultinomialFree extends Operator with Multinomial {

  def Multinomial7Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Int TypeOr Long)#check[T2])    : FS[(Tensor[T2])]

}
@free trait NegFree extends Operator with Neg {

  def Neg1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Neg6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait NonZeroFree extends Operator with NonZero {

  def NonZero9Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[Long])]

}
@free trait NormalizerFree extends Operator with Normalizer {

  def Normalizer1Free[@sp T : Numeric:ClassTag](name: String,norm : Option[(String)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait NotFree extends Operator with Not {

  def Not1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T])    : FS[(Tensor[T])]

}
@free trait OneHotEncoderFree extends Operator with OneHotEncoder {

  def OneHotEncoder1Free[@sp T : Numeric:ClassTag](name: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,zeros : Option[(Int)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr String TypeOr Long TypeOr Int TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[Float])]

}
@free trait OneHotFree extends Operator with OneHot {

  def OneHot9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag,@sp T3 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,indices: Option[Tensor[T1]], depth: Option[Tensor[T2]], values: Option[Tensor[T3]])
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T2],evT3:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T3])    : FS[(Tensor[T3])]

}
@free trait OrFree extends Operator with Or {

  def Or1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Or7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait PReluFree extends Operator with PRelu {

  def PRelu1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]], slope: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]


  def PRelu6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], slope: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]


  def PRelu7Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], slope: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]


  def PRelu9Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], slope: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : FS[(Tensor[T])]

}
@free trait PadFree extends Operator with Pad {

  def Pad1Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,paddings : Option[(Array[Int])],value : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Pad2Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,pads : Option[(Array[Int])],value : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ParametricSoftplusFree extends Operator with ParametricSoftplus {

  def ParametricSoftplus1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait PowFree extends Operator with Pow {

  def Pow1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,X: Option[Tensor[T]], Y: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Pow7Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], Y: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait RNNFree extends Operator with RNN {

  def RNN1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]


  def RNN7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : FS[(Tensor[T], Tensor[T])]

}
@free trait RandomNormalFree extends Operator with RandomNormal {

  def RandomNormal1Free[@sp T : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None,shape : Option[(Array[Int])])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait RandomNormalLikeFree extends Operator with RandomNormalLike {

  def RandomNormalLike1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : FS[(Tensor[T2])]

}
@free trait RandomUniformFree extends Operator with RandomUniform {

  def RandomUniform1Free[@sp T : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None,shape : Option[(Array[Int])])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait RandomUniformLikeFree extends Operator with RandomUniformLike {

  def RandomUniformLike1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : FS[(Tensor[T2])]

}
@free trait ReciprocalFree extends Operator with Reciprocal {

  def Reciprocal1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Reciprocal6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceL1Free extends Operator with ReduceL1 {

  def ReduceL11Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceL2Free extends Operator with ReduceL2 {

  def ReduceL21Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceLogSumExpFree extends Operator with ReduceLogSumExp {

  def ReduceLogSumExp1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceLogSumFree extends Operator with ReduceLogSum {

  def ReduceLogSum1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceMaxFree extends Operator with ReduceMax {

  def ReduceMax1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceMeanFree extends Operator with ReduceMean {

  def ReduceMean1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceMinFree extends Operator with ReduceMin {

  def ReduceMin1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceProdFree extends Operator with ReduceProd {

  def ReduceProd1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceSumFree extends Operator with ReduceSum {

  def ReduceSum1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReduceSumSquareFree extends Operator with ReduceSumSquare {

  def ReduceSumSquare1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ReluFree extends Operator with Relu {

//  def Relu1Free[@sp T](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
//    : FS[(Tensor[T])]


  def Relu6Free[@sp T : Numeric: ClassTag](name: String,X: Option[Tensor[T]])
//  (implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
    : FS[(Tensor[T])]

}
@free trait ReshapeFree extends Operator with Reshape {

  def Reshape1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,shape : Option[(Array[Int])] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Reshape5Free[@sp T : Numeric:ClassTag](name: String,data: Option[Tensor[T]], shape: Option[Tensor[Long]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SVMClassifierFree extends Operator with SVMClassifier {

  def SVMClassifier1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,post_transform : Option[(String)] = None,prob_a : Option[(Array[Float])] = None,prob_b : Option[(Array[Float])] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None,vectors_per_class : Option[(Array[Int])] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2], Tensor[Float])]

}
@free trait SVMRegressorFree extends Operator with SVMRegressor {

  def SVMRegressor1Free[@sp T : Numeric:ClassTag](name: String,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,n_supports : Option[(Int)] = None,one_class : Option[(Int)] = None,post_transform : Option[(String)] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait ScaleFree extends Operator with Scale {

  def Scale1Free[@sp T : Numeric:ClassTag](name: String,scaleAttr : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ScaledTanhFree extends Operator with ScaledTanh {

  def ScaledTanh1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ScalerFree extends Operator with Scaler {

  def Scaler1Free[@sp T : Numeric:ClassTag](name: String,offset : Option[(Array[Float])] = None,scaleAttr : Option[(Array[Float])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait ScanFree extends Operator with Scan {

  def Scan9Free[@sp V : Numeric:ClassTag](name: String,body : Option[(Graph)],num_scan_inputs : Option[(Int)],scan_input_axes : Option[(Array[Int])] = None,scan_input_directions : Option[(Array[Int])] = None,scan_output_axes : Option[(Array[Int])] = None,scan_output_directions : Option[(Array[Int])] = None,initial_state_and_scan_inputs: Seq[Option[Tensor[V]]])
(implicit evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : FS[(Tensor[V])]

}
@free trait ScatterFree extends Operator with Scatter {

  def Scatter9Free[@sp T : Numeric:ClassTag,@sp Tind : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,data: Option[Tensor[T]], indices: Option[Tensor[Tind]], updates: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : FS[(Tensor[T])]

}
@free trait SeluFree extends Operator with Selu {

  def Selu1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,gamma : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Selu6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait ShapeFree extends Operator with Shape {

  def Shape1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T1])]

}
@free trait ShrinkFree extends Operator with Shrink {

  def Shrink9Free[@sp T : Numeric:ClassTag](name: String,bias : Option[(Float)] = None,lambd : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SigmoidFree extends Operator with Sigmoid {

  def Sigmoid1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sigmoid6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SignFree extends Operator with Sign {

  def Sign9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SinFree extends Operator with Sin {

  def Sin7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SinhFree extends Operator with Sinh {

  def Sinh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SizeFree extends Operator with Size {

  def Size1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T1])]

}
@free trait SliceFree extends Operator with Slice {

  def Slice1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,ends : Option[(Array[Int])],starts : Option[(Array[Int])],data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SoftmaxFree extends Operator with Softmax {

  def Softmax1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SoftplusFree extends Operator with Softplus {

  def Softplus1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SoftsignFree extends Operator with Softsign {

  def Softsign1Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SpaceToDepthFree extends Operator with SpaceToDepth {

  def SpaceToDepth1Free[@sp T : Numeric:ClassTag](name: String,blocksize : Option[(Int)],input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SplitFree extends Operator with Split {

  def Split1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None,input: Option[Tensor[T]],split: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Split2Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SqrtFree extends Operator with Sqrt {

  def Sqrt1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sqrt6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SqueezeFree extends Operator with Squeeze {

  def Squeeze1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait SubFree extends Operator with Sub {

  def Sub1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sub6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sub7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait SumFree extends Operator with Sum {

  def Sum6Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Sum8Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait TanFree extends Operator with Tan {

  def Tan7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait TanhFree extends Operator with Tanh {

  def Tanh1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]


  def Tanh6Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait TfIdfVectorizerFree extends Operator with TfIdfVectorizer {

  def TfIdfVectorizer9Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,max_gram_length : Option[(Int)],max_skip_count : Option[(Int)],min_gram_length : Option[(Int)],mode : Option[(String)],ngram_counts : Option[(Array[Int])],ngram_indexes : Option[(Array[Int])],pool_int64s : Option[(Array[Int])] = None,pool_strings : Option[(Array[String])] = None,weights : Option[(Array[Float])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr String TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Float)#check[T1])    : FS[(Tensor[T1])]

}
@free trait ThresholdedReluFree extends Operator with ThresholdedRelu {

  def ThresholdedRelu1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : FS[(Tensor[T])]

}
@free trait TileFree extends Operator with Tile {

  def Tile1Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]], tiles: Option[Tensor[T]], axis: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Tile6Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,input: Option[Tensor[T]], repeats: Option[Tensor[T1]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : FS[(Tensor[T])]

}
@free trait TopKFree extends Operator with TopK {

  def TopK1Free[@sp T : Numeric:ClassTag,@sp I : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,k : Option[(Int)],X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : FS[(Tensor[T], Tensor[I])]

}
@free trait TransposeFree extends Operator with Transpose {

  def Transpose1Free[@sp T : Numeric:ClassTag](name: String,perm : Option[(Array[Int])] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait TreeEnsembleClassifierFree extends Operator with TreeEnsembleClassifier {

  def TreeEnsembleClassifier1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,base_values : Option[(Array[Float])] = None,class_ids : Option[(Array[Int])] = None,class_nodeids : Option[(Array[Int])] = None,class_treeids : Option[(Array[Int])] = None,class_weights : Option[(Array[Float])] = None,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : FS[(Tensor[T2], Tensor[Float])]

}
@free trait TreeEnsembleRegressorFree extends Operator with TreeEnsembleRegressor {

  def TreeEnsembleRegressor1Free[@sp T : Numeric:ClassTag](name: String,aggregate_function : Option[(String)] = None,base_values : Option[(Array[Float])] = None,n_targets : Option[(Int)] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,target_ids : Option[(Array[Int])] = None,target_nodeids : Option[(Array[Int])] = None,target_treeids : Option[(Array[Int])] = None,target_weights : Option[(Array[Float])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : FS[(Tensor[Float])]

}
@free trait UnsqueezeFree extends Operator with Unsqueeze {

  def Unsqueeze1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])],data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait UpsampleFree extends Operator with Upsample {

  def Upsample1Free[@sp T : Numeric:ClassTag](name: String,height_scaleAttr : Option[(Float)],mode : Option[(String)] = None,width_scaleAttr : Option[(Float)],X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Upsample7Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,scaleAttrs : Option[(Array[Float])],X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]


  def Upsample9Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,X: Option[Tensor[T]], scales: Option[Tensor[Float]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait WhereFree extends Operator with Where {

  def Where9Free[@sp B : Numeric:ClassTag,@sp T : Numeric:ClassTag](name: String,condition: Option[Tensor[B]], X: Option[Tensor[T]], Y: Option[Tensor[T]])
(implicit evB:(UNil TypeOr Boolean)#check[B],evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : FS[(Tensor[T])]

}
@free trait XorFree extends Operator with Xor {

  def Xor1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]


  def Xor7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : FS[(Tensor[T1])]

}
@free trait ZipMapFree extends Operator with ZipMap {

  def ZipMap1Free[@sp T : Numeric:ClassTag](name: String,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,X: Option[Tensor[Float]])
(implicit evT:(UNil TypeOr Seq[Map[String, Float]] TypeOr Seq[Map[Long, Float]])#check[T])    : FS[(T)]

}}
