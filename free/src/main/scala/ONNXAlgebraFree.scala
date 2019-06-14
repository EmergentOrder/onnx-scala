package org.emergentorder

import scalaz.zio.Task
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
import org.emergentorder.union.UnionType._

import onnx._
package object onnxFree {

    trait DataSourceFree  {
  def inputDataFree[T : Numeric:ClassTag](implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Task[Tensor[T]]
  def getParamsFree[T : Numeric:ClassTag](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Task[Tensor[T]]
  def getAttributesFree[T : Numeric:ClassTag](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Task[Tensor[T]]
}
trait AbsFree extends Operator {

  def Abs1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Abs6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait AcosFree extends Operator {

  def Acos7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait AcoshFree extends Operator {

  def Acosh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait AddFree extends Operator {

  def Add1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Add6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Add7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait AndFree extends Operator {

  def And1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]


  def And7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]

}
trait ArgMaxFree extends Operator {

  def ArgMax1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[Long])]

}
trait ArgMinFree extends Operator {

  def ArgMin1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[Long])]

}
trait ArrayFeatureExtractorFree extends Operator {

  def ArrayFeatureExtractor1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], Y: Option[Tensor[Long]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int TypeOr String)#check[T])    : Task[(Tensor[T])]

}
trait AsinFree extends Operator {

  def Asin7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait AsinhFree extends Operator {

  def Asinh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait AtanFree extends Operator {

  def Atan7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait AtanhFree extends Operator {

  def Atanh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait AveragePoolFree extends Operator {

  def AveragePool1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def AveragePool7Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def AveragePool10Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,ceil_mode : Option[(Int)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait BatchNormalizationFree extends Operator {

  def BatchNormalization1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])],epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]], mean: Option[Tensor[T]], someVar: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization6Free[@sp T : Numeric:ClassTag](name: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]], mean: Option[Tensor[T]], someVar: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization7Free[@sp T : Numeric:ClassTag](name: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None,X: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]], mean: Option[Tensor[T]], someVar: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]


  def BatchNormalization9Free[@sp T : Numeric:ClassTag](name: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,X: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]], mean: Option[Tensor[T]], someVar: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T], Tensor[T], Tensor[T], Tensor[T])]

}
trait BinarizerFree extends Operator {

  def Binarizer1Free[@sp T : Numeric:ClassTag](name: String,threshold : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : Task[(Tensor[T])]

}
trait CastFree extends Operator {

  def Cast1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,to : Option[(String)],input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T2])    : Task[(Tensor[T2])]


  def Cast6Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,to : Option[(Int)],input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T2])    : Task[(Tensor[T2])]


  def Cast9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,to : Option[(Int)],input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean TypeOr String)#check[T2])    : Task[(Tensor[T2])]

}
trait CastMapFree extends Operator {

  def CastMap1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,cast_to : Option[(String)] = None,map_form : Option[(String)] = None,max_map : Option[(Int)] = None,X: Option[T1])
(implicit evT1:(UNil TypeOr Map[Long, String] TypeOr Map[Long, Float])#check[T1],evT2:(UNil TypeOr String TypeOr Float TypeOr Long)#check[T2])    : Task[(Tensor[T2])]

}
trait CategoryMapperFree extends Operator {

  def CategoryMapper1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : Task[(Tensor[T2])]

}
trait CeilFree extends Operator {

  def Ceil1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Ceil6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ClipFree extends Operator {

  def Clip1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Clip6Free[@sp T : Numeric:ClassTag](name: String,max : Option[(Float)] = None,min : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait CompressFree extends Operator {

  def Compress9Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]], condition: Option[Tensor[T1]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T])]

}
trait ConcatFree extends Operator {

  def Concat4Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)],inputs: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait ConstantFree extends Operator {

  def Constant1Free[@sp T : Numeric:ClassTag](name: String,value : Option[(Tensor[T])])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Constant9Free[@sp T : Numeric:ClassTag](name: String,value : Option[(Tensor[T])])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait ConstantOfShapeFree extends Operator {

  def ConstantOfShape9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,value : Option[(Tensor[T2])] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Long)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : Task[(Tensor[T2])]

}
trait ConvFree extends Operator {

  def Conv1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]], W: Option[Tensor[T]],B: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ConvIntegerFree extends Operator {

  def ConvInteger10Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag,@sp T3 : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,x: Option[Tensor[T1]], w: Option[Tensor[T2]],x_zero_point: Option[Tensor[T1]] = None, w_zero_point: Option[Tensor[T2]] = None)
(implicit evT1:(UNil TypeOr Byte TypeOr UByte)#check[T1],evT2:(UNil TypeOr Byte TypeOr UByte)#check[T2],evT3:(UNil TypeOr Int)#check[T3])    : Task[(Tensor[T3])]

}
trait ConvTransposeFree extends Operator {

  def ConvTranspose1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,output_padding : Option[(Array[Int])] = None,output_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]], W: Option[Tensor[T]],B: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait CosFree extends Operator {

  def Cos7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait CoshFree extends Operator {

  def Cosh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait DepthToSpaceFree extends Operator {

  def DepthToSpace1Free[@sp T : Numeric:ClassTag](name: String,blocksize : Option[(Int)],input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait DequantizeLinearFree extends Operator {

  def DequantizeLinear10Free[@sp T : Numeric:ClassTag](name: String,x: Option[Tensor[T]], x_scale: Option[Tensor[Float]],x_zero_point: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Byte TypeOr UByte TypeOr Int)#check[T])    : Task[(Tensor[Float])]

}
trait DictVectorizerFree extends Operator {

  def DictVectorizer1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,int64_vocabulary : Option[(Array[Int])] = None,string_vocabulary : Option[(Array[String])] = None,X: Option[T1])
(implicit evT1:(UNil TypeOr Map[String, Long] TypeOr Map[Long, String] TypeOr Map[Long, Float] TypeOr Map[Long, Double] TypeOr Map[String, Float] TypeOr Map[String, Double])#check[T1],evT2:(UNil TypeOr Long TypeOr Float TypeOr Double TypeOr String)#check[T2])    : Task[(Tensor[T2])]

}
trait DivFree extends Operator {

  def Div1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Div6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Div7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait DropoutFree extends Operator {

  def Dropout1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T])]


  def Dropout6Free[@sp T : Numeric:ClassTag](name: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T])]


  def Dropout7Free[@sp T : Numeric:ClassTag](name: String,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T])]


  def Dropout10Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T], Tensor[T1])]

}
trait EluFree extends Operator {

  def Elu1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Elu6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait EqualFree extends Operator {

  def Equal1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]


  def Equal7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]

}
trait ErfFree extends Operator {

  def Erf9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ExpFree extends Operator {

  def Exp1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Exp6Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ExpandFree extends Operator {

  def Expand8Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]], shape: Option[Tensor[Long]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait EyeLikeFree extends Operator {

  def EyeLike9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,k : Option[(Int)] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : Task[(Tensor[T2])]

}
trait FlattenFree extends Operator {

  def Flatten1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Flatten9Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait FloorFree extends Operator {

  def Floor1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Floor6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait GRUFree extends Operator {

  def GRU1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : Task[(Tensor[T], Tensor[T])]


  def GRU3Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : Task[(Tensor[T], Tensor[T])]


  def GRU7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : Task[(Tensor[T], Tensor[T])]

}
trait GatherFree extends Operator {

  def Gather1Free[@sp T : Numeric:ClassTag,@sp Tind : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,data: Option[Tensor[T]], indices: Option[Tensor[Tind]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : Task[(Tensor[T])]

}
trait GemmFree extends Operator {

  def Gemm1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]


  def Gemm6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]


  def Gemm7Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]


  def Gemm9Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]

}
trait GlobalAveragePoolFree extends Operator {

  def GlobalAveragePool1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait GlobalLpPoolFree extends Operator {

  def GlobalLpPool1Free[@sp T : Numeric:ClassTag](name: String,p : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def GlobalLpPool2Free[@sp T : Numeric:ClassTag](name: String,p : Option[(Int)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait GlobalMaxPoolFree extends Operator {

  def GlobalMaxPool1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait GreaterFree extends Operator {

  def Greater1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]


  def Greater7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]


  def Greater9Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]

}
trait HardSigmoidFree extends Operator {

  def HardSigmoid1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def HardSigmoid6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait HardmaxFree extends Operator {

  def Hardmax1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait IdentityFree extends Operator {

  def Identity1Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait IfFree extends Operator {

  def If1Free[@sp B : Numeric:ClassTag,@sp V : Numeric:ClassTag](name: String,else_branch : Option[(Graph)],then_branch : Option[(Graph)],cond: Option[Tensor[B]])
(implicit evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : Task[(Tensor[V])]

}
trait ImputerFree extends Operator {

  def Imputer1Free[@sp T : Numeric:ClassTag](name: String,imputed_value_floats : Option[(Array[Float])] = None,imputed_value_int64s : Option[(Array[Int])] = None,replaced_value_float : Option[(Float)] = None,replaced_value_int64 : Option[(Int)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : Task[(Tensor[T])]

}
trait InstanceNormalizationFree extends Operator {

  def InstanceNormalization1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,epsilon : Option[(Float)] = None,input: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def InstanceNormalization6Free[@sp T : Numeric:ClassTag](name: String,epsilon : Option[(Float)] = None,input: Option[Tensor[T]], scale: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait IsInfFree extends Operator {

  def IsInf10Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,detect_negative : Option[(Int)] = None,detect_positive : Option[(Int)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Boolean)#check[T2])    : Task[(Tensor[T2])]

}
trait IsNaNFree extends Operator {

  def IsNaN9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Boolean)#check[T2])    : Task[(Tensor[T2])]

}
trait LRNFree extends Operator {

  def LRN1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : Option[(Int)],X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait LSTMFree extends Operator {

  def LSTM1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : Task[(Tensor[T], Tensor[T], Tensor[T])]


  def LSTM7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None, initial_c: Option[Tensor[T]] = None, P: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : Task[(Tensor[T], Tensor[T], Tensor[T])]

}
trait LabelEncoderFree extends Operator {

  def LabelEncoder1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,classes_strings : Option[(Array[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T1],evT2:(UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T2])    : Task[(Tensor[T2])]


  def LabelEncoder2Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,default_float : Option[(Float)] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None,keys_floats : Option[(Array[Float])] = None,keys_int64s : Option[(Array[Int])] = None,keys_strings : Option[(Array[String])] = None,values_floats : Option[(Array[Float])] = None,values_int64s : Option[(Array[Int])] = None,values_strings : Option[(Array[String])] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T1],evT2:(UNil TypeOr String TypeOr Long TypeOr String TypeOr Long TypeOr Float)#check[T2])    : Task[(Tensor[T2])]

}
trait LeakyReluFree extends Operator {

  def LeakyRelu1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def LeakyRelu6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait LessFree extends Operator {

  def Less1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]


  def Less7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]


  def Less9Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]

}
trait LinearClassifierFree extends Operator {

  def LinearClassifier1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : Option[(Array[Float])],intercepts : Option[(Array[Float])] = None,multi_class : Option[(Int)] = None,post_transform : Option[(String)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : Task[(Tensor[T2], Tensor[Float])]

}
trait LinearRegressorFree extends Operator {

  def LinearRegressor1Free[@sp T : Numeric:ClassTag](name: String,coefficients : Option[(Array[Float])] = None,intercepts : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,targets : Option[(Int)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : Task[(Tensor[Float])]

}
trait LogFree extends Operator {

  def Log1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Log6Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait LogSoftmaxFree extends Operator {

  def LogSoftmax1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait LoopFree extends Operator {

  def Loop1Free[@sp I : Numeric:ClassTag,@sp B : Numeric:ClassTag,@sp V : Numeric:ClassTag](name: String,body : Option[(Graph)],M: Option[Tensor[I]] = None, cond: Option[Tensor[B]] = None,v_initial: Seq[Option[Tensor[V]]])
(implicit evI:(UNil TypeOr Long)#check[I],evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : Task[(Tensor[V])]

}
trait LpNormalizationFree extends Operator {

  def LpNormalization1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,p : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait LpPoolFree extends Operator {

  def LpPool1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])] = None,p : Option[(Float)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def LpPool2Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])],p : Option[(Int)] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait MatMulFree extends Operator {

  def MatMul1Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]


  def MatMul9Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]

}
trait MatMulIntegerFree extends Operator {

  def MatMulInteger10Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag,@sp T3 : Numeric:ClassTag](name: String,A: Option[Tensor[T1]], B: Option[Tensor[T2]],a_zero_point: Option[Tensor[T1]] = None, b_zero_point: Option[Tensor[T2]] = None)
(implicit evT1:(UNil TypeOr Byte TypeOr UByte)#check[T1],evT2:(UNil TypeOr Byte TypeOr UByte)#check[T2],evT3:(UNil TypeOr Int)#check[T3])    : Task[(Tensor[T3])]

}
trait MaxFree extends Operator {

  def Max6Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Max8Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait MaxPoolFree extends Operator {

  def MaxPool1Free[@sp T : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def MaxPool8Free[@sp T : Numeric:ClassTag,@sp I : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : Task[(Tensor[T], Tensor[I])]


  def MaxPool10Free[@sp T : Numeric:ClassTag,@sp I : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,ceil_mode : Option[(Int)] = None,dilations : Option[(Array[Int])] = None,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : Task[(Tensor[T], Tensor[I])]

}
trait MaxRoiPoolFree extends Operator {

  def MaxRoiPool1Free[@sp T : Numeric:ClassTag](name: String,pooled_shape : Option[(Array[Int])],spatial_scaleAttr : Option[(Float)] = None,X: Option[Tensor[T]], rois: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait MaxUnpoolFree extends Operator {

  def MaxUnpool9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,kernel_shape : Option[(Array[Int])],pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,X: Option[Tensor[T1]], I: Option[Tensor[T2]],output_shapeInput: Option[Tensor[T2]] = None)
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Long)#check[T2])    : Task[(Tensor[T1])]

}
trait MeanFree extends Operator {

  def Mean6Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Mean8Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait MeanVarianceNormalizationFree extends Operator {

  def MeanVarianceNormalization9Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait MinFree extends Operator {

  def Min6Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Min8Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ModFree extends Operator {

  def Mod10Free[@sp T : Numeric:ClassTag](name: String,fmod : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait MulFree extends Operator {

  def Mul1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Mul6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Mul7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait MultinomialFree extends Operator {

  def Multinomial7Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Int TypeOr Long)#check[T2])    : Task[(Tensor[T2])]

}
trait NegFree extends Operator {

  def Neg1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Neg6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait NonMaxSuppressionFree extends Operator {

  def NonMaxSuppression10Free(name: String,center_point_box : Option[(Int)] = None,boxes: Option[Tensor[Float]], scores: Option[Tensor[Float]],max_output_boxes_per_class: Option[Tensor[Long]] = None, iou_threshold: Option[Tensor[Float]] = None, score_threshold: Option[Tensor[Float]] = None)
    : Task[(Tensor[Long])]

}
trait NonZeroFree extends Operator {

  def NonZero9Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[Long])]

}
trait NormalizerFree extends Operator {

  def Normalizer1Free[@sp T : Numeric:ClassTag](name: String,norm : Option[(String)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : Task[(Tensor[Float])]

}
trait NotFree extends Operator {

  def Not1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T])    : Task[(Tensor[T])]

}
trait OneHotEncoderFree extends Operator {

  def OneHotEncoder1Free[@sp T : Numeric:ClassTag](name: String,cats_int64s : Option[(Array[Int])] = None,cats_strings : Option[(Array[String])] = None,zeros : Option[(Int)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr String TypeOr Long TypeOr Int TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[Float])]

}
trait OneHotFree extends Operator {

  def OneHot9Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag,@sp T3 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,indices: Option[Tensor[T1]], depth: Option[Tensor[T2]], values: Option[Tensor[T3]])
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T2],evT3:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T3])    : Task[(Tensor[T3])]

}
trait OrFree extends Operator {

  def Or1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]


  def Or7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]

}
trait PReluFree extends Operator {

  def PRelu1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]], slope: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]


  def PRelu6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], slope: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]


  def PRelu7Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], slope: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]


  def PRelu9Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], slope: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])]

}
trait PadFree extends Operator {

  def Pad1Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,paddings : Option[(Array[Int])],value : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Pad2Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,pads : Option[(Array[Int])],value : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait PowFree extends Operator {

  def Pow1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,X: Option[Tensor[T]], Y: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Pow7Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]], Y: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait QLinearConvFree extends Operator {

  def QLinearConv10Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag,@sp T3 : Numeric:ClassTag,@sp T4 : Numeric:ClassTag](name: String,auto_pad : Option[(String)] = None,dilations : Option[(Array[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Array[Int])] = None,pads : Option[(Array[Int])] = None,strides : Option[(Array[Int])] = None,x: Option[Tensor[T1]], x_scale: Option[Tensor[Float]], x_zero_point: Option[Tensor[T1]], w: Option[Tensor[T2]], w_scale: Option[Tensor[Float]], w_zero_point: Option[Tensor[T2]], y_scale: Option[Tensor[Float]], y_zero_point: Option[Tensor[T3]],B: Option[Tensor[T4]] = None)
(implicit evT1:(UNil TypeOr Byte TypeOr UByte)#check[T1],evT2:(UNil TypeOr Byte TypeOr UByte)#check[T2],evT3:(UNil TypeOr Byte TypeOr UByte)#check[T3],evT4:(UNil TypeOr Int)#check[T4])    : Task[(Tensor[T3])]

}
trait QLinearMatMulFree extends Operator {

  def QLinearMatMul10Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag,@sp T3 : Numeric:ClassTag](name: String,a: Option[Tensor[T1]], a_scale: Option[Tensor[Float]], a_zero_point: Option[Tensor[T1]], b: Option[Tensor[T2]], b_scale: Option[Tensor[Float]], b_zero_point: Option[Tensor[T2]], y_scale: Option[Tensor[Float]], y_zero_point: Option[Tensor[T3]])
(implicit evT1:(UNil TypeOr Byte TypeOr UByte)#check[T1],evT2:(UNil TypeOr Byte TypeOr UByte)#check[T2],evT3:(UNil TypeOr Byte TypeOr UByte)#check[T3])    : Task[(Tensor[T3])]

}
trait QuantizeLinearFree extends Operator {

  def QuantizeLinear10Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,x: Option[Tensor[T1]], y_scale: Option[Tensor[Float]],y_zero_point: Option[Tensor[T2]] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Int)#check[T1],evT2:(UNil TypeOr Byte TypeOr UByte)#check[T2])    : Task[(Tensor[T2])]

}
trait RNNFree extends Operator {

  def RNN1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : Task[(Tensor[T], Tensor[T])]


  def RNN7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,activation_alpha : Option[(Array[Float])] = None,activation_beta : Option[(Array[Float])] = None,activations : Option[(Array[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,X: Option[Tensor[T]], W: Option[Tensor[T]], R: Option[Tensor[T]],B: Option[Tensor[T]] = None, sequence_lens: Option[Tensor[T1]] = None, initial_h: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : Task[(Tensor[T], Tensor[T])]

}
trait RandomNormalFree extends Operator {

  def RandomNormal1Free[@sp T : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None,shape : Option[(Array[Int])])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait RandomNormalLikeFree extends Operator {

  def RandomNormalLike1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : Task[(Tensor[T2])]

}
trait RandomUniformFree extends Operator {

  def RandomUniform1Free[@sp T : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None,shape : Option[(Array[Int])])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait RandomUniformLikeFree extends Operator {

  def RandomUniformLike1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None,input: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : Task[(Tensor[T2])]

}
trait ReciprocalFree extends Operator {

  def Reciprocal1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Reciprocal6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceL1Free extends Operator {

  def ReduceL11Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceL2Free extends Operator {

  def ReduceL21Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceLogSumExpFree extends Operator {

  def ReduceLogSumExp1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceLogSumFree extends Operator {

  def ReduceLogSum1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceMaxFree extends Operator {

  def ReduceMax1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceMeanFree extends Operator {

  def ReduceMean1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceMinFree extends Operator {

  def ReduceMin1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceProdFree extends Operator {

  def ReduceProd1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceSumFree extends Operator {

  def ReduceSum1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReduceSumSquareFree extends Operator {

  def ReduceSumSquare1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,keepdims : Option[(Int)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReluFree extends Operator {

  def Relu1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Relu6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ReshapeFree extends Operator {

  def Reshape1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,shape : Option[(Array[Int])] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Reshape5Free[@sp T : Numeric:ClassTag](name: String,data: Option[Tensor[T]], shape: Option[Tensor[Long]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait ResizeFree extends Operator {

  def Resize10Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,X: Option[Tensor[T]], scales: Option[Tensor[Float]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait ReverseSequenceFree extends Operator {

  def ReverseSequence10Free[@sp T : Numeric:ClassTag](name: String,batch_axis : Option[(Int)] = None,time_axis : Option[(Int)] = None,input: Option[Tensor[T]], sequence_lens: Option[Tensor[Long]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait RoiAlignFree extends Operator {

  def RoiAlign10Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,mode : Option[(String)] = None,output_height : Option[(Int)] = None,output_width : Option[(Int)] = None,sampling_ratio : Option[(Int)] = None,spatial_scaleAttr : Option[(Float)] = None,X: Option[Tensor[T1]], rois: Option[Tensor[T1]], batch_indices: Option[Tensor[T2]])
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Long)#check[T2])    : Task[(Tensor[T1])]

}
trait SVMClassifierFree extends Operator {

  def SVMClassifier1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,classlabels_ints : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,post_transform : Option[(String)] = None,prob_a : Option[(Array[Float])] = None,prob_b : Option[(Array[Float])] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None,vectors_per_class : Option[(Array[Int])] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : Task[(Tensor[T2], Tensor[Float])]

}
trait SVMRegressorFree extends Operator {

  def SVMRegressor1Free[@sp T : Numeric:ClassTag](name: String,coefficients : Option[(Array[Float])] = None,kernel_params : Option[(Array[Float])] = None,kernel_type : Option[(String)] = None,n_supports : Option[(Int)] = None,one_class : Option[(Int)] = None,post_transform : Option[(String)] = None,rho : Option[(Array[Float])] = None,support_vectors : Option[(Array[Float])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : Task[(Tensor[Float])]

}
trait ScalerFree extends Operator {

  def Scaler1Free[@sp T : Numeric:ClassTag](name: String,offset : Option[(Array[Float])] = None,scaleAttr : Option[(Array[Float])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : Task[(Tensor[Float])]

}
trait ScanFree extends Operator {

  def Scan9Free[@sp V : Numeric:ClassTag](name: String,body : Option[(Graph)],num_scan_inputs : Option[(Int)],scan_input_axes : Option[(Array[Int])] = None,scan_input_directions : Option[(Array[Int])] = None,scan_output_axes : Option[(Array[Int])] = None,scan_output_directions : Option[(Array[Int])] = None,initial_state_and_scan_inputs: Seq[Option[Tensor[V]]])
(implicit evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : Task[(Tensor[V])]

}
trait ScatterFree extends Operator {

  def Scatter9Free[@sp T : Numeric:ClassTag,@sp Tind : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,data: Option[Tensor[T]], indices: Option[Tensor[Tind]], updates: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : Task[(Tensor[T])]

}
trait SeluFree extends Operator {

  def Selu1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Array[Int])] = None,gamma : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Selu6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait ShapeFree extends Operator {

  def Shape1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : Task[(Tensor[T1])]

}
trait ShrinkFree extends Operator {

  def Shrink9Free[@sp T : Numeric:ClassTag](name: String,bias : Option[(Float)] = None,lambd : Option[(Float)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SigmoidFree extends Operator {

  def Sigmoid1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Sigmoid6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SignFree extends Operator {

  def Sign9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SinFree extends Operator {

  def Sin7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SinhFree extends Operator {

  def Sinh9Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SizeFree extends Operator {

  def Size1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : Task[(Tensor[T1])]

}
trait SliceFree extends Operator {

  def Slice1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,ends : Option[(Array[Int])],starts : Option[(Array[Int])],data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Slice10Free[@sp T : Numeric:ClassTag,@sp Tind : Numeric:ClassTag](name: String,data: Option[Tensor[T]], starts: Option[Tensor[Tind]], ends: Option[Tensor[Tind]],axes: Option[Tensor[Tind]] = None, steps: Option[Tensor[Tind]] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : Task[(Tensor[T])]

}
trait SoftmaxFree extends Operator {

  def Softmax1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SoftplusFree extends Operator {

  def Softplus1Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SoftsignFree extends Operator {

  def Softsign1Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SpaceToDepthFree extends Operator {

  def SpaceToDepth1Free[@sp T : Numeric:ClassTag](name: String,blocksize : Option[(Int)],input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait SplitFree extends Operator {

  def Split1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None,input: Option[Tensor[T]],split: Option[Tensor[T]] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Split2Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,splitAttr : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait SqrtFree extends Operator {

  def Sqrt1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Sqrt6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SqueezeFree extends Operator {

  def Squeeze1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait StringNormalizerFree extends Operator {

  def StringNormalizer10Free(name: String,case_change_action : Option[(String)] = None,is_case_sensitive : Option[(Int)] = None,locale : Option[(String)] = None,stopwords : Option[(Array[String])] = None,X: Option[Tensor[String]])
    : Task[(Tensor[String])]

}
trait SubFree extends Operator {

  def Sub1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Sub6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Sub7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait SumFree extends Operator {

  def Sum6Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Sum8Free[@sp T : Numeric:ClassTag](name: String,data_0: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait TanFree extends Operator {

  def Tan7Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait TanhFree extends Operator {

  def Tanh1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]


  def Tanh6Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait TfIdfVectorizerFree extends Operator {

  def TfIdfVectorizer9Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,max_gram_length : Option[(Int)],max_skip_count : Option[(Int)],min_gram_length : Option[(Int)],mode : Option[(String)],ngram_counts : Option[(Array[Int])],ngram_indexes : Option[(Array[Int])],pool_int64s : Option[(Array[Int])] = None,pool_strings : Option[(Array[String])] = None,weights : Option[(Array[Float])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr String TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Float)#check[T1])    : Task[(Tensor[T1])]

}
trait ThresholdedReluFree extends Operator {

  def ThresholdedRelu10Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])]

}
trait TileFree extends Operator {

  def Tile1Free[@sp T : Numeric:ClassTag](name: String,input: Option[Tensor[T]], tiles: Option[Tensor[T]], axis: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Tile6Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,input: Option[Tensor[T]], repeats: Option[Tensor[T1]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : Task[(Tensor[T])]

}
trait TopKFree extends Operator {

  def TopK1Free[@sp T : Numeric:ClassTag,@sp I : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,k : Option[(Int)],X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : Task[(Tensor[T], Tensor[I])]


  def TopK10Free[@sp T : Numeric:ClassTag,@sp I : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,X: Option[Tensor[T]], K: Option[Tensor[Long]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : Task[(Tensor[T], Tensor[I])]

}
trait TransposeFree extends Operator {

  def Transpose1Free[@sp T : Numeric:ClassTag](name: String,perm : Option[(Array[Int])] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait TreeEnsembleClassifierFree extends Operator {

  def TreeEnsembleClassifier1Free[@sp T1 : Numeric:ClassTag,@sp T2 : Numeric:ClassTag](name: String,base_values : Option[(Array[Float])] = None,class_ids : Option[(Array[Int])] = None,class_nodeids : Option[(Array[Int])] = None,class_treeids : Option[(Array[Int])] = None,class_weights : Option[(Array[Float])] = None,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,X: Option[Tensor[T1]])
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : Task[(Tensor[T2], Tensor[Float])]

}
trait TreeEnsembleRegressorFree extends Operator {

  def TreeEnsembleRegressor1Free[@sp T : Numeric:ClassTag](name: String,aggregate_function : Option[(String)] = None,base_values : Option[(Array[Float])] = None,n_targets : Option[(Int)] = None,nodes_falsenodeids : Option[(Array[Int])] = None,nodes_featureids : Option[(Array[Int])] = None,nodes_hitrates : Option[(Array[Float])] = None,nodes_missing_value_tracks_true : Option[(Array[Int])] = None,nodes_modes : Option[(Array[String])] = None,nodes_nodeids : Option[(Array[Int])] = None,nodes_treeids : Option[(Array[Int])] = None,nodes_truenodeids : Option[(Array[Int])] = None,nodes_values : Option[(Array[Float])] = None,post_transform : Option[(String)] = None,target_ids : Option[(Array[Int])] = None,target_nodeids : Option[(Array[Int])] = None,target_treeids : Option[(Array[Int])] = None,target_weights : Option[(Array[Float])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : Task[(Tensor[Float])]

}
trait UnsqueezeFree extends Operator {

  def Unsqueeze1Free[@sp T : Numeric:ClassTag](name: String,axes : Option[(Array[Int])],data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait UpsampleFree extends Operator {

  def Upsample1Free[@sp T : Numeric:ClassTag](name: String,height_scaleAttr : Option[(Float)],mode : Option[(String)] = None,width_scaleAttr : Option[(Float)],X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Upsample7Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,scaleAttrs : Option[(Array[Float])],X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Upsample9Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,X: Option[Tensor[T]], scales: Option[Tensor[Float]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]


  def Upsample10Free[@sp T : Numeric:ClassTag](name: String,mode : Option[(String)] = None,X: Option[Tensor[T]], scales: Option[Tensor[Float]])
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait WhereFree extends Operator {

  def Where9Free[@sp B : Numeric:ClassTag,@sp T : Numeric:ClassTag](name: String,condition: Option[Tensor[B]], X: Option[Tensor[T]], Y: Option[Tensor[T]])
(implicit evB:(UNil TypeOr Boolean)#check[B],evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : Task[(Tensor[T])]

}
trait XorFree extends Operator {

  def Xor1Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]


  def Xor7Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T1])]

}
trait ZipMapFree extends Operator {

  def ZipMap1Free[@sp T : Numeric:ClassTag](name: String,classlabels_int64s : Option[(Array[Int])] = None,classlabels_strings : Option[(Array[String])] = None,X: Option[Tensor[Float]])
(implicit evT:(UNil TypeOr Seq[Map[String, Float]] TypeOr Seq[Map[Long, Float]])#check[T])    : Task[(T)]

}}
