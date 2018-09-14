package org.emergentorder

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
  type Tensor[U, J <: XInt] = Tuple2[Seq[U], Seq[J]]
  trait Operator
trait Graph
object UnionType {

      trait inv[-A] {}

      sealed trait OrR {
        type L <: OrR
        type R
        type invIntersect
        type intersect
      }

      sealed class TypeOr[A <: OrR, B] extends OrR {
        type L = A
        type R = B

        type intersect = (L#intersect with R)
        type invIntersect = (L#invIntersect with inv[R])
        type check[X] = invIntersect <:< inv[X]
      }

      object UNil extends OrR {
        type intersect = Any
        type invIntersect = inv[Nothing]
      }
      type UNil = UNil.type

    }
    
    import UnionType._
    trait DataSource {
  def inputData[T : Numeric:ClassTag:Field, J <: XInt](implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Tensor[T, J]
  def getParams[T : Numeric:ClassTag:Field, J <: XInt](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Tensor[T, J]
  def getAttributes[T : Numeric:ClassTag:Field, J <: XInt](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Tensor[T, J]
}
trait SVMClassifier extends Operator {

  def SVMClassifier1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classlabels_ints : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,coefficients : Option[(Seq[Float])] = None,kernel_params : Option[(Seq[Float])] = None,kernel_type : Option[(String)] = None,post_transform : Option[(String)] = None,prob_a : Option[(Seq[Float])] = None,prob_b : Option[(Seq[Float])] = None,rho : Option[(Seq[Float])] = None,support_vectors : Option[(Seq[Float])] = None,vectors_per_class : Option[(Seq[Int])] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2, J], Tensor[Float, J])

}
trait PRelu extends Operator {

  def PRelu1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def PRelu6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def PRelu7[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, slope: Tensor[T, J], slopename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Size extends Operator {

  def Size1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : (Tensor[T1, J])

}
trait DictVectorizer extends Operator {

  def DictVectorizer1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: T1, Xname: String,int64_vocabulary : Option[(Seq[Int])] = None,string_vocabulary : Option[(Seq[String])] = None)
(implicit evT1:(UNil TypeOr Map[String, Long] TypeOr Map[Long, String] TypeOr Map[Long, Float] TypeOr Map[Long, Double] TypeOr Map[String, Float] TypeOr Map[String, Double])#check[T1],evT2:(UNil TypeOr Long TypeOr Float TypeOr Double TypeOr String)#check[T2])    : (Tensor[T2, J])

}
trait Scale extends Operator {

  def Scale1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Multinomial extends Operator {

  def Multinomial7[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,sample_size : Option[(Int)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T1],evT2:(UNil TypeOr Int TypeOr Long)#check[T2])    : (Tensor[T2, J])

}
trait Normalizer extends Operator {

  def Normalizer1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,norm : Option[(String)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float, J])

}
trait RandomNormal extends Operator {

  def RandomNormal1[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait MaxRoiPool extends Operator {

  def MaxRoiPool1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, rois: Tensor[T, J], roisname: String,pooled_shape : (Seq[Int]),spatial_scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait TreeEnsembleClassifier extends Operator {

  def TreeEnsembleClassifier1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,base_values : Option[(Seq[Float])] = None,class_ids : Option[(Seq[Int])] = None,class_nodeids : Option[(Seq[Int])] = None,class_treeids : Option[(Seq[Int])] = None,class_weights : Option[(Seq[Float])] = None,classlabels_int64s : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,nodes_falsenodeids : Option[(Seq[Int])] = None,nodes_featureids : Option[(Seq[Int])] = None,nodes_hitrates : Option[(Seq[Float])] = None,nodes_missing_value_tracks_true : Option[(Seq[Int])] = None,nodes_modes : Option[(Seq[String])] = None,nodes_nodeids : Option[(Seq[Int])] = None,nodes_treeids : Option[(Seq[Int])] = None,nodes_truenodeids : Option[(Seq[Int])] = None,nodes_values : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2, J], Tensor[Float, J])

}
trait Min extends Operator {

  def Min1[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Min6[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Min8[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait If extends Operator {

  def If1[B : Numeric:ClassTag:Field,V : Numeric:ClassTag:Field, J <: XInt](name: String,cond: Tensor[B, J], condname: String,else_branch : (Graph),then_branch : (Graph))
(implicit evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : (Tensor[V, J])

}
trait Pad extends Operator {

  def Pad1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(String)] = None,paddings : (Seq[Int]),value : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Pad2[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,mode : Option[(String)] = None,pads : (Seq[Int]),value : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Sqrt extends Operator {

  def Sqrt1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Sqrt6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ReduceMin extends Operator {

  def ReduceMin1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait LSTM extends Operator {

  def LSTM1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def LSTM7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None, initial_c: Option[Tensor[T, J]] = None, P: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,input_forget : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T, J], Tensor[T, J], Tensor[T, J])

}
trait And extends Operator {

  def And1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])


  def And7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])

}
trait Elu extends Operator {

  def Elu1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Elu6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ParametricSoftplus extends Operator {

  def ParametricSoftplus1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait RandomNormalLike extends Operator {

  def RandomNormalLike1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,mean : Option[(Float)] = None,scaleAttr : Option[(Float)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : (Tensor[T2, J])

}
trait LinearRegressor extends Operator {

  def LinearRegressor1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,coefficients : Option[(Seq[Float])] = None,intercepts : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None,targets : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float, J])

}
trait Shape extends Operator {

  def Shape1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : (Tensor[T1, J])

}
trait CastMap extends Operator {

  def CastMap1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: T1, Xname: String,cast_to : Option[(String)] = None,map_form : Option[(String)] = None,max_map : Option[(Int)] = None)
(implicit evT1:(UNil TypeOr Map[Long, String] TypeOr Map[Long, Float])#check[T1],evT2:(UNil TypeOr String TypeOr Float TypeOr Long)#check[T2])    : (Tensor[T2, J])

}
trait LinearClassifier extends Operator {

  def LinearClassifier1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classlabels_ints : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None,coefficients : (Seq[Float]),intercepts : Option[(Seq[Float])] = None,multi_class : Option[(Int)] = None,post_transform : Option[(String)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2, J], Tensor[Float, J])

}
trait Acos extends Operator {

  def Acos7[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait CategoryMapper extends Operator {

  def CategoryMapper1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,cats_int64s : Option[(Seq[Int])] = None,cats_strings : Option[(Seq[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2, J])

}
trait ReduceL1 extends Operator {

  def ReduceL11[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Floor extends Operator {

  def Floor1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Floor6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Softsign extends Operator {

  def Softsign1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Softmax extends Operator {

  def Softmax1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait MaxPool extends Operator {

  def MaxPool1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def MaxPool8[T : Numeric:ClassTag:Field,I : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,storage_order : Option[(Int)] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : (Tensor[T, J], Tensor[I, J])

}
trait Gather extends Operator {

  def Gather1[T : Numeric:ClassTag:Field,Tind : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, indices: Tensor[Tind, J], indicesname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    : (Tensor[T, J])

}
trait Ceil extends Operator {

  def Ceil1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Ceil6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait SVMRegressor extends Operator {

  def SVMRegressor1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,coefficients : Option[(Seq[Float])] = None,kernel_params : Option[(Seq[Float])] = None,kernel_type : Option[(String)] = None,n_supports : Option[(Int)] = None,one_class : Option[(Int)] = None,post_transform : Option[(String)] = None,rho : Option[(Seq[Float])] = None,support_vectors : Option[(Seq[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float, J])

}
trait ConstantFill extends Operator {

  def ConstantFill1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Option[Tensor[T1, J]] = None,dtype : Option[(Int)] = None,extra_shape : Option[(Seq[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Seq[Int])] = None,value : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr Float TypeOr Int TypeOr Long TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float TypeOr Int TypeOr Long TypeOr Boolean)#check[T2])    : (Tensor[T2, J])

}
trait ReduceLogSum extends Operator {

  def ReduceLogSum1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Sin extends Operator {

  def Sin7[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait FeatureVectorizer extends Operator {

  def FeatureVectorizer1[J <:XInt](name: String)
    : (Tensor[Float, J])

}
trait BatchNormalization extends Operator {

  def BatchNormalization1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,consumed_inputs : (Seq[Int]),epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def BatchNormalization6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Float)] = None,is_test : Option[(Int)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])


  def BatchNormalization7[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String, mean: Tensor[T, J], meanname: String, someVar: Tensor[T, J], varname: String,epsilon : Option[(Float)] = None,momentum : Option[(Float)] = None,spatial : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J], Tensor[T, J])

}
trait LpNormalization extends Operator {

  def LpNormalization1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None,p : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Scan extends Operator {

  def Scan8[I : Numeric:ClassTag:Field,V : Numeric:ClassTag:Field, J <: XInt](name: String,sequence_lens: Option[Tensor[I, J]] = None,body : (Graph),directions : Option[(Seq[Int])] = None,num_scan_inputs : (Int))
(implicit evI:(UNil TypeOr Long)#check[I],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : (Tensor[V, J])

}
trait ReduceSumSquare extends Operator {

  def ReduceSumSquare1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Greater extends Operator {

  def Greater1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])


  def Greater7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])

}
trait Constant extends Operator {

  def Constant1[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Gemm extends Operator {

  def Gemm1[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Gemm6[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Gemm7[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String, C: Tensor[T, J], Cname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Not extends Operator {

  def Not1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T])    : (Tensor[T, J])

}
trait ThresholdedRelu extends Operator {

  def ThresholdedRelu1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait TopK extends Operator {

  def TopK1[T : Numeric:ClassTag:Field,I : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,axis : Option[(Int)] = None,k : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evI:(UNil TypeOr Long)#check[I])    : (Tensor[T, J], Tensor[I, J])

}
trait ScaledTanh extends Operator {

  def ScaledTanh1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait InstanceNormalization extends Operator {

  def InstanceNormalization1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,consumed_inputs : Option[(Seq[Int])] = None,epsilon : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def InstanceNormalization6[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, scale: Tensor[T, J], scalename: String, B: Tensor[T, J], Bname: String,epsilon : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Cast extends Operator {

  def Cast1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (String))
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : (Tensor[T2, J])


  def Cast6[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,to : (Int))
(implicit evT1:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Boolean)#check[T2])    : (Tensor[T2, J])

}
trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Max extends Operator {

  def Max1[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Max6[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Max8[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Relu extends Operator {

  def Relu1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Relu6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Imputer extends Operator {

  def Imputer1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,imputed_value_floats : Option[(Seq[Float])] = None,imputed_value_int64s : Option[(Seq[Int])] = None,replaced_value_float : Option[(Float)] = None,replaced_value_int64 : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[T, J])

}
trait ConvTranspose extends Operator {

  def ConvTranspose1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Seq[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Seq[Int])] = None,output_padding : Option[(Seq[Int])] = None,output_shape : Option[(Seq[Int])] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait LeakyRelu extends Operator {

  def LeakyRelu1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def LeakyRelu6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Mean extends Operator {

  def Mean1[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Mean6[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Mean8[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Scaler extends Operator {

  def Scaler1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,offset : Option[(Seq[Float])] = None,scaleAttr : Option[(Seq[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float, J])

}
trait GRU extends Operator {

  def GRU1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T, J], Tensor[T, J])


  def GRU3[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T, J], Tensor[T, J])


  def GRU7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,linear_before_reset : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T, J], Tensor[T, J])

}
trait Transpose extends Operator {

  def Transpose1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,perm : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait Equal extends Operator {

  def Equal1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])


  def Equal7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])

}
trait TreeEnsembleRegressor extends Operator {

  def TreeEnsembleRegressor1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,aggregate_function : Option[(String)] = None,base_values : Option[(Seq[Float])] = None,n_targets : Option[(Int)] = None,nodes_falsenodeids : Option[(Seq[Int])] = None,nodes_featureids : Option[(Seq[Int])] = None,nodes_hitrates : Option[(Seq[Float])] = None,nodes_missing_value_tracks_true : Option[(Seq[Int])] = None,nodes_modes : Option[(Seq[String])] = None,nodes_nodeids : Option[(Seq[Int])] = None,nodes_treeids : Option[(Seq[Int])] = None,nodes_truenodeids : Option[(Seq[Int])] = None,nodes_values : Option[(Seq[Float])] = None,post_transform : Option[(String)] = None,target_ids : Option[(Seq[Int])] = None,target_nodeids : Option[(Seq[Int])] = None,target_treeids : Option[(Seq[Int])] = None,target_weights : Option[(Seq[Float])] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[Float, J])

}
trait GRUUnit extends Operator {

  def GRUUnit1[T : Numeric:ClassTag:Field, J <: XInt](name: String,hidden_prev: Tensor[T, J], hidden_prevname: String, gates: Tensor[T, J], gatesname: String, seq_lengths: Tensor[T, J], seq_lengthsname: String, t: Tensor[T, J], tname: String,drop_states : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Xor extends Operator {

  def Xor1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])


  def Xor7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])

}
trait Tan extends Operator {

  def Tan7[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ImageScaler extends Operator {

  def ImageScaler1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,bias : Option[(Seq[Float])] = None,scaleAttr : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait LabelEncoder extends Operator {

  def LabelEncoder1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T1, J], Xname: String,classes_strings : Option[(Seq[String])] = None,default_int64 : Option[(Int)] = None,default_string : Option[(String)] = None)
(implicit evT1:(UNil TypeOr String TypeOr Long)#check[T1],evT2:(UNil TypeOr String TypeOr Long)#check[T2])    : (Tensor[T2, J])

}
trait LpPool extends Operator {

  def LpPool1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : Option[(Seq[Int])] = None,p : Option[(Float)] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def LpPool2[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),p : Option[(Int)] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait LogSoftmax extends Operator {

  def LogSoftmax1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Identity extends Operator {

  def Identity1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait Mul extends Operator {

  def Mul1[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Mul6[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Mul7[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Abs extends Operator {

  def Abs1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Abs6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Cos extends Operator {

  def Cos7[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ReduceL2 extends Operator {

  def ReduceL21[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Clip extends Operator {

  def Clip1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None,max : Option[(Float)] = None,min : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Clip6[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,max : Option[(Float)] = None,min : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Tanh extends Operator {

  def Tanh1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Tanh6[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Less extends Operator {

  def Less1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])


  def Less7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])

}
trait Slice extends Operator {

  def Slice1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,ends : (Seq[Int]),starts : (Seq[Int]))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait Tile extends Operator {

  def Tile1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, tiles: Tensor[T, J], tilesname: String, axis: Tensor[T, J], axisname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])


  def Tile6[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, repeats: Tensor[T1, J], repeatsname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evT1:(UNil TypeOr Long)#check[T1])    : (Tensor[T, J])

}
trait Conv extends Operator {

  def Conv1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String,B: Option[Tensor[T, J]] = None,auto_pad : Option[(String)] = None,dilations : Option[(Seq[Int])] = None,group : Option[(Int)] = None,kernel_shape : Option[(Seq[Int])] = None,pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait RNN extends Operator {

  def RNN1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None,output_sequence : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T, J], Tensor[T, J])


  def RNN7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, W: Tensor[T, J], Wname: String, R: Tensor[T, J], Rname: String,B: Option[Tensor[T, J]] = None, sequence_lens: Option[Tensor[T1, J]] = None, initial_h: Option[Tensor[T, J]] = None,activation_alpha : Option[(Seq[Float])] = None,activation_beta : Option[(Seq[Float])] = None,activations : Option[(Seq[String])] = None,clip : Option[(Float)] = None,direction : Option[(String)] = None,hidden_size : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Int)#check[T1])    : (Tensor[T, J], Tensor[T, J])

}
trait RandomUniform extends Operator {

  def RandomUniform1[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Dropout extends Operator {

  def Dropout1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J], Tensor[T, J])


  def Dropout6[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J], Tensor[T, J])


  def Dropout7[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,ratio : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J], Tensor[T, J])

}
trait Concat extends Operator {

  def Concat1[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])


  def Concat4[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait ReduceProd extends Operator {

  def ReduceProd1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ReduceSum extends Operator {

  def ReduceSum1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Or extends Operator {

  def Or1[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])


  def Or7[T : Numeric:ClassTag:Field,T1 : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Boolean)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : (Tensor[T1, J])

}
trait Reshape extends Operator {

  def Reshape1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,consumed_inputs : Option[(Seq[Int])] = None,shape : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])


  def Reshape5[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String, shape: Tensor[Long, J], shapename: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait Loop extends Operator {

  def Loop1[I : Numeric:ClassTag:Field,B : Numeric:ClassTag:Field,V : Numeric:ClassTag:Field, J <: XInt](name: String,M: I, Mname: String, cond: B, condname: String,body : (Graph))
(implicit evI:(UNil TypeOr Long)#check[I],evB:(UNil TypeOr Boolean)#check[B],evV:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[V])    : (Tensor[V, J])

}
trait Expand extends Operator {

  def Expand8[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String, shape: Tensor[Long, J], shapename: String)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait ReduceMean extends Operator {

  def ReduceMean1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Split extends Operator {

  def Split1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,split: Option[Tensor[T, J]] = None,axis : Option[(Int)] = None,splitAttr : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])


  def Split2[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None,splitAttr : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait Sum extends Operator {

  def Sum1[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Sum6[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Sum8[T : Numeric:ClassTag:Field, J <: XInt](name: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Log extends Operator {

  def Log1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Log6[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait GlobalLpPool extends Operator {

  def GlobalLpPool1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def GlobalLpPool2[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,p : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Asin extends Operator {

  def Asin7[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait HardSigmoid extends Operator {

  def HardSigmoid1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def HardSigmoid6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Sub extends Operator {

  def Sub1[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Sub6[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Sub7[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait SpaceToDepth extends Operator {

  def SpaceToDepth1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (Int))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait GivenTensorFill extends Operator {

  def GivenTensorFill1[T : Numeric:ClassTag:Field, J <: XInt](name: String,shapeInput: Option[Tensor[T, J]] = None,extra_shape : Option[(Seq[Int])] = None,input_as_shape : Option[(Int)] = None,shape : Option[(Seq[Int])] = None,values : Option[(Seq[Float])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,across_channels : Option[(Int)] = None,normalize_variance : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ArrayFeatureExtractor extends Operator {

  def ArrayFeatureExtractor1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[Long, J], Yname: String)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int TypeOr String)#check[T])    : (Tensor[T, J])

}
trait Selu extends Operator {

  def Selu1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,consumed_inputs : Option[(Seq[Int])] = None,gamma : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Selu6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,gamma : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Pow extends Operator {

  def Pow1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Pow7[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String, Y: Tensor[T, J], Yname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ArgMax extends Operator {

  def ArgMax1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[Long, J])

}
trait Atan extends Operator {

  def Atan7[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait AveragePool extends Operator {

  def AveragePool1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def AveragePool7[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,auto_pad : Option[(String)] = None,count_include_pad : Option[(Int)] = None,kernel_shape : (Seq[Int]),pads : Option[(Seq[Int])] = None,strides : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Hardmax extends Operator {

  def Hardmax1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Squeeze extends Operator {

  def Squeeze1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait Div extends Operator {

  def Div1[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Div6[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Div7[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Neg extends Operator {

  def Neg1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : (Tensor[T, J])


  def Neg6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float TypeOr Int TypeOr Byte TypeOr Short TypeOr Long TypeOr Float16 TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Unsqueeze extends Operator {

  def Unsqueeze1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : (Seq[Int]))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait GlobalMaxPool extends Operator {

  def GlobalMaxPool1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait MatMul extends Operator {

  def MatMul1[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Exp extends Operator {

  def Exp1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Exp6[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait RandomUniformLike extends Operator {

  def RandomUniformLike1[T1 : Numeric:ClassTag:Field,T2 : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T1, J], inputname: String,dtype : Option[(Int)] = None,high : Option[(Float)] = None,low : Option[(Float)] = None,seed : Option[(Float)] = None)
(implicit evT1:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T1],evT2:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T2])    : (Tensor[T2, J])

}
trait ZipMap extends Operator {

  def ZipMap1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[Float, J], Xname: String,classlabels_int64s : Option[(Seq[Int])] = None,classlabels_strings : Option[(Seq[String])] = None)
(implicit evT:(UNil TypeOr Seq[Map[String, Float]] TypeOr Seq[Map[Long, Float]])#check[T])    : (T)

}
trait GlobalAveragePool extends Operator {

  def GlobalAveragePool1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Binarizer extends Operator {

  def Binarizer1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,threshold : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float TypeOr Double TypeOr Long TypeOr Int)#check[T])    : (Tensor[T, J])

}
trait DepthToSpace extends Operator {

  def DepthToSpace1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,blocksize : (Int))
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait Add extends Operator {

  def Add1[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Add6[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Add7[T : Numeric:ClassTag:Field, J <: XInt](name: String,A: Tensor[T, J], Aname: String, B: Tensor[T, J], Bname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Sigmoid extends Operator {

  def Sigmoid1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Sigmoid6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ArgMin extends Operator {

  def ArgMin1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axis : Option[(Int)] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[Long, J])

}
trait Softplus extends Operator {

  def Softplus1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait OneHotEncoder extends Operator {

  def OneHotEncoder1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,cats_int64s : Option[(Seq[Int])] = None,cats_strings : Option[(Seq[String])] = None,zeros : Option[(Int)] = None)
(implicit evT:(UNil TypeOr String TypeOr Long TypeOr Int TypeOr Float TypeOr Double)#check[T])    : (Tensor[Float, J])

}
trait Upsample extends Operator {

  def Upsample1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,height_scaleAttr : (Float),mode : Option[(String)] = None,width_scaleAttr : (Float))
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])


  def Upsample7[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,mode : Option[(String)] = None,scaleAttrs : (Seq[Float]))
(implicit evT:(UNil TypeOr Boolean TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])    : (Tensor[T, J])

}
trait Flatten extends Operator {

  def Flatten1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,axis : Option[(Int)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Affine extends Operator {

  def Affine1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait LRN extends Operator {

  def LRN1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,bias : Option[(Float)] = None,size : (Int))
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Reciprocal extends Operator {

  def Reciprocal1[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String,consumed_inputs : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])


  def Reciprocal6[T : Numeric:ClassTag:Field, J <: XInt](name: String,X: Tensor[T, J], Xname: String)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait Crop extends Operator {

  def Crop1[T : Numeric:ClassTag:Field, J <: XInt](name: String,input: Tensor[T, J], inputname: String,border : Option[(Seq[Int])] = None,scaleAttr : Option[(Seq[Int])] = None)
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}
trait ReduceMax extends Operator {

  def ReduceMax1[T : Numeric:ClassTag:Field, J <: XInt](name: String,data: Tensor[T, J], dataname: String,axes : Option[(Seq[Int])] = None,keepdims : Option[(Int)] = None)
(implicit evT:(UNil TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : (Tensor[T, J])

}}
