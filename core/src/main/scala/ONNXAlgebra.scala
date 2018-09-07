package org.emergentorder

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
trait Equal extends Operator {

  def Equal1(name: String,A: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long], Aname: String, B: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Boolean])


  def Equal7(name: String,A: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long], Aname: String, B: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long], Bname: String)
    : (Tensor[Boolean])

}
trait LpNormalization extends Operator {

  def LpNormalization1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None,p : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Crop extends Operator {

  def Crop1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,border : Option[(Seq[String])] = None,scaleAttr : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait DepthToSpace extends Operator {

  def DepthToSpace1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,blocksize : (String))
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Dropout extends Operator {

  def Dropout1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,consumed_inputs : Option[(Seq[String])] = None,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Dropout6(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,is_test : Option[(String)] = None,ratio : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Dropout7(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,ratio : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Max extends Operator {

  def Max1(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Max6(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ReduceMean extends Operator {

  def ReduceMean1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Cos extends Operator {

  def Cos7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait SpaceToDepth extends Operator {

  def SpaceToDepth1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,blocksize : (String))
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Atan extends Operator {

  def Atan7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Log extends Operator {

  def Log1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Log6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait RandomUniform extends Operator {

  def RandomUniform1(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Or extends Operator {

  def Or1(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Boolean])


  def Or7(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String)
    : (Tensor[Boolean])

}
trait Min extends Operator {

  def Min1(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Min6(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait SVMRegressor extends Operator {

  def SVMRegressor1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,coefficients : Option[(Seq[Int])] = None,kernel_params : Option[(Seq[Int])] = None,kernel_type : Option[(Tensor[_])] = None,n_supports : Option[(String)] = None,one_class : Option[(String)] = None,post_transform : Option[(Tensor[_])] = None,rho : Option[(Seq[Int])] = None,support_vectors : Option[(Seq[Int])] = None)
    : (Tensor[Float])

}
trait ParametricSoftplus extends Operator {

  def ParametricSoftplus1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Softsign extends Operator {

  def Softsign1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait GivenTensorFill extends Operator {

  def GivenTensorFill1(name: String,shapeInput: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,values : Option[(Seq[Int])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ReduceLogSum extends Operator {

  def ReduceLogSum1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ArgMin extends Operator {

  def ArgMin1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : (Tensor[Long])

}
trait Tan extends Operator {

  def Tan7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait LRN extends Operator {

  def LRN1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,bias : Option[(Int)] = None,size : (String))
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Tile extends Operator {

  def Tile1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String, tiles: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], tilesname: String, axis: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], axisname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Tile6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String, repeats: Tensor[Long], repeatsname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ReduceSum extends Operator {

  def ReduceSum1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Less extends Operator {

  def Less1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Boolean])


  def Less7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : (Tensor[Boolean])

}
trait And extends Operator {

  def And1(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Boolean])


  def And7(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String)
    : (Tensor[Boolean])

}
trait Selu extends Operator {

  def Selu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None,gamma : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Selu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,gamma : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait CastMap extends Operator {

  def CastMap1(name: String,X: Map[Long, String] |: Map[Long, Float], Xname: String,cast_to : Option[(Tensor[_])] = None,map_form : Option[(Tensor[_])] = None,max_map : Option[(String)] = None)
    : (Tensor[String] |: Tensor[Float] |: Tensor[Long])

}
trait AveragePool extends Operator {

  def AveragePool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def AveragePool7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,count_include_pad : Option[(String)] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Elu extends Operator {

  def Elu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Elu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ThresholdedRelu extends Operator {

  def ThresholdedRelu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Clip extends Operator {

  def Clip1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,consumed_inputs : Option[(Seq[String])] = None,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Clip6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,max : Option[(Int)] = None,min : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait SVMClassifier extends Operator {

  def SVMClassifier1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,classlabels_ints : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[_]])] = None,coefficients : Option[(Seq[Int])] = None,kernel_params : Option[(Seq[Int])] = None,kernel_type : Option[(Tensor[_])] = None,post_transform : Option[(Tensor[_])] = None,prob_a : Option[(Seq[Int])] = None,prob_b : Option[(Seq[Int])] = None,rho : Option[(Seq[Int])] = None,support_vectors : Option[(Seq[Int])] = None,vectors_per_class : Option[(Seq[String])] = None)
    : (Tensor[String] |: Tensor[Long], Tensor[Float])

}
trait Abs extends Operator {

  def Abs1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Abs6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Sqrt extends Operator {

  def Sqrt1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Sqrt6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Softmax extends Operator {

  def Softmax1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Gather extends Operator {

  def Gather1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], dataname: String, indices: Tensor[Int] |: Tensor[Long], indicesname: String,axis : Option[(String)] = None)
    : (Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait ReduceLogSumExp extends Operator {

  def ReduceLogSumExp1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait TreeEnsembleClassifier extends Operator {

  def TreeEnsembleClassifier1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,base_values : Option[(Seq[Int])] = None,class_ids : Option[(Seq[String])] = None,class_nodeids : Option[(Seq[String])] = None,class_treeids : Option[(Seq[String])] = None,class_weights : Option[(Seq[Int])] = None,classlabels_int64s : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[_]])] = None,nodes_falsenodeids : Option[(Seq[String])] = None,nodes_featureids : Option[(Seq[String])] = None,nodes_hitrates : Option[(Seq[Int])] = None,nodes_missing_value_tracks_true : Option[(Seq[String])] = None,nodes_modes : Option[(Seq[Tensor[_]])] = None,nodes_nodeids : Option[(Seq[String])] = None,nodes_treeids : Option[(Seq[String])] = None,nodes_truenodeids : Option[(Seq[String])] = None,nodes_values : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[_])] = None)
    : (Tensor[String] |: Tensor[Long], Tensor[Float])

}
trait Sin extends Operator {

  def Sin7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Shape extends Operator {

  def Shape1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[Boolean], dataname: String)
    : (Tensor[Long])

}
trait ArrayFeatureExtractor extends Operator {

  def ArrayFeatureExtractor1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int] |: Tensor[String], Xname: String, Y: Tensor[Long], Yname: String)
    : (Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int] |: Tensor[String])

}
trait PRelu extends Operator {

  def PRelu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, slope: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], slopename: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def PRelu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, slope: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], slopename: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def PRelu7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, slope: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], slopename: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait TopK extends Operator {

  def TopK1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,axis : Option[(String)] = None,k : (String))
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Long])

}
trait ReduceSumSquare extends Operator {

  def ReduceSumSquare1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Pow extends Operator {

  def Pow1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, Y: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Yname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Pow7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, Y: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Yname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Sum extends Operator {

  def Sum1(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Sum6(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Multinomial extends Operator {

  def Multinomial7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,dtype : Option[(String)] = None,sample_size : Option[(String)] = None,seed : Option[(Int)] = None)
    : (Tensor[Int] |: Tensor[Long])

}
trait Affine extends Operator {

  def Affine1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Cast extends Operator {

  def Cast1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Boolean], inputname: String,to : (Tensor[_]))
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Boolean])


  def Cast6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Boolean], inputname: String,to : (String))
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Boolean])

}
trait Asin extends Operator {

  def Asin7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ReduceL1 extends Operator {

  def ReduceL11(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait GRUUnit extends Operator {

  def GRUUnit1(name: String,hidden_prev: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], hidden_prevname: String, gates: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], gatesname: String, seq_lengths: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], seq_lengthsname: String, t: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], tname: String,drop_states : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait GRU extends Operator {

  def GRU1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def GRU3(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def GRU7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,linear_before_reset : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Pad extends Operator {

  def Pad1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,mode : Option[(Tensor[_])] = None,paddings : (Seq[String]),value : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Pad2(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,mode : Option[(Tensor[_])] = None,pads : (Seq[String]),value : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Relu extends Operator {

  def Relu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Relu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait MeanVarianceNormalization extends Operator {

  def MeanVarianceNormalization1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,across_channels : Option[(String)] = None,normalize_variance : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait GlobalMaxPool extends Operator {

  def GlobalMaxPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait InstanceNormalization extends Operator {

  def InstanceNormalization1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,consumed_inputs : Option[(Seq[String])] = None,epsilon : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def InstanceNormalization6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,epsilon : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait LeakyRelu extends Operator {

  def LeakyRelu1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def LeakyRelu6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Squeeze extends Operator {

  def Squeeze1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], dataname: String,axes : (Seq[String]))
    : (Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait Gemm extends Operator {

  def Gemm1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, C: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Gemm6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, C: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,broadcast : Option[(String)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Gemm7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, C: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Cname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,transA : Option[(String)] = None,transB : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Mul extends Operator {

  def Mul1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Mul6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Mul7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Floor extends Operator {

  def Floor1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Floor6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait HardSigmoid extends Operator {

  def HardSigmoid1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def HardSigmoid6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Reciprocal extends Operator {

  def Reciprocal1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Reciprocal6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait RNN extends Operator {

  def RNN1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def RNN7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait If extends Operator {

  def If1(name: String,cond: Tensor[Boolean], condname: String,else_branch : (Seq[Float]),then_branch : (Seq[Float]))
    : (Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait ConstantFill extends Operator {

  def ConstantFill1(name: String,input: Option[Tensor[Float] |: Tensor[Int] |: Tensor[Long] |: Tensor[Boolean]] = None,dtype : Option[(String)] = None,extra_shape : Option[(Seq[String])] = None,input_as_shape : Option[(String)] = None,shape : Option[(Seq[String])] = None,value : Option[(Int)] = None)
    : (Tensor[Float] |: Tensor[Int] |: Tensor[Long] |: Tensor[Boolean])

}
trait Scaler extends Operator {

  def Scaler1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,offset : Option[(Seq[Int])] = None,scaleAttr : Option[(Seq[Int])] = None)
    : (Tensor[Float])

}
trait Split extends Operator {

  def Split1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,split: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Split2(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None,splitAttr : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Constant extends Operator {

  def Constant1(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ImageScaler extends Operator {

  def ImageScaler1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,bias : Option[(Seq[Int])] = None,scaleAttr : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Normalizer extends Operator {

  def Normalizer1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,norm : Option[(Tensor[_])] = None)
    : (Tensor[Float])

}
trait DictVectorizer extends Operator {

  def DictVectorizer1(name: String,X: Map[String, Long] |: Map[Long, String] |: Map[Long, Float] |: Map[Long, Double] |: Map[String, Float] |: Map[String, Double], Xname: String,int64_vocabulary : Option[(Seq[String])] = None,string_vocabulary : Option[(Seq[Tensor[_]])] = None)
    : (Tensor[Long] |: Tensor[Float] |: Tensor[Double] |: Tensor[String])

}
trait BatchNormalization extends Operator {

  def BatchNormalization1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, mean: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], meanname: String, someVar: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], varname: String,consumed_inputs : (Seq[String]),epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def BatchNormalization6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, mean: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], meanname: String, someVar: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], varname: String,epsilon : Option[(Int)] = None,is_test : Option[(String)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def BatchNormalization7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, scale: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], scalename: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String, mean: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], meanname: String, someVar: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], varname: String,epsilon : Option[(Int)] = None,momentum : Option[(Int)] = None,spatial : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait OneHotEncoder extends Operator {

  def OneHotEncoder1(name: String,X: Tensor[String] |: Tensor[Long] |: Tensor[Int] |: Tensor[Float] |: Tensor[Double], Xname: String,cats_int64s : Option[(Seq[String])] = None,cats_strings : Option[(Seq[Tensor[_]])] = None,zeros : Option[(String)] = None)
    : (Tensor[Float])

}
trait ArgMax extends Operator {

  def ArgMax1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axis : Option[(String)] = None,keepdims : Option[(String)] = None)
    : (Tensor[Long])

}
trait MatMul extends Operator {

  def MatMul1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Neg extends Operator {

  def Neg1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Float] |: Tensor[Int] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Float] |: Tensor[Int] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Double])


  def Neg6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Float] |: Tensor[Int] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Float] |: Tensor[Int] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Double])

}
trait Xor extends Operator {

  def Xor1(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Boolean])


  def Xor7(name: String,A: Tensor[Boolean], Aname: String, B: Tensor[Boolean], Bname: String)
    : (Tensor[Boolean])

}
trait Acos extends Operator {

  def Acos7(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Softplus extends Operator {

  def Softplus1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Slice extends Operator {

  def Slice1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,ends : (Seq[String]),starts : (Seq[String]))
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait LogSoftmax extends Operator {

  def LogSoftmax1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Not extends Operator {

  def Not1(name: String,X: Tensor[Boolean], Xname: String)
    : (Tensor[Boolean])

}
trait LSTM extends Operator {

  def LSTM1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, initial_c: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, P: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None,output_sequence : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def LSTM7(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String, R: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Rname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, sequence_lens: Option[Tensor[Int]] = None, initial_h: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, initial_c: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None, P: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,activation_alpha : Option[(Seq[Int])] = None,activation_beta : Option[(Seq[Int])] = None,activations : Option[(Seq[Tensor[_]])] = None,clip : Option[(Int)] = None,direction : Option[(Tensor[_])] = None,hidden_size : Option[(String)] = None,input_forget : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait RandomNormal extends Operator {

  def RandomNormal1(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Mean extends Operator {

  def Mean1(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Mean6(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait RandomNormalLike extends Operator {

  def RandomNormalLike1(name: String,input: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], inputname: String,dtype : Option[(String)] = None,mean : Option[(Int)] = None,scaleAttr : Option[(Int)] = None,seed : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait GlobalAveragePool extends Operator {

  def GlobalAveragePool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Upsample extends Operator {

  def Upsample1(name: String,X: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], Xname: String,height_scaleAttr : (Int),mode : Option[(Tensor[_])] = None,width_scaleAttr : (Int))
    : (Tensor[Boolean] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])


  def Upsample7(name: String,X: Tensor[Boolean] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], Xname: String,mode : Option[(Tensor[_])] = None,scaleAttrs : (Seq[Int]))
    : (Tensor[Boolean] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait Unsqueeze extends Operator {

  def Unsqueeze1(name: String,data: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], dataname: String,axes : (Seq[String]))
    : (Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait Add extends Operator {

  def Add1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Add6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Add7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Conv extends Operator {

  def Conv1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,auto_pad : Option[(Tensor[_])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Concat extends Operator {

  def Concat1(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])


  def Concat4(name: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait MaxPool extends Operator {

  def MaxPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait TreeEnsembleRegressor extends Operator {

  def TreeEnsembleRegressor1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,aggregate_function : Option[(Tensor[_])] = None,base_values : Option[(Seq[Int])] = None,n_targets : Option[(String)] = None,nodes_falsenodeids : Option[(Seq[String])] = None,nodes_featureids : Option[(Seq[String])] = None,nodes_hitrates : Option[(Seq[Int])] = None,nodes_missing_value_tracks_true : Option[(Seq[String])] = None,nodes_modes : Option[(Seq[Tensor[_]])] = None,nodes_nodeids : Option[(Seq[String])] = None,nodes_treeids : Option[(Seq[String])] = None,nodes_truenodeids : Option[(Seq[String])] = None,nodes_values : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[_])] = None,target_ids : Option[(Seq[String])] = None,target_nodeids : Option[(Seq[String])] = None,target_treeids : Option[(Seq[String])] = None,target_weights : Option[(Seq[Int])] = None)
    : (Tensor[Float])

}
trait Transpose extends Operator {

  def Transpose1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,perm : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Hardmax extends Operator {

  def Hardmax1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait LoopIndexTensor extends Operator {

  def LoopIndexTensor1(name: String,T: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], Tname: String, loop_idx: Int, loop_idxname: String,axis : Option[(String)] = None)
    : (Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait Exp extends Operator {

  def Exp1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Exp6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait CategoryMapper extends Operator {

  def CategoryMapper1(name: String,X: Tensor[String] |: Tensor[Long], Xname: String,cats_int64s : Option[(Seq[String])] = None,cats_strings : Option[(Seq[Tensor[_]])] = None,default_int64 : Option[(String)] = None,default_string : Option[(Tensor[_])] = None)
    : (Tensor[String] |: Tensor[Long])

}
trait Sub extends Operator {

  def Sub1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Sub6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Sub7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Imputer extends Operator {

  def Imputer1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,imputed_value_floats : Option[(Seq[Int])] = None,imputed_value_int64s : Option[(Seq[String])] = None,replaced_value_float : Option[(Int)] = None,replaced_value_int64 : Option[(String)] = None)
    : (Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int])

}
trait ScaledTanh extends Operator {

  def ScaledTanh1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,alpha : Option[(Int)] = None,beta : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ReduceProd extends Operator {

  def ReduceProd1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ReduceMin extends Operator {

  def ReduceMin1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Scale extends Operator {

  def Scale1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,scaleAttr : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Div extends Operator {

  def Div1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Div6(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Div7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Greater extends Operator {

  def Greater1(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String,axis : Option[(String)] = None,broadcast : Option[(String)] = None)
    : (Tensor[Boolean])


  def Greater7(name: String,A: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Aname: String, B: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Bname: String)
    : (Tensor[Boolean])

}
trait Sigmoid extends Operator {

  def Sigmoid1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Sigmoid6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait LinearRegressor extends Operator {

  def LinearRegressor1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,coefficients : Option[(Seq[Int])] = None,intercepts : Option[(Seq[Int])] = None,post_transform : Option[(Tensor[_])] = None,targets : Option[(String)] = None)
    : (Tensor[Float])

}
trait Tanh extends Operator {

  def Tanh1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Tanh6(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Flatten extends Operator {

  def Flatten1(name: String,input: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], inputname: String,axis : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ConvTranspose extends Operator {

  def ConvTranspose1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, W: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Wname: String,B: Option[Tensor[Float16] |: Tensor[Float] |: Tensor[Double]] = None,auto_pad : Option[(Tensor[_])] = None,dilations : Option[(Seq[String])] = None,group : Option[(String)] = None,kernel_shape : Option[(Seq[String])] = None,output_padding : Option[(Seq[String])] = None,output_shape : Option[(Seq[String])] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Identity extends Operator {

  def Identity1(name: String,input: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], inputname: String)
    : (Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait Size extends Operator {

  def Size1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[UByte] |: Tensor[UShort] |: Tensor[Boolean], dataname: String)
    : (Tensor[Long])

}
trait FeatureVectorizer extends Operator {

  def FeatureVectorizer1(name: String)
    : (Tensor[Float])

}
trait ReduceMax extends Operator {

  def ReduceMax1(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait LinearClassifier extends Operator {

  def LinearClassifier1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,classlabels_ints : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[_]])] = None,coefficients : (Seq[Int]),intercepts : Option[(Seq[Int])] = None,multi_class : Option[(String)] = None,post_transform : Option[(Tensor[_])] = None)
    : (Tensor[String] |: Tensor[Long], Tensor[Float])

}
trait Binarizer extends Operator {

  def Binarizer1(name: String,X: Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int], Xname: String,threshold : Option[(Int)] = None)
    : (Tensor[Float] |: Tensor[Double] |: Tensor[Long] |: Tensor[Int])

}
trait LpPool extends Operator {

  def LpPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : Option[(Seq[String])] = None,p : Option[(Int)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def LpPool2(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,auto_pad : Option[(Tensor[_])] = None,kernel_shape : (Seq[String]),p : Option[(String)] = None,pads : Option[(Seq[String])] = None,strides : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait MaxRoiPool extends Operator {

  def MaxRoiPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String, rois: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], roisname: String,pooled_shape : (Seq[String]),spatial_scaleAttr : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Ceil extends Operator {

  def Ceil1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,consumed_inputs : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Ceil6(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait GlobalLpPool extends Operator {

  def GlobalLpPool1(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,p : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def GlobalLpPool2(name: String,X: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], Xname: String,p : Option[(String)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait ReduceL2 extends Operator {

  def ReduceL21(name: String,data: Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,axes : Option[(Seq[String])] = None,keepdims : Option[(String)] = None)
    : (Tensor[UInt] |: Tensor[ULong] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Reshape extends Operator {

  def Reshape1(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String,consumed_inputs : Option[(Seq[String])] = None,shape : Option[(Seq[String])] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])


  def Reshape5(name: String,data: Tensor[Float16] |: Tensor[Float] |: Tensor[Double], dataname: String, shape: Tensor[Long], shapename: String)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait RandomUniformLike extends Operator {

  def RandomUniformLike1(name: String,input: Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean], inputname: String,dtype : Option[(String)] = None,high : Option[(Int)] = None,low : Option[(Int)] = None,seed : Option[(Int)] = None)
    : (Tensor[Float16] |: Tensor[Float] |: Tensor[Double])

}
trait Loop extends Operator {

  def Loop1(name: String,M: Long, Mname: String, cond: Boolean, condname: String,body : (Seq[Float]))
    : (Tensor[UByte] |: Tensor[UShort] |: Tensor[UInt] |: Tensor[ULong] |: Tensor[Byte] |: Tensor[Short] |: Tensor[Int] |: Tensor[Long] |: Tensor[Float16] |: Tensor[Float] |: Tensor[Double] |: Tensor[String] |: Tensor[Boolean])

}
trait LabelEncoder extends Operator {

  def LabelEncoder1(name: String,X: Tensor[String] |: Tensor[Long], Xname: String,classes_strings : Option[(Seq[Tensor[_]])] = None,default_int64 : Option[(String)] = None,default_string : Option[(Tensor[_])] = None)
    : (Tensor[String] |: Tensor[Long])

}
trait ZipMap extends Operator {

  def ZipMap1(name: String,X: Tensor[Float], Xname: String,classlabels_int64s : Option[(Seq[String])] = None,classlabels_strings : Option[(Seq[Tensor[_]])] = None)
    : (Seq[Map[String, Float]] |: Seq[Map[Long, Float]])

}}
