package org.emergentorder.onnx

import org.emergentorder.onnx.UnionType._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import scala.language.higherKinds

trait NCF {
  val Gather: Gather
  val Mul: Mul
  val Gemm: Gemm
  val Sigmoid: Sigmoid
  val dataSource: DataSource
  def program[T : (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check:Numeric:ClassTag]: List[Tensor[Float]]  = 
    for {
      nodeactual_input_1 <- List(dataSource.inputData[Long])
      nodelearned_0 <- List(dataSource.inputData[Long])
      nodelearned_1 <- List( dataSource.getParams[Float]("(learned_1,Float)"))
      nodelearned_2 <- List( dataSource.getParams[Float]("(learned_2,Float)"))
      nodelearned_3 <- List( dataSource.getParams[Float]("(learned_3,Float)"))
      nodelearned_4 <- List( dataSource.getParams[Float]("(learned_4,Float)"))
      node6 <- List(Gather.Gather1("6" ,data = Some(nodelearned_1),indices = Some(nodeactual_input_1)))
      node7 <- List(Gather.Gather1("7" ,data = Some(nodelearned_2),indices = Some(nodelearned_0)))
      node8 <- List(Mul.Mul7("8" ,A = Some(node6),B = Some(node7)))
      node9 <- List(Gemm.Gemm9("9" ,transB = Some((1)),A = Some(node8),B = Some(nodelearned_3),C = Some(nodelearned_4)))
      nodeoutput1 <- List(Sigmoid.Sigmoid6[Float]("output1" ,X = Some(node9)))
    } yield (nodeoutput1)
}
