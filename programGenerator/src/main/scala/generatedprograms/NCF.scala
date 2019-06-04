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
  val Concat: Concat
  val Mul: Mul
  val Gemm: Gemm
  val Relu: Relu
  val Sigmoid: Sigmoid
  val dataSource: DataSource
  def program: List[Tensor[Float]]  = 
    for {
      nodeactual_input_1 <- List(dataSource.inputData[Long])
      nodelearned_0 <- List(dataSource.inputData[Long])
      nodeaffine_outputbias <- List( dataSource.getParams[Float]("affine_output.bias"))
      nodeaffine_outputweight <- List( dataSource.getParams[Float]("affine_output.weight"))
      nodefc_layers0bias <- List( dataSource.getParams[Float]("fc_layers.0.bias"))
      nodefc_layers0weight <- List( dataSource.getParams[Float]("fc_layers.0.weight"))
      nodefc_layers1bias <- List( dataSource.getParams[Float]("fc_layers.1.bias"))
      nodefc_layers1weight <- List( dataSource.getParams[Float]("fc_layers.1.weight"))
      nodefc_layers2bias <- List( dataSource.getParams[Float]("fc_layers.2.bias"))
      nodefc_layers2weight <- List( dataSource.getParams[Float]("fc_layers.2.weight"))
      nodelearned_1 <- List( dataSource.getParams[Float]("learned_1"))
      nodelearned_2 <- List( dataSource.getParams[Float]("learned_2"))
      nodelearned_3 <- List( dataSource.getParams[Float]("learned_3"))
      nodelearned_4 <- List( dataSource.getParams[Float]("learned_4"))
      node14 <- List(Gather.Gather1("14" ,data = Some(nodelearned_1),indices = Some(nodeactual_input_1)))
      node15 <- List(Gather.Gather1("15" ,data = Some(nodelearned_2),indices = Some(nodelearned_0)))
      node16 <- List(Gather.Gather1("16" ,data = Some(nodelearned_3),indices = Some(nodeactual_input_1)))
      node17 <- List(Gather.Gather1("17" ,data = Some(nodelearned_4),indices = Some(nodelearned_0)))
      node18 <- List(Concat.Concat4("18" ,axis = Some((-1)),inputs = Seq(Some(node14),Some(node15))))
      node19 <- List(Mul.Mul7("19" ,A = Some(node16),B = Some(node17)))
      node20 <- List(Gemm.Gemm9("20" ,transB = Some((1)),A = Some(node18),B = Some(nodefc_layers0weight),C = Some(nodefc_layers0bias)))
      node21 <- List(Relu.Relu6("21" ,X = Some(node20)))
      node22 <- List(Gemm.Gemm9("22" ,transB = Some((1)),A = Some(node21),B = Some(nodefc_layers1weight),C = Some(nodefc_layers1bias)))
      node23 <- List(Relu.Relu6("23" ,X = Some(node22)))
      node24 <- List(Gemm.Gemm9("24" ,transB = Some((1)),A = Some(node23),B = Some(nodefc_layers2weight),C = Some(nodefc_layers2bias)))
      node25 <- List(Relu.Relu6("25" ,X = Some(node24)))
      node26 <- List(Concat.Concat4("26" ,axis = Some((-1)),inputs = Seq(Some(node25),Some(node19))))
      node27 <- List(Gemm.Gemm9("27" ,transB = Some((1)),A = Some(node26),B = Some(nodeaffine_outputweight),C = Some(nodeaffine_outputbias)))
      nodeoutput1 <- List(Sigmoid.Sigmoid6[Float]("output1" ,X = Some(node27)))
    } yield (nodeoutput1)
}
