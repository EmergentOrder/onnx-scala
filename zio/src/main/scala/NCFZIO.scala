package org.emergentorder.onnxZIO

import zio.Task
import org.emergentorder.onnx._
import org.emergentorder.union.UnionType._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import scala.language.higherKinds
import scala.io.Source
import scala.reflect.io.Streamable

//TODO: Add changes to generator; Generate both full model and layerwise programs each time
class NCFZIO(byteArray: Array[Byte], userIdsMap: Map[Long, Long], itemIdsMap: Map[Long, Long]) {

  val onnxHelper                = new ONNXHelper(byteArray)
  val GatherZIO: GatherZIO      = new ONNXNGraphHandlers(onnxHelper)
  val ConcatZIO: ConcatZIO      = new ONNXNGraphHandlers(onnxHelper)
  val MulZIO: MulZIO            = new ONNXNGraphHandlers(onnxHelper)
  val GemmZIO: GemmZIO          = new ONNXNGraphHandlers(onnxHelper)
  val ReluZIO: ReluZIO          = new ONNXNGraphHandlers(onnxHelper)
  val SigmoidZIO: SigmoidZIO    = new ONNXNGraphHandlers(onnxHelper)
  val dataSource: DataSourceZIO = new ONNXNGraphHandlers(onnxHelper)
  val fullNgraphHandler         = new ONNXNGraphHandlers(onnxHelper)
  def fullNCF(
      inputDataactual_input_1: Task[Tensor[Long]],
      inputDatalearned_0: Task[Tensor[Long]]
  ): Task[Tensor[Float]] =
    for {
      nodeactual_input_1 <- inputDataactual_input_1.map(
        x => TensorFactory.getTensor(x._1.map(y => userIdsMap(y)), x._2)
      )
      nodelearned_0 <- inputDatalearned_0.map(
        x => TensorFactory.getTensor(x._1.map(y => itemIdsMap(y)), x._2)
      )
      nodeFullOutput <- Task {
        (fullNgraphHandler
          .fullModel[Long, Long, Long, Float](Some(nodeactual_input_1), Some(nodelearned_0), None))
      }
    } yield (nodeFullOutput)

  def fineNCF(
      inputDataactual_input_1: Task[Tensor[Long]],
      inputDatalearned_0: Task[Tensor[Long]]
  ): Task[Tensor[Float]] =
    for {
      nodeactual_input_1 <- inputDataactual_input_1.map(
        x => TensorFactory.getTensor(x._1.map(y => userIdsMap(y)), x._2)
      )
      nodelearned_0 <- inputDatalearned_0.map(
        x => TensorFactory.getTensor(x._1.map(y => itemIdsMap(y)), x._2)
      )
      nodeaffine_outputbias <- dataSource.getParamsZIO[Float](
        "affine_output.bias"
      )
      nodeaffine_outputweight <- dataSource.getParamsZIO[Float](
        "affine_output.weight"
      )
      nodefc_layers0bias <- dataSource.getParamsZIO[Float]("fc_layers.0.bias")
      nodefc_layers0weight <- dataSource.getParamsZIO[Float](
        "fc_layers.0.weight"
      )
      nodefc_layers1bias <- dataSource.getParamsZIO[Float]("fc_layers.1.bias")
      nodefc_layers1weight <- dataSource.getParamsZIO[Float](
        "fc_layers.1.weight"
      )
      nodefc_layers2bias <- dataSource.getParamsZIO[Float]("fc_layers.2.bias")
      nodefc_layers2weight <- dataSource.getParamsZIO[Float](
        "fc_layers.2.weight"
      )
      nodelearned_1 <- dataSource.getParamsZIO[Float]("learned_1")
      nodelearned_2 <- dataSource.getParamsZIO[Float]("learned_2")
      nodelearned_3 <- dataSource.getParamsZIO[Float]("learned_3")
      nodelearned_4 <- dataSource.getParamsZIO[Float]("learned_4")
      node14 <- GatherZIO.Gather1ZIO(
        "14",
        data = Some(nodelearned_1),
        indices = Some(nodeactual_input_1)
      )
      node15 <- GatherZIO.Gather1ZIO(
        "15",
        data = Some(nodelearned_2),
        indices = Some(nodelearned_0)
      )
      node16 <- GatherZIO.Gather1ZIO(
        "16",
        data = Some(nodelearned_3),
        indices = Some(nodeactual_input_1)
      )
      node17 <- GatherZIO.Gather1ZIO(
        "17",
        data = Some(nodelearned_4),
        indices = Some(nodelearned_0)
      )
      node18 <- ConcatZIO.Concat4ZIO(
        "18",
        axis = Some((-1)),
        inputs = Seq(Some(node14), Some(node15))
      )
      node19 <- MulZIO.Mul7ZIO("19", A = Some(node16), B = Some(node17))
      node20 <- GemmZIO.Gemm9ZIO(
        "20",
        transB = Some((1)),
        A = Some(node18),
        B = Some(nodefc_layers0weight),
        C = Some(nodefc_layers0bias)
      )
      node21 <- ReluZIO.Relu6ZIO("21", X = Some(node20))
      node22 <- GemmZIO.Gemm9ZIO(
        "22",
        transB = Some((1)),
        A = Some(node21),
        B = Some(nodefc_layers1weight),
        C = Some(nodefc_layers1bias)
      )
      node23 <- ReluZIO.Relu6ZIO("23", X = Some(node22))
      node24 <- GemmZIO.Gemm9ZIO(
        "24",
        transB = Some((1)),
        A = Some(node23),
        B = Some(nodefc_layers2weight),
        C = Some(nodefc_layers2bias)
      )
      node25 <- ReluZIO.Relu6ZIO("25", X = Some(node24))
      node26 <- ConcatZIO.Concat4ZIO(
        "26",
        axis = Some((-1)),
        inputs = Seq(Some(node25), Some(node19))
      )
      node27 <- GemmZIO.Gemm9ZIO(
        "27",
        transB = Some((1)),
        A = Some(node26),
        B = Some(nodeaffine_outputweight),
        C = Some(nodeaffine_outputbias)
      )
      nodeoutput1 <- SigmoidZIO.Sigmoid6ZIO[Float]("output1", X = Some(node27))
    } yield (nodeoutput1)
}
