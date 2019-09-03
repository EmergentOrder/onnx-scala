package org.emergentorder.onnxZIO

import zio.Task
import org.emergentorder.onnx._
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
import org.bytedeco.javacpp.PointerScope
import org.bytedeco.javacpp.Pointer

//TODO: Add changes to generator; Generate both full model and layerwise programs each time
class NCFZIO(byteArray: Array[Byte], userIdsMap: Map[Long, Long], itemIdsMap: Map[Long, Long])
    extends AutoCloseable {

  val scope             = new PointerScope()
  val fullNgraphHandler = new ONNXNGraphHandlers(byteArray)

  def fullNCF(
      inputDataactual_input_1: Task[Tensor[Long]],
      inputDatalearned_0: Task[Tensor[Long]]
  ): Task[Tensor[Float]] = {
    //println(Pointer.totalBytes)
    val scope = new PointerScope()
    val result = for {
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
    scope.close
    result
  }

  override def close(): Unit = {
    fullNgraphHandler.close
    scope.close
  }
}
