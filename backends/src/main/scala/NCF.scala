package org.emergentorder.onnx

import org.emergentorder.onnx._
import org.emergentorder.onnx.backends._
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

//TODO: Add changes to generator; Generate both full model and layerwise programs each time
class NCF(byteArray: Array[Byte], userIdsMap: Map[Long, Long], itemIdsMap: Map[Long, Long])
    extends AutoCloseable {

  val scope             = new PointerScope()
  val fullNgraphHandler = new NGraphBackend(byteArray)

  def fullNCF(
      inputDataactual_input_1: List[Tensor[Long]],
      inputDatalearned_0: List[Tensor[Long]]
  ): List[Tensor[Float]] = {
//    val scope = new PointerScope()
    val result = for {
      nodeactual_input_1 <- inputDataactual_input_1.map(
        x => TensorFactory.getTensor(x._1.map(y => userIdsMap(y)), x._2)
      )
      nodelearned_0 <- inputDatalearned_0.map(
        x => TensorFactory.getTensor(x._1.map(y => itemIdsMap(y)), x._2)
      )
      nodeFullOutput <- List(
        fullNgraphHandler
          .fullModel[Long, Long, Long, Float](Some(nodeactual_input_1), Some(nodelearned_0), None)
      )
    } yield (nodeFullOutput)
//    scope.close
    System.runFinalization
    result
  }

  override def close(): Unit = {
    fullNgraphHandler.close
    scope.close
  }
}
