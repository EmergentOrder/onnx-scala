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
import org.bytedeco.javacpp.PointerScope

//TODO: Add changes to generator; Generate both full model and layerwise programs each time
class NCF(byteArray: Array[Byte], userIdsMap: Map[Long, Long], itemIdsMap: Map[Long, Long])
    extends AutoCloseable {

  val scope             = new PointerScope()
  val fullNgraphHandler = new ORTModelBackend(byteArray)

  def fullNCF(
      inputDataactual_input_1: Tensor[Long],
      inputDatalearned_0: Tensor[Long]
  ): Tensor[Float] = {
//    val scope = new PointerScope()
    val nodeactual_input_1 = TensorFactory.getTensor(
      inputDataactual_input_1._1.map(y => userIdsMap(y)),
      inputDataactual_input_1._2
    )

    val nodelearned_0 =
      TensorFactory.getTensor(inputDatalearned_0._1.map(y => itemIdsMap(y)), inputDatalearned_0._2)

    //Note: Don't need to specify all the type params except in Dotty
    val nodeFullOutput: Tuple1[Tensor[Float]] =
      fullNgraphHandler
        .fullModel[Tensor[Float]](
          //TODO: testing less than enough inputs
          Some((nodeactual_input_1))
        )

    //    scope.close
    System.runFinalization
    nodeFullOutput(0) //.asInstanceOf[Tensor[Float]] //Bad
  }

  override def close(): Unit = {
    fullNgraphHandler.close
    scope.close
  }
}
