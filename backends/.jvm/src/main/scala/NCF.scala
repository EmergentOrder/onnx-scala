package org.emergentorder.onnx

import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
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

//TODO: Add changes to generator; Generate both full model and layerwise programs each time
class NCF(byteArray: Array[Byte], userIdsMap: Map[Long, Long], itemIdsMap: Map[Long, Long])
    extends AutoCloseable {


  val fullORTBackend = new ORTModelBackend(byteArray)

  def fullNCF(
      inputDataactual_input_1: Tensor[Long, ?],
      inputDatalearned_0: Tensor[Long, ?]
  ): Tensor[Float, ?] = {
//    val scope = new PointerScope()
    val nodeactual_input_1 = Tuple1((
      inputDataactual_input_1._1.map(y => userIdsMap(y)),
      inputDataactual_input_1._2)
    )

    val nodelearned_0 = Tuple1((inputDatalearned_0._1.map(y => itemIdsMap(y)), inputDatalearned_0._2))

    //Note: Don't need to specify all the type params except in Dotty
    val nodeFullOutput: Tensor[Float, ?] =
      fullORTBackend
        .fullModel[Float, Axes](
          //TODO: testing less than enough inputs
          (nodeactual_input_1)
        )



    nodeFullOutput //.asInstanceOf[Tensor[Float]] //Bad
  }

  override def close(): Unit = {
    fullORTBackend.close

  }
}
