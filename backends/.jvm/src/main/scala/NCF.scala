package org.emergentorder.onnx

import cats.effect.IO
import org.emergentorder.compiletime._
import org.emergentorder.io.kjaer.compiletime._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends._

import scala.language.higherKinds

//TODO: Add changes to generator; Generate both full model and layerwise programs each time
class NCF(byteArray: Array[Byte], userIdsMap: Map[Long, Long], itemIdsMap: Map[Long, Long])
    extends AutoCloseable {

   val fullORTBackend = new ORTModelBackend(byteArray)

   def fullNCF(
       inputDataactual_input_1: Tensor[Long, Axes],
       inputDatalearned_0: Tensor[Long, Axes]
   ): Tensor[Float, Axes] = {
//    val scope = new PointerScope()
      def dataToUserIds(in: IO[Array[Long]]) = in.map(x => x.map(y => userIdsMap(y)))

      def dataToItemIds(in: IO[Array[Long]]) = in.map(x => x.map(y => itemIdsMap(y)))

      val nodeactual_input_1 = Tuple1(
        IO(dataToUserIds(inputDataactual_input_1.data), inputDataactual_input_1.shape)
      )

      
      Tuple1(
        IO(dataToItemIds(inputDatalearned_0.data), inputDatalearned_0.shape)
      )

      // Note: Don't need to specify all the type params except in Dotty
      val nodeFullOutput: Tensor[Float, Axes] =
         fullORTBackend
            .fullModel[Float, "TensorType", "DimensionDenotation" ##: TSNil, 1 #: 1000 #: SNil](
              // TODO: testing less than enough inputs
              (nodeactual_input_1)
            )

      nodeFullOutput // .asInstanceOf[Tensor[Float]] //Bad
   }

   override def close(): Unit = {
      fullORTBackend.close()

   }
}
