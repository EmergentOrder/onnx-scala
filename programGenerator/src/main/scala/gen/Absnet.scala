package org.emergentorder.onnx

import org.emergentorder.onnx.backends._
import org.emergentorder.union._
import org.emergentorder.onnx.Tensors._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.math.Numeric
import io.kjaer.compiletime._
import org.emergentorder.compiletime._

class Absnet() {
  val Abs: AbsV6 = new ORTOperatorBackendAll()
  def program(inputDatax: Tensor[Float, ("test", "test" ##: TSNil, 1 #: 5 #: SNil)]): List[Tensor[Float, ("test", "test" ##: TSNil, 1 #: 5 #: SNil)]] =
    for {
      nodex <- List(inputDatax)
      nodey <- List(Abs.AbsV6[Float, "test", "test" ##: TSNil, 1 #: 5 #: SNil]("y", X = nodex))
    } yield (nodey)
}
