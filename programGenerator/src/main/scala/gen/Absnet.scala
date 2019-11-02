package org.emergentorder.onnx

import org.emergentorder.onnx.backends._
import org.emergentorder.union._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.math.Numeric

class Absnet(byteArray: Array[Byte]) {
  val Abs: Abs = new NGraphBackendFullAtoL(byteArray)
  def program(inputDatax: Tensor[Float]): List[Tensor[Float]]  = 
    for {
      nodex <- List(inputDatax)
      nodey <- List(Abs.Abs6("y" ,X = Some(nodex)))
    } yield (nodey)
}
