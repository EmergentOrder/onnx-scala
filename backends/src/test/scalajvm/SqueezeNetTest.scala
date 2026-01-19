package org.emergentorder.onnx.backends

import cats.effect.testing.scalatest.AsyncIOSpec
import org.emergentorder.compiletime._
import org.emergentorder.io.kjaer.compiletime._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends._
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should._

import java.io.File
import java.net.URI
import java.nio.file.Files
import java.nio.file.Paths
import scala.language.postfixOps

import sys.process._

class ONNXScalaSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

   if(! Files.exists(Paths.get("squeezenet1_1_Opset18.onnx"))){
   new URI(
     "https://media.githubusercontent.com/media/onnx/models/main/Computer_Vision/squeezenet1_1_Opset18_torch_hub/squeezenet1_1_Opset18.onnx"
   ).toURL #> new File("squeezenet1_1_Opset18.onnx") !!
   }
   "SqueezeNet ONNX-Scala model should predict dummy image class" in {
      val squeezenetBytes =
         Files.readAllBytes(Paths.get("squeezenet1_1_Opset18.onnx")) // .quant.onnx"))
      val squeezenet = new ORTModelBackend(squeezenetBytes)
      val data       = Array.fill(1 * 3 * 224 * 224) { 42f }
      // In NCHW tensor image format
      val shape                 = 1 #: 3 #: 224 #: 224 #: SNil
      val tensorShapeDenotation = "Batch" ##: "Channel" ##: "Height" ##: "Width" ##: TSNil

      val tensorDenotation: String & Singleton = "Image"

      val imageTens = Tensor(data, tensorDenotation, tensorShapeDenotation, shape)

      // or as a shorthand if you aren't concerned with enforcing denotations
      Tensor(data, shape)
      val out = squeezenet.fullModel[
        Float,
        "ImageNetClassification",
        "Batch" ##: "Class" ##: TSNil,
        1 #: 1000 #: SNil
      ](Tuple(imageTens))

      // The output shape
      // and the highest probability (predicted) class
      val singleIO = cats.effect.IO.both(out.shape, out.data)
      singleIO.asserting(x =>
         ((x._1(0), x._1(1), x._2.indices.maxBy(x._2))
            shouldBe
               (1, 1000, 753))
      )

   }
}
