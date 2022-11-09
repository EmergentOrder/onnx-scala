package org.emergentorder.onnx.backends

import sys.process._
import java.net.URL
import java.io.File
import java.nio.file.{Files, Paths}
import scala.language.postfixOps
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.backends._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should._
import cats.effect.testing.scalatest.AsyncIOSpec

class ONNXScalaSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

   new URL(
     "https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-12.onnx"
   ) #> new File("squeezenet1.0-12.onnx") !!

   "SqueezeNet ONNX-Scala model should predict dummy image class" in {
      val squeezenetBytes = Files.readAllBytes(Paths.get("squeezenet1.0-12.onnx"))
      val squeezenet      = new ORTModelBackend(squeezenetBytes)
      val data            = Array.fill(1 * 3 * 224 * 224) { 42f }
      // In NCHW tensor image format
      val shape                 = 1 #: 3 #: 224 #: 224 #: SNil
      val tensorShapeDenotation = "Batch" ##: "Channel" ##: "Height" ##: "Width" ##: TSNil

      val tensorDenotation: String & Singleton = "Image"

      val imageTens = Tensor(data, tensorDenotation, tensorShapeDenotation, shape)

      // or as a shorthand if you aren't concerned with enforcing denotations
      val imageTensDefaultDenotations = Tensor(data, shape)
      val out = squeezenet.fullModel[
        Float,
        "ImageNetClassification",
        "Batch" ##: "Class" ##: TSNil,
        1 #: 1000 #: 1 #: 1 #: SNil
      ](Tuple(imageTens))

      // The output shape
      out.shape.asserting(_(0) shouldBe 1)
      out.shape.asserting(_(1) shouldBe 1000)

      // The highest probability (predicted) class
      out.data.asserting(x => x.indices.maxBy(x) shouldBe 549)
   }
}
