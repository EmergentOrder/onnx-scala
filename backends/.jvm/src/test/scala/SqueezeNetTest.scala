package org.emergentorder.onnx.backends

import sys.process._
import java.net.URL
import java.io.File
import java.nio.file.{Files, Paths}
import scala.language.postfixOps
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should._

class ONNXScalaSpec extends AnyFlatSpec with Matchers {

   new URL(
     "https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-12.onnx"
   ) #> new File("squeezenet1.0-12.onnx") !!

   "SqueezeNet ONNX-Scala model" should "predict dummy image class" in {
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
      val out = squeezenet.fullModelResult[
        Float,
        "ImageNetClassification",
        "Batch" ##: "Class" ##: TSNil,
        1 #: 1000 #: 1 #: 1 #: SNil
      ](Tuple(imageTens))

      // The output shape
      assert(out.shape(0) == 1)
      assert(out.shape(1) == 1000)

      // The highest probability (predicted) class
      assert(out.data.indices.maxBy(out.data) == 549)
   }
}
