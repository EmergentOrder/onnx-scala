package org.emergentorder.onnx.backends

import sys.process._
import scala.language.postfixOps
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.backends._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._
import org.emergentorder.onnx.onnxruntimeWeb.mod.{InferenceSession => OrtSession}
import cats.effect.IO

import org.scalatest._

import org.scalatest.flatspec.AsyncFlatSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should._
import cats.effect.testing.scalatest.AsyncIOSpec
import org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession

class ONNXScalaSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

   implicit override def executionContext =
      scala.scalajs.concurrent.JSExecutionContext.Implicits.queue
   // TODO: push this inside ORTWebModelBackend, and use other create() which takes arraybufferlike
   val session: IO[
     org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
   ] = IO.fromFuture(IO { OrtSession.create("squeezenet1.0-12.onnx",
      {
        val opts = InferenceSession.SessionOptions()
        opts.executionProviders = scala.scalajs.js.Array("wasm")
        opts
      }
      ).toFuture })

   "SqueezeNet ONNX-Scala model should predict dummy image class" in {
      val squeezenet = new ORTWebModelBackend(session)
      val data       = Array.fill(1 * 3 * 224 * 224) { 42f }
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
      // and the highest probability (predicted) class
      val singleIO = cats.effect.IO.both(out.shape, out.data)
      singleIO.asserting(x => ((x._1(0),
                                x._1(1),
                                x._2.indices.maxBy(x._2))
                              shouldBe
                               (1,
                                1000,
                                549)))

   }
}
