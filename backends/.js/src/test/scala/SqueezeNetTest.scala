package org.emergentorder.onnx.backends

import sys.process._
import scala.language.postfixOps
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.backends._
import org.emergentorder.compiletime._
import org.emergentorder.io.kjaer.compiletime._
import org.emergentorder.onnx.onnxruntimeNode.mod.binding.InferenceSession
import cats.effect.IO

import org.scalatest._

import scala.concurrent.Future
import org.scalatest.flatspec.AsyncFlatSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should._
import cats.effect.testing.scalatest.AsyncIOSpec


class ONNXScalaSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {

  import org.emergentorder.onnx.onnxruntimeNode.mod.listSupportedBackends
  println(listSupportedBackends())

   implicit override def executionContext =
      scala.scalajs.concurrent.JSExecutionContext.Implicits.queue
   // TODO: push this inside ORTWebModelBackend, and use other create() which takes arraybufferlike
   val session: IO[
     org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession 
   ] = IO.fromFuture(IO {
//      val infSess = new org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession()

         org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession.create(
           "./squeezenet1_1_Opset18.onnx", {
              val opts = org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession.SessionOptions()
              opts.executionProviders = scala.scalajs.js.Array("cpu")
              opts
           }
         ).toFuture
   })

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
        1 #: 1000 #: SNil
      ](Tuple(imageTens))

      // The output shape
      // and the highest probability (predicted) class
      val singleIO = cats.effect.IO.both(out.shape, out.data)
      singleIO.asserting(x =>
         ((x._1(0), x._1(1), x._2.indices.maxBy(x._2))
            shouldBe
               (1,
               1000,
               753))
      )

   }
}
