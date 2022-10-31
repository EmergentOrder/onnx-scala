package org.emergentorder.onnx.backends

import sys.process._
import scala.language.postfixOps
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._
import typings.onnxruntimeNode.mod.{InferenceSession => OrtSession}

import org.scalatest._

import org.scalatest.flatspec.AsyncFlatSpec
import org.scalatest.matchers.should._

class ONNXScalaSpec extends AsyncFlatSpec with Matchers {

  implicit override def executionContext = scala.scalajs.concurrent.JSExecutionContext.Implicits.queue
  //TODO: push this inside ORTWebModelBackend, and use other create() which takes arraybufferlike
   val sess: scala.scalajs.js.Promise[
        typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession
      ] = OrtSession.create("squeezenet1.0-12.onnx")


   "SqueezeNet ONNX-Scala model" should "predict dummy image class" in {
 
      val futSess = sess.toFuture
      futSess.map{ session =>
        val squeezenet      = new ORTWebModelBackend(session)
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
        out.map{ res =>
          assert(res.shape(0) == 1)
          assert(res.shape(1) == 1000)
          // The highest probability (predicted) class
          assert(res.data.indices.maxBy(res.data) == 549)
        }
      }.flatten
   }
}
