package org.emergentorder.onnx.backends

import cats.effect.IO
import cats.effect.testing.scalatest.AsyncIOSpec
import org.emergentorder.compiletime._
import org.emergentorder.io.kjaer.compiletime._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends._
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should._

import scala.language.postfixOps

class ONNXScalaSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {
//   Required to use onnxruntime-node
//   import org.emergentorder.onnx.onnxruntimeNode.mod.listSupportedBackends
//   listSupportedBackends()

//import org.emergentorder.onnx.onnxruntimeWeb.wasmMod.onnxruntimeBackend.createInferenceSessionHandler
   implicit override def executionContext =
      scala.scalajs.concurrent.JSExecutionContext.Implicits.queue

   // TODO: push this inside ORTWebModelBackend, and use other create() which takes arraybufferlike
   val session: IO[
     org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
   ] = IO.fromFuture(IO {
//      val infSess = new org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession()

      org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
         .create(
           "./squeezenet1_1_Opset18.onnx", {
              val opts =
                 org.emergentorder.onnx.onnxruntimeCommon.inferenceSessionMod.InferenceSession
                    .SessionOptions()
              opts.executionProviders = scala.scalajs.js.Array("cpu")
//              opts.intraOpNumThreads = 1
//              opts.interOpNumThreads = 1
              opts
           }
         )
         .toFuture
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
      Tensor(data, shape)
      val out = squeezenet.fullModel[
        Float,
        "ImageNetClassification",
        "Batch" ##: "Class" ##: TSNil,
        1 #: 1000 #: SNil
      ](Tuple(imageTens))

      // js.Dynamic.global.foo = "42"
      // js.Dynamic.global.ort.env.wasm.wasmPaths.asInstanceOf[js.UndefOr[String]]
      // The output shape
      // and the highest probability (predicted) class
      val both = cats.effect.IO.both(out.shape, out.data)
      both.asserting(x =>
         ((x._1(0), x._1(1), x._2.indices.maxBy(x._2))
            shouldBe
               (1, 1000, 753))
      )

   }
}
