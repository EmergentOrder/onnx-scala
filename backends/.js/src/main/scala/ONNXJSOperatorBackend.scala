package org.emergentorder.onnx.backends

import scala.concurrent.duration._
//import typings.onnxruntimeWeb.tensorMod._
//import typings.onnxruntimeWeb.tensorMod.Tensor.FloatType
//import typings.onnxruntimeWeb.tensorMod.Tensor.DataType
//import typings.onnxjs.libTensorMod.Tensor.DataTypeMap.DataTypeMapOps
import typings.onnxruntimeWeb.mod.InferenceSession
import typings.onnxruntimeWeb.mod.Tensor.{^ => Tensor}
//import typings.onnxruntimeWeb.ort.InferenceSession.{^ => InferenceSess}
//import typings.onnxjs.onnxMod.Onnx
import scala.scalajs.js.typedarray
//import typings.onnxruntimeWeb.onnxImplMod._

//import scala.scalajs.js.Thenable.Implicits._
import scala.concurrent.Await
import scala.language.postfixOps
import scala.scalajs.js

trait ONNXJSOperatorBackend {
   //  extends OpToONNXBytesConverter
   // with AutoCloseable {
   implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
   def test() = {

      val session   = InferenceSession.create("squeezenet1.1.onnx")
//      val dataTypes = new FloatType {}

      val dataType = "float32"
      val dims = scala.scalajs.js.Array(1.0, 3.0, 224.0, 224.0)

      val rawData = typedarray.floatArray2Float32Array((0 until 150528).map(_ => 42.0f).toArray) 
      val tensor = new Tensor(dataType, rawData, dims)
     
      //r.Tensor(dims, dataType.asInstanceOf[typings.onnxruntimeWeb.tensorMod.Tensor.DataType])
      //data.set(scala.scalajs.js.Array[Double](0.0),rawData) 

      //println(tensor.data)
      val inputs = Array(tensor)

      val feeds = js.Dictionary("data" -> tensor)
      val sess = session.toFuture
      sess.foreach { realSess =>
        val res = realSess.run(feeds.asInstanceOf[typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession.FeedsType])
        res.toFuture.foreach {result => 
           println("RESULT :" + result.asInstanceOf[typings.onnxruntimeCommon.inferenceSessionMod.InferenceSession.OnnxValueMapType].get("squeezenet0_flatten0_reshape0").getOrElse(null).data)
        }
      }

      //TODO: FIX README, to clarify that squeezenet output unnormalized scores, not probabilities

      //Getting error:
      //worker.js onmessage() captured an uncaught exception: ReferenceError: ortWasmThreaded is not defined

      //
      //
      //println("before run")
      //val res = sess.run(scala.scalajs.js.Array(inputs: _*))
      //println("after run")
      //res
   }
}
