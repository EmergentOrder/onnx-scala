package org.emergentorder.onnx.backends

import scala.concurrent.duration._
import typings.onnxjs.onnxImplMod.Tensor.{^ => Tensor}
//import typings.onnxjs.libTensorMod.Tensor.DataTypeMap.DataTypeMapOps
import typings.onnxjs.onnxImplMod.InferenceSession.{^ => InferenceSession}
//import typings.onnxjs.onnxMod.Onnx

import typings.onnxjs.onnxImplMod._

import scala.scalajs.js.|

trait ONNXJSOperatorBackend{
  //  extends OpToONNXBytesConverter
   // with AutoCloseable {
implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
  def test() = {

    val session = new InferenceSession()
    val url = "relu.onnx"
    val modelFuture = session.loadModel(url).toFuture

    val dataTypes = new typings.onnxjs.libTensorMod.Tensor.FloatType {} 

    val outputFuture = modelFuture.map{x =>
      val inputs = Array(new Tensor(
        scala.scalajs.js.Array[Boolean | Double]((1 until 61).map(_.toDouble:Boolean | Double).toArray:_*), 
      typings.onnxjs.onnxjsStrings.float32, scala.scalajs.js.Array(3.0, 4.0, 5.0)):typings.onnxjs.tensorMod.Tensor);
      println("before run")
      val res = session.run(scala.scalajs.js.Array(inputs:_*)).toFuture
      println("after run")
      res
    }.flatten


import scala.util.{Success, Failure}

    outputFuture onComplete {
      case Success(t) => println(t.get("y").get.dims)
      case Failure(fail) => println(fail)
    }

  }
}
