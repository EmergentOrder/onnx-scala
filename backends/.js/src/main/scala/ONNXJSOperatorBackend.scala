package org.emergentorder.onnx.backends

import scala.concurrent.duration._
import typings.onnxruntimeWeb.tensorMod.Tensor
import typings.onnxruntimeWeb.tensorMod.Tensor.FloatType
import typings.onnxruntimeWeb.tensorMod.Tensor.DataType
//import typings.onnxjs.libTensorMod.Tensor.DataTypeMap.DataTypeMapOps
import typings.onnxruntimeWeb.mod.InferenceSession
//import typings.onnxruntimeWeb.ort.InferenceSession.{^ => InferenceSession}
//import typings.onnxjs.onnxMod.Onnx

//import typings.onnxruntimeWeb.onnxImplMod._

import scala.scalajs.js.|

trait ONNXJSOperatorBackend {
   //  extends OpToONNXBytesConverter
   // with AutoCloseable {
   implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
   def test() = {

      val session   = InferenceSession.create("relu.onnx")
      val dataTypes = new FloatType {}

      /*
      val inputs = Array(
           new Tensor(
             "float32",
             scala.scalajs.js.Array[Double](
               (1 until 61).map(_.toDouble: Double).toArray: _*
             ),
             scala.scalajs.js.Array(3.0, 4.0, 5.0)
           )
         )
       */
      //println("before run")
      //val res = session.run(scala.scalajs.js.Array(inputs: _*))
      //println("after run")
      //res
   }
}
