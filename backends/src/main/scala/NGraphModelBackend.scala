package org.emergentorder.onnx.backends

import scala.reflect.ClassTag
import org.bytedeco.javacpp._
import org.bytedeco.ngraph.global._
import ngraph.import_onnx_model

import org.emergentorder.onnx.Model

class NGraphModelBackend(onnxBytes: Array[Byte])
    extends Model(onnxBytes)
    with NGraphOperatorBackend
    with AutoCloseable {

  val modelString = new BytePointer(onnxBytes: _*)

  val ngraphFunc = import_onnx_model(modelString)

  val executable  = ngraphBackend.compile(ngraphFunc)
  val outputShape = ngraphFunc.get_output_shape(0)
  val outputType  = ngraphFunc.get_output_element_type(0)

  override def fullModel[
      T: ClassTag
  ](
      inputs: Option[NonEmptyTuple]
  ): (Tuple1[T]) = {
    callNGraphExecutable[
      T
    ](
      executable,
      inputs,
      outputShape,
      outputType
    )
  }

  override def close(): Unit = {
    executable.close
    outputShape.close
    outputType.close
    modelString.close
    ngraphFunc.close
    scope.close
    super.close
  }
}
