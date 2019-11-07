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

  val executable = ngraphBackend.compile(ngraphFunc)
  val outputShape = ngraphFunc.get_output_shape(0)
  val outputType  = ngraphFunc.get_output_element_type(0)

  override def fullModel[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag,
      T9: ClassTag,
      T10: ClassTag,
      T11: ClassTag,
      T12: ClassTag,
      T13: ClassTag,
      T14: ClassTag,
      T15: ClassTag,
      T16: ClassTag,
      T17: ClassTag
  ](
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8]
  ): (T9) = {
    callNGraphExecutable[T, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17](
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
