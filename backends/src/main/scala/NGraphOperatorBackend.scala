package org.emergentorder.onnx.backends

import java.io.PrintWriter;
import java.io.File;
import java.io.FileInputStream;
import java.nio.file._

import scala.{specialized => sp}
import scala.collection.mutable.{Map => MMap};
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.Numeric
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex

import org.emergentorder.onnx._
import org.emergentorder.union._
import org.bytedeco.javacpp._;
import org.bytedeco.onnx.ModelProto;
import org.bytedeco.onnx.global.onnx.ParseProtoFromBytes;
import org.bytedeco.onnx.MessageLite;
import org.bytedeco.onnx.NodeProto;
import org.bytedeco.onnx.GraphProto
import org.bytedeco.ngraph.global._
import ngraph.import_onnx_model
import org.bytedeco.ngraph.Backend

//TODO: Make typeclasses for ops
//TODO: ORT backend, dotty only
//TODO: fix readme perf claims, closer to 20% over ORT
//TODO: Fix program generator, port over shape-safe tensors (use tf-dotty Shape?)
//TODO: Update README for dotty, mdoc doesn't support it though..
//TODO: consider wrong output data type
// TODO: check import org.bytedeco.onnx.global.onnx.check_model

//TODEFER: ONNX-JS backend for both JS and JVM
//TODEFER: ONNX Runtime backend for JVM (and Native?)
trait NGraphOperatorBackend
    extends OpToONNXBytesConverter
    with NGraphBackendUtils
    with AutoCloseable {

  val scope = new PointerScope()

  val ngraphBackend = Backend.create("CPU")

  def callByteArrayOp[
      T: ClassTag
  ](
      opModel: Array[Byte],
      inputs: Option[NonEmptyTuple]
  ): (Tuple1[T]) = {
    val modelString = new BytePointer(opModel: _*)

    val ngraphFunc = import_onnx_model(modelString)
//    modelString.close

    val outputShape = ngraphFunc.get_output_shape(0)
    val outputType  = ngraphFunc.get_output_element_type(0)

    val executable = ngraphBackend.compile(ngraphFunc)

    //  ngraphFunc.close
    val res = callNGraphExecutable[
      T
    ](
      executable,
      inputs,
      outputShape,
      outputType
    )

//    outputShape.close
//    outputType.close
    //  executable.close
    res
  }

  def callNGraphExecutable[
      T: ClassTag
  ](
      executable: org.bytedeco.ngraph.Executable,
      inputs: Option[NonEmptyTuple],
      outputShape: org.bytedeco.ngraph.Shape,
      outputType: org.bytedeco.ngraph.Type
  ): (Tuple1[T]) = {
    val scope = new PointerScope()

    val inputShapes: Seq[org.bytedeco.ngraph.Shape] = inputs match {
      case Some(x) => {
        val size: Int = x.size
        (0 until size).map(y => getTensorShape(x(y))).flatten
      }
      case None => Seq()
    }

    val inputTensors: Seq[(Pointer, org.bytedeco.ngraph.Type)] = inputs match {
      case Some(x) => {
        val size: Int = x.size
        (0 until size).map(y => getTensorPointerAndType(x(y))).flatten
      }
      case None => Seq()
    }

    val ngraphInputs =
      (inputShapes zip inputTensors).map(x => ngraphBackend.create_tensor(x._2._2, x._1, x._2._1))

    val output = ngraphBackend.create_tensor(outputType, outputShape)

//    println("OUTPUT TYPE" + outputType.get_type_enum())
    val inputVector  = new org.bytedeco.ngraph.TensorVector(ngraphInputs: _*)
    val outputVector = new org.bytedeco.ngraph.TensorVector(output)

    def t = {
      val before = System.nanoTime
      executable.call(outputVector, inputVector)
      val after = System.nanoTime
//      println("Elapsed per Op: " + "  : " + (after - before))
    }

    t

    //convert result to onnx-scala Tensor

    val result = tensorVectorToOutputTensor[T](outputVector, outputShape)

    /*
    inputTensors.foreach { x: (Pointer, org.bytedeco.ngraph.Type) =>
      x._1.close //close pointers
      x._2.close //close shapes
    }

    inputShapes.foreach { x: org.bytedeco.ngraph.Shape =>
      x.close
    }

    ngraphInputs.foreach { x: org.bytedeco.ngraph.Tensor =>
      x.close
    }

    inputVector.close
    output.close
    outputVector.close
     */
//    scope.close

    result *: ()
  }

  def callOp[
      T: ClassTag
  ](
      name: String,
      opName: String,
      inputs: Option[NonEmptyTuple],
      //    outName: String,
      attrs: Map[String, Any]
  ): (Tuple1[T]) = {
    val onnxBytes = opToONNXBytes(name, opName, inputs, "outName", attrs)
    callByteArrayOp[T](
      onnxBytes,
      inputs
    )
  }

  override def close(): Unit = {
    ngraphBackend.close
    scope.close
    super.close
  }
}
