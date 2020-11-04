package org.emergentorder.onnx.backends

import scala.language.implicitConversions
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.indexer.FloatIndexer
import ai.onnxruntime._
import scala.jdk.CollectionConverters._

import org.emergentorder.onnx._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import ORTTensorUtils._

//TODO: Clean up, remove asInstaceOf, etc.
class ORTModelBackend(onnxBytes: Array[Byte])
    extends Model(onnxBytes)
    with ORTOperatorBackend
    with AutoCloseable {

  def getInputAndOutputNodeNamesAndDims(sess: OrtSession) = {
    val input_node_names = session.getInputNames
 
    val inputNodeDims = session.getInputInfo.values.asScala.map(_.getInfo.asInstanceOf[TensorInfo].getShape)

    val output_node_names = session.getOutputNames

    (input_node_names.asScala.toList, inputNodeDims.toArray, output_node_names.asScala.toList)
  }

  val session = getSession(onnxBytes)

  val allNodeNamesAndDims = getInputAndOutputNodeNamesAndDims(session)

  override def fullModel[
      T <: Supported,
      Ax <: Axes
  ](
      inputs: Tuple
  ): Tensor[T, Ax] = {




        val size = inputs.size
        val inputTensors = (0 until size).map { i =>
          val tup               = inputs.drop(i).take(1)
          tup match {
            case t: Tuple1[_] =>
              t(0) match {
                case tens: Tensor[T,Ax] => getOnnxTensor(tens.data, tens.shape, env)
              }
          }
        }.toArray

        val output = runModel(
          session,
          inputTensors,
          allNodeNamesAndDims._1,
          allNodeNamesAndDims._3
        )

        output.asInstanceOf[Tensor[T, Ax]]
  }

  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
