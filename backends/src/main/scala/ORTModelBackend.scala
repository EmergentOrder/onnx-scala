package org.emergentorder.onnx.backends

import scala.reflect.ClassTag
import scala.language.implicitConversions
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.onnxruntime._
import org.bytedeco.onnxruntime.global.onnxruntime._

import org.emergentorder.onnx._

//TODO: Clean up, remove asInstaceOf, etc.
class ORTModelBackend(onnxBytes: Array[Byte])
    extends Model(onnxBytes)
    with ORTOperatorBackend
    with AutoCloseable {

  def getInputAndOutputNodeNamesAndDims(sess: Session) = {
    val num_input_nodes  = session.GetInputCount();
    val input_node_names = new PointerPointer[BytePointer](num_input_nodes);

//    System.out.println("Number of inputs = " + num_input_nodes);

    val inputNodeDims = (0 until num_input_nodes.toInt).map { i =>
      // print input node names
      val input_name = session.GetInputName(i, allocator.asOrtAllocator())
//      println("Input " + i + " : name=" + input_name.getString())
      input_node_names.put(i, input_name)

      // print input node types
      val type_info   = session.GetInputTypeInfo(i)
      val tensor_info = type_info.GetTensorTypeAndShapeInfo()

//        val type = tensor_info.GetElementType()
//        println("Input " + i + " : type=" + type)

      // print input shapes/dims
      tensor_info.GetShape()
      //println("Input " + i + " : num_dims=" + input_node_dims.capacity())

    }

    val num_output_nodes  = session.GetOutputCount()
    val output_node_names = new PointerPointer[BytePointer](num_output_nodes)
    (0 until num_output_nodes.toInt).map { i =>
      val outputName = session.GetOutputName(i, allocator.asOrtAllocator());
      output_node_names.put(i, outputName);
    }

    (input_node_names, inputNodeDims.toArray, output_node_names)
  }

  val session = getSession(onnxBytes)

  val allNodeNamesAndDims = getInputAndOutputNodeNamesAndDims(session)

  override def fullModel[
      T: ClassTag
  ](
      inputs: Option[NonEmptyTuple]
  ): (Tuple1[T]) = {

    inputs match {
      case Some(x) => {

        val size = x.size
        val inputTensors = (0 until size).map { i =>
          val tens               = x.apply(i)
          val inputTensor: Value = getTensor(tens)
          inputTensor
        }.toArray

        val output = runModel(
          session,
          inputTensors,
          allNodeNamesAndDims._1,
          allNodeNamesAndDims._2,
          allNodeNamesAndDims._3
        )
//    val outputPointer = out.get(0).GetTensorMutableDataFloat().capacity(inputs.GetTensorTypeAndShapeInfo().GetElementCount());

//    println(outputPointer.get(0).IsTensor())

        Tuple1(output.asInstanceOf[T])
      }
      case None => Tuple1(TensorFactory.getTensor(Array(), Array[Int]()).asInstanceOf[T])

    }
  }

  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
