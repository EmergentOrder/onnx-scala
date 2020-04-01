package org.emergentorder.onnx.backends

import scala.reflect.ClassTag
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.onnxruntime._
import org.bytedeco.onnxruntime.global.onnxruntime._

import org.emergentorder.onnx._

class ORTModelBackend(onnxBytes: Array[Byte])
//    extends Model(onnxBytes)
    extends AutoCloseable {

  val allocator = new AllocatorWithDefaultOptions()

  //TODO: multiple inputs
  def getSession() = {
    val env = new Env(ORT_LOGGING_LEVEL_WARNING, "test")

    // initialize session options if needed
    val session_options = new SessionOptions

    val modelString = new BytePointer(onnxBytes: _*).capacity(onnxBytes.size)

    //System.out.println("Using Onnxruntime C++ API");
    new Session(env, modelString, onnxBytes.size, session_options)

    //new Session(env, new BytePointer("squeezenet1.1.onnx1"), session_options)

  }

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

  val session = getSession()

  val allNodeNamesAndDims = getInputAndOutputNodeNamesAndDims(session)

  //TODO: Support more than floats
  def runModel(
      sess: Session,
      input_tensor_values: FloatPointer,
      inputNames: PointerPointer[BytePointer],
      nodeDims: Array[LongPointer],
      outputNames: PointerPointer[BytePointer]
  ) = {
    //TODONT: Hardcode
    val input_tensor_size = 224 * 224 * 3 // simplify ... using known dim values to calculate size
    // use OrtGetTensorShapeElementCount() to get official size!

//      val input_tensor_values = new FloatPointer(input_tensor_size)
//      val output_node_names = new PointerPointer[BytePointer]("squeezenet0_flatten0_reshape0")

    // initialize input data with values in [0.0, 1.0]
//      val idx = FloatIndexer.create(input_tensor_values)
//      (0 until input_tensor_size).foreach{i => idx.put(i, (i.toFloat / (input_tensor_size + 1))) }

    // create input tensor object from data values
    val memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    val input_tensor = Value.CreateTensorFloat(
      memory_info.asOrtMemoryInfo,
      input_tensor_values,
      input_tensor_size.toLong,
      nodeDims(0),
      4
    )
    assert(input_tensor.IsTensor())

    val size = input_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    // score model & input tensor, get back output tensor
    val output_tensor = sess.Run(new RunOptions(), inputNames, input_tensor, 1, outputNames, 1)
//      assert(output_tensor.size() == 1 && output_tensor.get(0).IsTensor())

    // Get pointer to output tensor float values

//      val floatarr = output_tensor.get(0).GetTensorMutableDataFloat();
//      assert(Math.abs(floatarr.get(0) - 0.000045) < 1e-6)
//      val info: UnownedTensorTypeAndShapeInfo = output_tensor.get(0).GetTypeInfo().GetTensorTypeAndShapeInfo()
//        println(info)
    //.GetElementCount())
    //

    //TODO: Don't hardcode
    output_tensor.get(0).GetTensorMutableDataFloat().capacity(1000);
//       output_tensor
  }

//  val input_tensor_values = new FloatPointer(224*224*3)
//  val out = runModel(session, input_tensor_values, allNodeNamesAndDims._1, allNodeNamesAndDims._2)

  def fullModel[
      T: ClassTag
  ](
      inputs: Tensor[T]
  ): (Tuple1[Tensor[T]]) = {

    val inputArray = inputs._1

    val inputPointer = new FloatPointer(inputArray.asInstanceOf[Array[Float]]: _*)
    val outputPointer = runModel(
      session,
      inputPointer,
      allNodeNamesAndDims._1,
      allNodeNamesAndDims._2,
      allNodeNamesAndDims._3
    )
//    val outputPointer = out.get(0).GetTensorMutableDataFloat().capacity(inputs.GetTensorTypeAndShapeInfo().GetElementCount());

//    println(outputPointer.get(0).IsTensor())

    val fb = outputPointer.asByteBuffer.asFloatBuffer

    val res = (0 until fb.capacity).map { x =>
      fb.get(x).asInstanceOf[Float] //unsafe : asInstanceOf
    }.toArray

    Tuple1(TensorFactory.getTensor(res, Array(1, 1000)).asInstanceOf[Tensor[T]])
  }

  /*
  def callOp[
      T: ClassTag,
  ](
      name: String,
      opName: String,
      inputs: Option[NonEmptyTuple],
      //    outName: String,
      attrs: Map[String, Any]
  ): (Tuple1[T]) = ???
   */

  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
