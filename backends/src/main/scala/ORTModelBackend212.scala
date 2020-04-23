package org.emergentorder.onnx.backends

import scala.reflect.ClassTag
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.onnxruntime._
import org.bytedeco.onnxruntime.global.onnxruntime._

import org.emergentorder.onnx._


//TODO: Clean up, remove asInstaceOf, multiple inputs, etc.
class ORTModelBackend(onnxBytes: Array[Byte])
    extends Model(onnxBytes)
    with AutoCloseable {

  val allocator = new AllocatorWithDefaultOptions()

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

    val tens = inputs._1.asInstanceOf[Tensor[Float]]
    val inputArray = tens._1

    val inputPointer = new FloatPointer(inputArray.asInstanceOf[Array[Float]]: _*)
      
    val size: Long = tens._2.size
     
    val lp: LongPointer = new LongPointer(size)
    (0 until size.toInt).map(i => lp.put(i,tens._2(i)))

    val memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)

    val inputTensorSize = (0 until size.toInt).map(j => lp.get(j)).reduce(_*_)

    val inputTensor: Value = Value.CreateTensorFloat(
            memory_info.asOrtMemoryInfo,
            inputPointer,
            inputTensorSize,
            lp,
            size
          )

    
    //TODO: multiple inputs
    val output = runModel(
      session,
      Array(inputTensor),
      allNodeNamesAndDims._1,
      allNodeNamesAndDims._2,
      allNodeNamesAndDims._3
    )
//    val outputPointer = out.get(0).GetTensorMutableDataFloat().capacity(inputs.GetTensorTypeAndShapeInfo().GetElementCount());

//    println(outputPointer.get(0).IsTensor())

    val shapeSize: Long = output._2.capacity
    val shape = (0 until shapeSize.toInt).map(x => output._2.get(x).toInt).toArray

    TensorFactory.getTensor(output._1, shape).asInstanceOf[T9]
  }

   def getSession(bytes: Array[Byte]) = {
    val env = new Env(ORT_LOGGING_LEVEL_WARNING, "test")

    val session_options = new SessionOptions

    val modelString = new BytePointer(bytes: _*).capacity(bytes.size)


    new Session(env, modelString, bytes.size, session_options)

  }

  //TODO: Support more than floats
  def runModel(
      sess: Session,
      input_tensor_values: Array[Value],
      inputNames: PointerPointer[BytePointer],
      nodeDims: Array[LongPointer],
      outputNames: PointerPointer[BytePointer] 
  ) = {

    
    val value = new Value(input_tensor_values.size)
  
//    val memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)  

    val input_tensor_size = (0 until input_tensor_values.size).foreach{i =>
/*
      val size: Long = nodeDims(i).capacity
      val inputTensorSize = (0 until size.toInt).map(j => nodeDims(i).get(j)).reduce(_*_)
      
      val inputTensor: Value = Value.CreateTensorFloat(
            memory_info.asOrtMemoryInfo,
            input_tensor_values(i),
            inputTensorSize,
            nodeDims(i),
            size
          )
  */    
      value.position(i).put(input_tensor_values(i)) 
    }


    val output_tensor = sess.Run(new RunOptions(), inputNames, value.position(0), input_tensor_values.size, outputNames, 1) 

    //TODO: More outputs
    val firstOut = output_tensor.get(0)
      val size: Long = firstOut.GetTensorTypeAndShapeInfo.GetElementCount

      val shape: LongPointer = firstOut.GetTensorTypeAndShapeInfo.GetShape();
 
    (getTensorFromValue(firstOut), shape)
  }

  def getTensorFromValue(value: Value) = {
    val dtype = value.GetTensorTypeAndShapeInfo.GetElementType
    val size = value.GetTensorTypeAndShapeInfo.GetElementCount
    dtype match {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT =>{
        val point = value.GetTensorMutableDataFloat.capacity(size)
        val buff = point.asByteBuffer.asFloatBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE =>{
        val point = value.GetTensorMutableDataDouble.capacity(size)
        val buff = point.asByteBuffer.asDoubleBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 =>{
        val point = value.GetTensorMutableDataByte.capacity(size)
        val buff = point.asByteBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 =>{
        val point = value.GetTensorMutableDataShort.capacity(size)
        val buff = point.asByteBuffer.asShortBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 =>{
        val point = value.GetTensorMutableDataInt.capacity(size)
        val buff = point.asByteBuffer.asIntBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 =>{
        val point = value.GetTensorMutableDataLong.capacity(size)
        val buff = point.asByteBuffer.asLongBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
    }
  }


  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
