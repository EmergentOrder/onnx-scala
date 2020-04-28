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

  val memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)

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

       
      val inputTensors = Array(
      getInput(inputs._1),
      getInput(inputs._2),
      getInput(inputs._3),
      getInput(inputs._4),
      getInput(inputs._5),
      getInput(inputs._6),
      getInput(inputs._7),
      getInput(inputs._8),
      getInput(inputs._9)
    ).flatten
 
    val output = runModel(
      session,
      inputTensors,
      allNodeNamesAndDims._1,
      allNodeNamesAndDims._2,
      allNodeNamesAndDims._3
    )
//    val outputPointer = out.get(0).GetTensorMutableDataFloat().capacity(inputs.GetTensorTypeAndShapeInfo().GetElementCount());

//    println(outputPointer.get(0).IsTensor())

    output.asInstanceOf[T9]
  }

   def getSession(bytes: Array[Byte]) = {
    val env = new Env(ORT_LOGGING_LEVEL_WARNING, "test")

    val session_options = new SessionOptions

    val modelString = new BytePointer(bytes: _*).capacity(bytes.size)


    new Session(env, modelString, bytes.size, session_options)

  }
 
   def getInput[T: ClassTag](
        input: T
    ): Option[Value] = {
     input match {
        case tensorOpt: Option[Tensor[Any]] => {
          tensorOpt match {
            case None    => None
          }
        }
      case tensor: Tensor[Any] => {
        Some(getTensor(tensor))
      }
    }
  }

  def runModel(
      sess: Session,
      input_tensor_values: Array[Value],
      inputNames: PointerPointer[BytePointer],
      nodeDims: Array[LongPointer],
      outputNames: PointerPointer[BytePointer] 
  ) = {

    
    val value = new Value(input_tensor_values.size)
  


    val input_tensor_size = (0 until input_tensor_values.size).foreach{i =>
      value.position(i).put(input_tensor_values(i)) 
    }


    val output_tensor = sess.Run(new RunOptions(), inputNames, value.position(0), input_tensor_values.size, outputNames, 1) 

    //TODO: More outputs
    val firstOut = output_tensor.get(0)
      val size: Long = firstOut.GetTensorTypeAndShapeInfo.GetElementCount

      val shape: LongPointer = firstOut.GetTensorTypeAndShapeInfo.GetShape();
 
    getTensorFromValue(firstOut, shape)
  }

  def getTensorFromValue(value: Value, shapePointer: LongPointer) = {
    val dtype = value.GetTensorTypeAndShapeInfo.GetElementType
    val size = value.GetTensorTypeAndShapeInfo.GetElementCount
    val shape = (0 until shapePointer.capacity().toInt).map(x => shapePointer.get(x).toInt).toArray
    val arr = dtype match { 
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
    TensorFactory.getTensor(arr, shape)
  }


  //TODO: Rest of the types
  def getTensor[T:ClassTag](input: T): Value = {

    input match{
      case Some(tens: Tensor[Long]) => {
        try{
          getTensorLong(tens)
        }
        catch{
          case e: Throwable => getTensorFloat(tens.asInstanceOf[Tensor[Float]])
        }
      }
      case tens: Tensor[Long] => {

        try{
          getTensorLong(tens)
        }
        catch{
          case e: Throwable => getTensorFloat(tens.asInstanceOf[Tensor[Float]])
        }
      }
//      case Some(tens: Tensor[Float]) => {
//        getTensorFloat(tens)
//      }
//      case tens: Tensor[Float] => {
//        getTensorFloat(tens)
//      }
    }
  }

  def getTensorLong(tens: Tensor[Long]): Value = {

          val inputArray = tens._1

          val inputPointer = new LongPointer(inputArray: _*)

//          input_node_names.put(i,new BytePointer(i.toString))

          val dims = new LongPointer(tens._2.size)
          (0 until tens._2.size).map{i =>
            dims.put(i, tens._2(i))
      }


      val size: Long = dims.capacity
      val inputTensorSize = (0 until size.toInt).map(j => dims.get(j)).reduce(_*_)

      val inputTensor: Value = Value.CreateTensorLong(
            memory_info.asOrtMemoryInfo,
            inputPointer,
            inputTensorSize,
            dims,
            size
          )
      inputTensor
  }

  def getTensorFloat(tens: Tensor[Float]): Value = {

          val inputArray = tens._1

          val inputPointer = new FloatPointer(inputArray: _*)

//          input_node_names.put(i,new BytePointer(i.toString))

          val dims = new LongPointer(tens._2.size)
          (0 until tens._2.size).map{i =>
            dims.put(i, tens._2(i))
      }


      val size: Long = dims.capacity
      val inputTensorSize = (0 until size.toInt).map(j => dims.get(j)).reduce(_*_)

      val inputTensor: Value = Value.CreateTensorFloat(
            memory_info.asOrtMemoryInfo,
            inputPointer,
            inputTensorSize,
            dims,
            size
          )
      inputTensor
  }

  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
