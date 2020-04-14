package org.emergentorder.onnx.backends

import scala.reflect.ClassTag
import org.bytedeco.javacpp._
import org.bytedeco.onnxruntime._
import org.bytedeco.onnxruntime.global.onnxruntime._

import org.emergentorder.onnx._

trait ORTOperatorBackend
    extends OpToONNXBytesConverter
    with AutoCloseable {

  val allocator = new AllocatorWithDefaultOptions()

  val memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)

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
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => value.GetTensorMutableDataFloat.capacity(size)
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => value.GetTensorMutableDataByte().capacity(size)
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => value.GetTensorMutableDataShort().capacity(size) 
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => value.GetTensorMutableDataInt().capacity(size) 
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => value.GetTensorMutableDataLong().capacity(size)
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => value.GetTensorMutableDataDouble().capacity(size)
    }
  }

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


  def callByteArrayOp[
      T: ClassTag
  ](
      opModel: Array[Byte],
      inputs: Option[NonEmptyTuple]
  ): (Tuple1[T]) = {

    val sess = getSession(opModel) 

    inputs match{
      case Some(x) => {

    val input_node_names = new PointerPointer[BytePointer](x.size)
   
    //TODO: more outputs
    val output_node_names = new PointerPointer[BytePointer](1) 
 

    val inputDimsAndValues: Array[Tuple2[LongPointer, Value]] = (0 until x.size).map{i => 
     

      input_node_names.put(i,new BytePointer(i.toString))

      (null, getTensor(x(i)))
    }.toArray

    output_node_names.put(0l,new BytePointer("outName"))
    
    //println(tens._2(0))
    val output = runModel(
      sess, 
      inputDimsAndValues.map(_._2),
      input_node_names,
      inputDimsAndValues.map(_._1),
      output_node_names
    )

    //TODO: data types
    val fb = output._1.asByteBuffer.asFloatBuffer

    val res = (0 until fb.capacity).map { x =>
      fb.get(x).asInstanceOf[Float] //unsafe : asInstanceOf
    }.toArray

    val shapeSize: Long = output._2.capacity
    val shape = (0 until shapeSize.toInt).map(x => output._2.get(x).toInt).toArray


    Tuple1(TensorFactory.getTensor(res, shape).asInstanceOf[T])
      } 
      case None => Tuple1(TensorFactory.getTensor(Array(), Array[Int]()).asInstanceOf[T])
    
    }
  }

  def callOp[
      T: ClassTag](
      name: String,
      opName: String,
      inputs: Option[NonEmptyTuple],
      //    outName: String,
      attrs: Map[String, Any]
  ): (Tuple1[T]) = {
    val bytes = opToONNXBytes(name, opName, inputs, "outName", attrs)
    callByteArrayOp[T](
      bytes,
      inputs
    )
  }

  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
