package org.emergentorder.onnx.backends

import scala.reflect.ClassTag
import scala.language.implicitConversions
import org.bytedeco.javacpp._
import org.bytedeco.onnxruntime._
import org.bytedeco.onnxruntime.global.onnxruntime._

import org.emergentorder.onnx._

  
val env = new Env(ORT_LOGGING_LEVEL_WARNING, "onnx-scala" + System.currentTimeMillis)

trait ORTOperatorBackend
    extends OpToONNXBytesConverter
    with AutoCloseable {

  val allocator = new AllocatorWithDefaultOptions()

  val memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)

  def getSession(bytes: Array[Byte]) = { 
    val session_options = new SessionOptions

//    session_options.SetIntraOpNumThreads(1)
    //Using DNNL
//    OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options.asOrtSessionOptions(), 1)
    val modelString = new BytePointer(bytes: _*).capacity(bytes.size)


    new Session(env, modelString, bytes.size, session_options)

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
    val shape: LongPointer = firstOut.GetTensorTypeAndShapeInfo.GetShape()

    getTensorFromValue(firstOut, shape)
  }

  def getTensorFromValue(value: Value, shapePointer: LongPointer) = {
    val dtype = value.GetTensorTypeAndShapeInfo.GetElementType
    val size = value.GetTensorTypeAndShapeInfo.GetElementCount
    val shape = (0 until shapePointer.capacity.toInt).map(x => shapePointer.get(x).toInt).toArray
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
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL =>{
        val point = value.GetTensorMutableDataBool.capacity(size)
        val booleanPoint = new BooleanPointer(point.asByteBuffer) //C++ bool size is not defined, could cause problems on some platforms
        (0 until booleanPoint.capacity().toInt).map { x =>
          booleanPoint.get(x)
        }.toArray
      }
    }
    TensorFactory.getTensor(arr, shape)
  }

  def getTensor[T:ClassTag](input: T): Value = {

    input match{

      case tensorOpt: Option[_] => {
        tensorOpt match {
          case Some(tens: Tensor[_]) => {
            val value: Value = tens._1 match {
              case b: Array[Byte] => getTensorByte(tens)
              case s: Array[Short] => getTensorShort(tens)
              case d: Array[Double] => getTensorDouble(tens)
              case f: Array[Float] => getTensorFloat(tens)
              case i: Array[Int]   => getTensorInt(tens)
              case l: Array[Long]  => getTensorLong(tens)
            }
            value
          }
          case None =>  new Value()
        }
      }

      case tensorOpt: Tensor[_] => {
        tensorOpt match {
          case tens => {
            val value: Value = tens._1 match {
              case b: Array[Byte] => getTensorByte(tens)
              case s: Array[Short] => getTensorShort(tens)
              case d: Array[Double] => getTensorDouble(tens)
              case f: Array[Float] => getTensorFloat(tens)
              case i: Array[Int]   => getTensorInt(tens)
              case l: Array[Long]  => getTensorLong(tens)
            }
            value
          }
        }
      }

    }
  }

  def getTensorByte(tens: Tensor[Byte]): Value = {
          val inputArray = tens._1
          val inputPointer = new BytePointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
          val dims = new LongPointer(tens._2.size)
          (0 until tens._2.size).map{i =>
            dims.put(i, tens._2(i))
      }

      val size: Long = dims.capacity
      val inputTensorSize = (0 until size.toInt).map(j => dims.get(j)).reduce(_*_)

      val inputTensor: Value = Value.CreateTensorByte(
            memory_info.asOrtMemoryInfo,
            inputPointer,
            inputTensorSize,
            dims,
            size
          )
      inputTensor
  }

  def getTensorShort(tens: Tensor[Short]): Value = {
          val inputArray = tens._1
          val inputPointer = new ShortPointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
          val dims = new LongPointer(tens._2.size)
          (0 until tens._2.size).map{i =>
            dims.put(i, tens._2(i))
      }

      val size: Long = dims.capacity
      val inputTensorSize = (0 until size.toInt).map(j => dims.get(j)).reduce(_*_)

      val inputTensor: Value = Value.CreateTensorShort(
            memory_info.asOrtMemoryInfo,
            inputPointer,
            inputTensorSize,
            dims,
            size
          )
      inputTensor
  }

  def getTensorDouble(tens: Tensor[Double]): Value = {
          val inputArray = tens._1
          val inputPointer = new DoublePointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
          val dims = new LongPointer(tens._2.size)
          (0 until tens._2.size).map{i =>
            dims.put(i, tens._2(i))
      }

      val size: Long = dims.capacity
      val inputTensorSize = (0 until size.toInt).map(j => dims.get(j)).reduce(_*_)

      val inputTensor: Value = Value.CreateTensorDouble(
            memory_info.asOrtMemoryInfo,
            inputPointer,
            inputTensorSize,
            dims,
            size
          )
      inputTensor
  }

  def getTensorInt(tens: Tensor[Int]): Value = {
          val inputArray = tens._1
          val inputPointer = new IntPointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
          val dims = new LongPointer(tens._2.size)
          (0 until tens._2.size).map{i =>
            dims.put(i, tens._2(i))
      }

      val size: Long = dims.capacity
      val inputTensorSize = (0 until size.toInt).map(j => dims.get(j)).reduce(_*_)

      val inputTensor: Value = Value.CreateTensorInt(
            memory_info.asOrtMemoryInfo,
            inputPointer,
            inputTensorSize,
            dims,
            size
          )
      inputTensor
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

    Tuple1(output.asInstanceOf[T])
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
