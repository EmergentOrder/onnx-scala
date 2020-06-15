package org.emergentorder.onnx.backends

import scala.reflect.ClassTag
import scala.language.existentials
import org.bytedeco.javacpp._
import org.bytedeco.onnxruntime._
import org.bytedeco.onnxruntime.global.onnxruntime._

import org.emergentorder.onnx._

object ORTOperatorBackend {
  val env = new Env(ORT_LOGGING_LEVEL_WARNING, "onnx-scala" + System.currentTimeMillis)
}

trait ORTOperatorBackend extends OpToONNXBytesConverter with AutoCloseable {

  val allocator   = new AllocatorWithDefaultOptions()
  val memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
//  val env = new Env(ORT_LOGGING_LEVEL_WARNING, "onnx-scala" + System.currentTimeMillis)

  def getSession(bytes: Array[Byte]) = {

    val session_options = new SessionOptions

    val modelString = new BytePointer(bytes: _*).capacity(bytes.size)

    new Session(ORTOperatorBackend.env, modelString, bytes.size, session_options)

  }

  def runModel(
      sess: Session,
      input_tensor_values: Array[Value],
      inputNames: PointerPointer[BytePointer], 
      outputNames: PointerPointer[BytePointer]
  ) = {

    val value = new Value(input_tensor_values.size)

    val input_tensor_size = (0 until input_tensor_values.size).foreach { i =>
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

    val output_tensor = sess.Run(
      new RunOptions(),
      inputNames,
      value.position(0),
      input_tensor_values.size,
      outputNames,
      1
    )

    //TODO: More outputs
    val firstOut = output_tensor.get(0)

    val shape: LongPointer = firstOut.GetTensorTypeAndShapeInfo.GetShape()

    getTensorFromValue(firstOut, shape)
  }

  def getTensorFromValue(value: Value, shapePointer: LongPointer) = {
    val dtype = value.GetTensorTypeAndShapeInfo.GetElementType
    val size  = value.GetTensorTypeAndShapeInfo.GetElementCount
    val shape = (0 until shapePointer.capacity().toInt).map(x => shapePointer.get(x).toInt).toArray
    val arr = dtype match {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
        val point = value.GetTensorMutableDataFloat.capacity(size)
        val buff  = point.asByteBuffer.asFloatBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
        val point = value.GetTensorMutableDataDouble.capacity(size)
        val buff  = point.asByteBuffer.asDoubleBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
        val point = value.GetTensorMutableDataByte.capacity(size)
        val buff  = point.asByteBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
        val point = value.GetTensorMutableDataShort.capacity(size)
        val buff  = point.asByteBuffer.asShortBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
        val point = value.GetTensorMutableDataInt.capacity(size)
        val buff  = point.asByteBuffer.asIntBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
        val point = value.GetTensorMutableDataLong.capacity(size)
        val buff  = point.asByteBuffer.asLongBuffer
        (0 until buff.capacity).map { x =>
          buff.get(x)
        }.toArray
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
        val point = value.GetTensorMutableDataBool.capacity(size)
        val booleanPoint =
          new BooleanPointer(
            point.asByteBuffer
          ) //C++ bool size is not defined, could cause problems on some platforms
        (0 until booleanPoint.capacity().toInt).map { x =>
          booleanPoint.get(x)
        }.toArray
      }
    }
    TensorFactory.getTensor(arr, shape)
  }

  def getTensor[T: ClassTag](input: T): Value = {

    input match {

      case tensorOpt: Option[Tensor[_]] => {
        tensorOpt match {
          case Some(tens) => {
            val value: Value = tens._1 match {
              case b: Array[Byte]   => getTensorByte(tens)
              case s: Array[Short]  => getTensorShort(tens)
              case d: Array[Double] => getTensorDouble(tens)
              case f: Array[Float]  => getTensorFloat(tens)
              case i: Array[Int]    => getTensorInt(tens)
              case l: Array[Long]   => getTensorLong(tens)
              case b: Array[Boolean] => getTensorBoolean(tens)
            }
            value
          }
          case None => new Value()
        }
      }

      case tensorOpt: Tensor[_] => {
        tensorOpt match {
          case tens => {
            val value: Value = tens._1 match {
              case b: Array[Byte]   => getTensorByte(tens)
              case s: Array[Short]  => getTensorShort(tens)
              case d: Array[Double] => getTensorDouble(tens)
              case f: Array[Float]  => getTensorFloat(tens)
              case i: Array[Int]    => getTensorInt(tens)
              case l: Array[Long]   => getTensorLong(tens)
              case b: Array[Boolean] => getTensorBoolean(tens)
            }
            value
          }
        }
      }

    }
  }

  def getTensorByte(tens: Tensor[Byte]): Value = {
    val inputArray   = tens._1
    val inputPointer = new BytePointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
    val dims = new LongPointer(tens._2.size)
    (0 until tens._2.size).map { i =>
      dims.put(i, tens._2(i))
    }

    val size: Long      = dims.capacity
    val inputTensorSize = tens._1.size 

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
    val inputArray   = tens._1
    val inputPointer = new ShortPointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
    val dims = new LongPointer(tens._2.size)
    (0 until tens._2.size).map { i =>
      dims.put(i, tens._2(i))
    }

    val size: Long      = dims.capacity
    val inputTensorSize = tens._1.size 

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
    val inputArray   = tens._1
    val inputPointer = new DoublePointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
    val dims = new LongPointer(tens._2.size)
    (0 until tens._2.size).map { i =>
      dims.put(i, tens._2(i))
    }

    val size: Long      = dims.capacity
    val inputTensorSize = tens._1.size 

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
    val inputArray   = tens._1
    val inputPointer = new IntPointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
    val dims = new LongPointer(tens._2.size)
    (0 until tens._2.size).map { i =>
      dims.put(i, tens._2(i))
    }

    val size: Long      = dims.capacity
    val inputTensorSize = tens._1.size 

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
    val inputArray   = tens._1
    val inputPointer = new LongPointer(inputArray: _*)
//          input_node_names.put(i,new BytePointer(i.toString))
    val dims = new LongPointer(tens._2.size)
    (0 until tens._2.size).map { i =>
      dims.put(i, tens._2(i))
    }

    val size: Long      = dims.capacity
    val inputTensorSize = tens._1.size 

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
    (0 until tens._2.size).map { i =>
      dims.put(i, tens._2(i))
    }

    val size: Long      = dims.capacity
    val inputTensorSize = tens._1.size 

    val inputTensor: Value = Value.CreateTensorFloat(
      memory_info.asOrtMemoryInfo,
      inputPointer,
      inputTensorSize,
      dims,
      size
    )
    inputTensor
  }

  def getTensorBoolean(tens: Tensor[Boolean]): Value = {

    val inputArray = tens._1

    val inputPointer = new BoolPointer(new BooleanPointer(inputArray: _*))

//          input_node_names.put(i,new BytePointer(i.toString))

    val dims = new LongPointer(tens._2.size)
    (0 until tens._2.size).map { i =>
      dims.put(i, tens._2(i))
    }

    val size: Long      = dims.capacity
    val inputTensorSize = tens._1.size 

    val inputTensor: Value = Value.CreateTensorBool(
      memory_info.asOrtMemoryInfo,
      inputPointer,
      inputTensorSize,
      dims,
      size
    )
    inputTensor
  }

  def callByteArrayOp[
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
      opModel: Array[Byte],
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8]
  ): (T9) = {

    val sess = getSession(opModel)

    val input_node_names = new PointerPointer[BytePointer](9)

    //TODO: more outputs
    val output_node_names = new PointerPointer[BytePointer](1)

//    val value = new Value(x.size)

//    println(x.size)

    input_node_names.put(0, new BytePointer("A"))
    input_node_names.put(1, new BytePointer("B"))
    input_node_names.put(2, new BytePointer("C"))
    input_node_names.put(3, new BytePointer("D"))
    input_node_names.put(4, new BytePointer("E"))
    input_node_names.put(5, new BytePointer("F"))
    input_node_names.put(6, new BytePointer("G"))
    input_node_names.put(7, new BytePointer("H"))
    input_node_names.put(8, new BytePointer("I"))

    val inputArr = Array(
      inputs._1.asInstanceOf[Option[Tensor[_]]],
      inputs._2.asInstanceOf[Option[Tensor[_]]],
      inputs._3.asInstanceOf[Option[Tensor[_]]],
      inputs._4.asInstanceOf[Option[Tensor[_]]],
      inputs._5.asInstanceOf[Option[Tensor[_]]],
      inputs._6.asInstanceOf[Option[Tensor[_]]],
      inputs._7.asInstanceOf[Option[Tensor[_]]],
      inputs._8.asInstanceOf[Option[Tensor[_]]],
      inputs._9.asInstanceOf[Option[Tensor[_]]]
    ).flatten

    val inputDimsAndValues: Array[Value] =
      inputArr.map(x => getTensor(x))

    /*
      (0 until 9).map{i =>


      //input_node_names.put(i,new BytePointer("A"))

      (new LongPointer(), getTensor(inputs._1))

//     (new LongPointer(), new FloatPointer())
    }.toArray
     */

//    println(inputDims.size)
    output_node_names.put(0L, new BytePointer("outName"))

    //println(tens._2(0))
    val output = runModel(
      sess,
      inputDimsAndValues,
      input_node_names,
      output_node_names
    )

    output.asInstanceOf[T9]
  }

  def callOp[
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
      name: String,
      opName: String,
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
      //    outName: String,
      attrs: Map[String, Any]
  ): (T9) = {
    val bytes = opToONNXBytes(name, opName, inputs, "outName", attrs)
    callByteArrayOp[T, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17](
      bytes,
      inputs
    )
  }

  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
