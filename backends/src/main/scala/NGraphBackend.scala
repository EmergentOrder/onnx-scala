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

// TODO: check import org.bytedeco.onnx.global.onnx.check_model

//TODEFER: Tracing Mode: Extract ModelProto modifications into generic layer, then do each layer-wise
// op in a lazy fashion, while doing the same for generating a single overall ModelProto, via ModelProto.MergeFrom.
// Between fine and full modes:
// Use one path for speed and dynamic graph tracing at runtime (the default), the other for sanity/type/shape/AxisType/control flow checking at compile time
//TODEFER: ONNX-JS backend for both JS and JVM
//TODEFER: ONNX Runtime backend for JVM (and Native?)
class NGraphBackend(onnxBytes: Array[Byte])
    extends DataSource
    with AutoCloseable {

  val scope = new PointerScope()

  val ngraphBackend = Backend.create("CPU")

  val onnxHelper = new ONNXHelper(onnxBytes)

  def paramsMap[T: spire.math.Numeric: ClassTag] =
    onnxHelper.params
      .map(x => x._1 -> (x._2, x._3.asInstanceOf[Array[T]], x._4))
      .toMap

  override def getParams[T: Numeric: ClassTag](name: String): Tensor[T] = {
    val params = paramsMap.get(name)
    params match {
      case Some(x) => TensorFactory.getTensor(x._2, x._3.map(z => z: XInt))
      case None =>
        throw new Exception("No params found for param name: " + name)
    }
  }

  /*
    val inputs: Seq[String] = node.input
          assert (inputs.size == 2 || inputs.size == 3, s"number of inputs of a conv node should always be 2 or 3, got ${inputs.size}")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of output of a conv node should always be 1")

          val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
          convNode(inputs, outputs.head, getConvMaxPAvPAttr(attributes))
   */

  def callOpNode[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag
  ](
      name: String,
      opName: String,
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
      outName: String,
      attrs: Map[String, Any]
  )
  //(
//        implicit evT:  (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
  //       Float] TypeOr Complex[Double])#check[T])
      : NodeProto = {
    val node = (new NodeProto).New()

    node.set_name(name)
    node.set_op_type(opName)
    node.add_output(outName)

    def handleAttrs = attrs.foreach {
      case (key, value) =>
        val longVal = value.asInstanceOf[Option[Int]] match {
          case Some(x) => {
            node.add_attribute
            val attr     = node.mutable_attribute(0)
            val attrName = new BytePointer(key)
            attr.set_name(attrName)
            attr.set_type(2)
            val longVal = x.toLong
            attr.set_i(longVal)
          }
          case None =>
        }
    }

    //TODO: Don't take tensor here
    def addInput[A](input: A, inputName: String): Unit = {

      input match {
        case tensorOpt: Option[Tensor[Any]] => {
          tensorOpt match {
            case Some(y) => node.add_input(inputName)
            case None    => //TODO: Handle non-tensors / don't assume tensor here
          }
        }
        case _ => ???
      }
    }
    //TODO: fix names
    addInput(inputs._1, "A")
    addInput(inputs._2, "B")
    addInput(inputs._3, "C")
    addInput(inputs._4, "D")
    addInput(inputs._5, "E")
    addInput(inputs._6, "F")
    addInput(inputs._7, "G")
    addInput(inputs._8, "H")
    addInput(inputs._9, "I")

    handleAttrs

    return node
  }

  def addInputToGraph[A](input: A, inputName: String, graph: GraphProto): Unit = {

    input match {
      case tensorOpt: Option[Tensor[_]] => {
        tensorOpt match {
          case Some(tens) => {

            val elemType = tens._1 match {
              case f: Array[Float] => 1
              case i: Array[Int]   => 6
              case l: Array[Long]  => 7
            }

            val inputValueInfo = graph.add_input

            inputValueInfo.set_name(inputName)
            inputValueInfo.mutable_type
            inputValueInfo.`type`.mutable_tensor_type
            inputValueInfo.`type`.tensor_type.set_elem_type(elemType)

            val dims = tens._2
            inputValueInfo.`type`.tensor_type.mutable_shape
            dims.foreach { x =>
              val inputDim = inputValueInfo.`type`.tensor_type.shape.add_dim

//              inputDim.set_dim_param("NAME?")
              inputDim.set_dim_value(x)

            }
          }
          case None =>
        }

      }
    }
  }

  def callOpModel[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag
  ](
      name: String,
      opName: String,
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
      outName: String,
      attrs: Map[String, Any]
  ): (ModelProto) = {

    val model = (new ModelProto).New()
    val graph = new org.bytedeco.onnx.GraphProto
    model.set_producer_name("ONNX-Scala")
    graph.set_name(name)

    //TODO: pass real names
    val origNode = callOpNode(name, opName, inputs, outName, attrs)

    val node = graph.add_node
    node.MergeFrom(origNode)

    origNode.close
    model.set_allocated_graph(graph)
    model.set_ir_version(3)

    model.add_opset_import
    model.opset_import(0).set_version(8)

    val outputValueInfo = graph.add_output

    outputValueInfo.set_name(outName)

    outputValueInfo.mutable_type
    outputValueInfo.`type`.mutable_tensor_type
    outputValueInfo.`type`.tensor_type.set_elem_type(1)

    addInputToGraph(inputs._1, "A", graph)
    addInputToGraph(inputs._2, "B", graph)
    addInputToGraph(inputs._3, "C", graph)
    addInputToGraph(inputs._4, "D", graph)
    addInputToGraph(inputs._5, "E", graph)
    addInputToGraph(inputs._6, "F", graph)
    addInputToGraph(inputs._7, "G", graph)
    addInputToGraph(inputs._8, "H", graph)
    addInputToGraph(inputs._9, "I", graph)

    //TODEFER: Merge models, ensuring the outer model is the last merged
    (model)
  }

  def tensorToPointerAndType[T: ClassTag](
      tens: Tensor[T]
  ): (Pointer, org.bytedeco.ngraph.Type) = {
    val data = tens._1
    data match {
      case dat: Array[Byte]  => (new BytePointer(dat.asInstanceOf[Array[Byte]]: _*), ngraph.i8)
      case dat: Array[Short] => (new ShortPointer(dat.asInstanceOf[Array[Short]]: _*), ngraph.i16)
      case dat: Array[Int]   => (new IntPointer(dat.asInstanceOf[Array[Int]]: _*), ngraph.i32)
      case dat: Array[Long]  => (new LongPointer(dat.asInstanceOf[Array[Long]]: _*), ngraph.i64)
      case dat: Array[Float] => (new FloatPointer(dat.asInstanceOf[Array[Float]]: _*), ngraph.f32)
      case dat: Array[Double] =>
        (new DoublePointer(dat.asInstanceOf[Array[Double]]: _*), ngraph.f64)

    }
  }

  def tensorToInputShape[T: ClassTag](tens: Tensor[T]): org.bytedeco.ngraph.Shape = {
    val dims = tens._2
    val s    = new org.bytedeco.ngraph.Shape(tens._2.size)
    s.resize(tens._2.size)
    val longShape = tens._2.map { x =>
      x.toLong
    }
    s.put(longShape: _*)
    s
  }

  def tensorVectorToOutputTensor[T: ClassTag](
      tensVec: org.bytedeco.ngraph.TensorVector,
      outputShape: org.bytedeco.ngraph.Shape
  ): (T) = {

    val arraySize: Long = (0 until outputShape.size.toInt)
      .map { x =>
        outputShape.get(x).toInt
      }
      .reduceLeft(_ * _)

    val tens          = tensVec.get(0)
    val elemType: Int = tens.get_element_type().get_type_enum()
    val i8: Int       = ngraph.i8().get_type_enum()
    val i16: Int      = ngraph.i16().get_type_enum()
    val i32: Int      = ngraph.i32().get_type_enum()
    val i64: Int      = ngraph.i64().get_type_enum()
    val f32: Int      = ngraph.f32().get_type_enum()
    val f64: Int      = ngraph.f64().get_type_enum()
    val fa = elemType match {
//TODO: Match not working here
      case `i8` => {

//        assert(elemType.equals(ngraph.i8().get_type_enum()))
        val fp = new BytePointer(arraySize)
        tens.read(fp, arraySize * 1)

        val fb = fp.asByteBuffer

        (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Byte] //unsafe : asInstanceOf
        }.toArray

      }

      case `i16` => {

//        assert(elemType.equals(ngraph.i16().get_type_enum()))
        val fp = new ShortPointer(arraySize)
        tens.read(fp, arraySize * 2)

        val fb = fp.asByteBuffer.asShortBuffer

        (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Short] //unsafe : asInstanceOf
        }.toArray

      }
      case `i32` => {

//        assert(elemType.equals(ngraph.i32().get_type_enum()))
        val fp = new IntPointer(arraySize)
        tens.read(fp, arraySize * 4)

        val fb = fp.asByteBuffer.asIntBuffer

        (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Int] //unsafe : asInstanceOf
        }.toArray

      }
      case `i64` => {

//        assert(elemType.equals(ngraph.i64().get_type_enum()))
        val fp = new LongPointer(arraySize)
        tens.read(fp, arraySize * 8)

        val fb = fp.asByteBuffer.asLongBuffer

        (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Long] //unsafe : asInstanceOf
        }.toArray

      }
      case `f32` => {

        // assert(elemType.equals(ngraph.f32().get_type_enum()))
        val fp = new FloatPointer(arraySize)
        tens.read(fp, arraySize * 4)

        val fb = fp.asByteBuffer.asFloatBuffer

        (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Float] //unsafe : asInstanceOf
        }.toArray

      }
      case `f64` => {

        //assert(elemType.equals(ngraph.f64().get_type_enum()))
        val fp = new DoublePointer(arraySize)
        tens.read(fp, arraySize * 8)

        val fb = fp.asByteBuffer.asDoubleBuffer

        (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Double] //unsafe : asInstanceOf
        }.toArray

      }
    }

    val shapeArray = (0 until outputShape.size.toInt).map { x =>
      outputShape.get(x).toInt
    }.toArray

    val result = TensorFactory.getTensor(fa, shapeArray.map(z => z: XInt)).asInstanceOf[T]
    tensVec.close
    outputShape.close
    (result)
  }

  def opFromModel[
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
      opModel: ModelProto,
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8]
  ): (T9) = {
    //, T10, T11, T12, T13, T14, T15, T16, T17] = { //TODO: Fix output type !!
    val scope = new PointerScope()

    //println(Pointer.totalBytes)
    val modelString = opModel.SerializeAsString
    opModel.close
    val modelStringBytes = modelString.getStringBytes
    modelString.close

    //println(Pointer.totalBytes)
    val result = opFromByteArray[
      T,
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      T7,
      T8,
      T9,
      T10,
      T11,
      T12,
      T13,
      T14,
      T15,
      T16,
      T17
    ](modelStringBytes, inputs)
    scope.close
    (result.asInstanceOf[T9]) //TODO: More outputs
  }

  def getTensorShape[T: ClassTag](t: T): Option[org.bytedeco.ngraph.Shape] = {
    t match {
      case tensorOpt: Option[Tensor[Any]] => {
        tensorOpt match {
          case Some(y) => Some(tensorToInputShape(y))
          case None    => None
        }
      }
      case _ => ??? //TODO: Handle non-tensors / don't assume tensor here

    }
  }

  def getTensorPointerAndType[T: ClassTag](t: T): Option[(Pointer, org.bytedeco.ngraph.Type)] = {

    t match {
      case tensorOpt: Option[Tensor[Any]] => {
        tensorOpt match {
          case Some(y: Tensor[Any]) => Some(tensorToPointerAndType(y))
          case None                 => None
        }
      }
    }
  }

  def opFromByteArray[
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
    val scope       = new PointerScope()
    val modelString = new BytePointer(opModel: _*)

    val ngraphFunc = import_onnx_model(modelString)
    modelString.close

    val inputShapes = Seq(
      getTensorShape(inputs._1),
      getTensorShape(inputs._2),
      getTensorShape(inputs._3),
      getTensorShape(inputs._4),
      getTensorShape(inputs._5),
      getTensorShape(inputs._6),
      getTensorShape(inputs._7),
      getTensorShape(inputs._8),
      getTensorShape(inputs._9)
    ).flatten

    val outputShape = ngraphFunc.get_output_shape(0)
    val outputType  = ngraphFunc.get_output_element_type(0)

    val inputTensors = Seq(
      getTensorPointerAndType(inputs._1),
      getTensorPointerAndType(inputs._2),
      getTensorPointerAndType(inputs._3),
      getTensorPointerAndType(inputs._4),
      getTensorPointerAndType(inputs._5),
      getTensorPointerAndType(inputs._6),
      getTensorPointerAndType(inputs._7),
      getTensorPointerAndType(inputs._8),
      getTensorPointerAndType(inputs._9)
    ).flatten

    val ngraphInputs =
      (inputShapes zip inputTensors).map(x => ngraphBackend.create_tensor(x._2._2, x._1, x._2._1))

    val output = ngraphBackend.create_tensor(outputType, outputShape)

    val inputVector = new org.bytedeco.ngraph.TensorVector(ngraphInputs: _*)

    val outputVector = new org.bytedeco.ngraph.TensorVector(output)

    //println(outputShape)
    //println(ngraphFunc)
    //println(ngraphFunc.get_output_shape(0))
    val executable = ngraphBackend.compile(ngraphFunc)

    def t = {
      val before = System.nanoTime
      executable.call(outputVector, inputVector)
      val after = System.nanoTime

      executable.close
//      println("Elapsed per Op: " + "  : " + (after - before))
    }

    t

    ngraphFunc.close
    modelString.close
    executable.close
    //convert result to onnx-scala Tensor

    val result = tensorVectorToOutputTensor[T9](outputVector, outputShape)

//    inputShape.close
//    secondInputShape.close
//    thirdInputShape.close
    outputType.close
//    inputTens._1.close
//    inputTens._2.close
//    secondInputTens._1.close
//    secondInputTens._2.close
//    thirdInputTens._1.close
//    thirdInputTens._2.close

//    input.close
    inputVector.close
    output.close
    outputVector.close
    outputShape.close
    scope.close
    (result)
  }

  def fullModel[
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

    //println(Pointer.totalBytes)
//    val scope = new PointerScope()

    val byteArray = onnxBytes
    /*
    val opModel = {
    val mod = (new ModelProto)
    val r = mod.New()
    val bytes = new BytePointer(byteArray: _*)

    ParseProtoFromBytes(
      r,
      bytes,
      byteArray.length.toLong
    )
    bytes.close
    mod.close
    r
    }

    //opModel.Clear()
    println(Pointer.totalBytes)

    //FIXME: Hardcoding the output size to match input size
    val aSize = A.map(x => x._2(0)) match {
      case Some(y) => y.toLong
      case None    => 0L
    }
    opModel.graph.input(0).`type`.tensor_type.shape.dim(0).set_dim_value(aSize)
    opModel.graph.input(1).`type`.tensor_type.shape.dim(0).set_dim_value(aSize)
    opModel.graph.output(0).`type`.tensor_type.shape.dim(0).set_dim_value(aSize)

     */
    //println(opModel.graph.input(0).name.getString)
    val result = opFromByteArray[
      T,
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      T7,
      T8,
      T9,
      T10,
      T11,
      T12,
      T13,
      T14,
      T15,
      T16,
      T17
    ](byteArray, inputs)
    //opModel.close
//    scope.close
    //println(Pointer.totalBytes)
    result
  }

  override def close(): Unit = {
    ngraphBackend.close
    scope.close
  }
}
