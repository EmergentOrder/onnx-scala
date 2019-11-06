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

trait NGraphBackendUtils extends AutoCloseable {

  private val scope = new PointerScope()

  protected def opToNode[
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
            case None    =>
          }
        }
        case _ => ??? //TODO: Handle non-tensors / don't assume tensor here

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

  protected def addInputToGraph[A](input: A, inputName: String, graph: GraphProto): Unit = {

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

  protected def tensorToPointerAndType[T: ClassTag](
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

  protected def tensorToInputShape[T: ClassTag](tens: Tensor[T]): org.bytedeco.ngraph.Shape = {
    val dims = tens._2
    val s    = new org.bytedeco.ngraph.Shape(tens._2.size)
    s.resize(tens._2.size)
    val longShape = tens._2.map { x =>
      x.toLong
    }
    s.put(longShape: _*)
    s
  }

  protected def tensorVectorToOutputTensor[T: ClassTag](
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

      case `i8` => {

//        assert(elemType.equals(ngraph.i8().get_type_enum()))
        val fp = new BytePointer(arraySize)
        tens.read(fp, arraySize * 1)

        val fb = fp.asByteBuffer

        val res = (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Byte] //unsafe : asInstanceOf
        }.toArray
        fp.close
        res
      }

      case `i16` => {

//        assert(elemType.equals(ngraph.i16().get_type_enum()))
        val fp = new ShortPointer(arraySize)
        tens.read(fp, arraySize * 2)

        val fb = fp.asByteBuffer.asShortBuffer

        val res = (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Short] //unsafe : asInstanceOf
        }.toArray

        fp.close
        res
      }
      case `i32` => {

//        assert(elemType.equals(ngraph.i32().get_type_enum()))
        val fp = new IntPointer(arraySize)
        tens.read(fp, arraySize * 4)

        val fb = fp.asByteBuffer.asIntBuffer

        val res = (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Int] //unsafe : asInstanceOf
        }.toArray
        fp.close
        res
      }
      case `i64` => {

//        assert(elemType.equals(ngraph.i64().get_type_enum()))
        val fp = new LongPointer(arraySize)
        tens.read(fp, arraySize * 8)

        val fb = fp.asByteBuffer.asLongBuffer

        val res = (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Long] //unsafe : asInstanceOf
        }.toArray
        fp.close
        res

      }
      case `f32` => {

        // assert(elemType.equals(ngraph.f32().get_type_enum()))
        val fp = new FloatPointer(arraySize)
        tens.read(fp, arraySize * 4)

        val fb = fp.asByteBuffer.asFloatBuffer

        val res = (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Float] //unsafe : asInstanceOf
        }.toArray
        fp.close
        res
      }
      case `f64` => {

        //assert(elemType.equals(ngraph.f64().get_type_enum()))
        val fp = new DoublePointer(arraySize)
        tens.read(fp, arraySize * 8)

        val fb = fp.asByteBuffer.asDoubleBuffer

        val res = (0 until fb.capacity).map { x =>
          fb.get(x).asInstanceOf[Double] //unsafe : asInstanceOf
        }.toArray
        fp.close
        res
      }
    }

    val shapeArray = (0 until outputShape.size.toInt).map { x =>
      outputShape.get(x).toInt
    }.toArray

    val result = TensorFactory.getTensor(fa, shapeArray).asInstanceOf[T]
    tensVec.close
    outputShape.close
    (result)
  }

  protected def getTensorShape[T: ClassTag](t: T): Option[org.bytedeco.ngraph.Shape] = {
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

  protected def getTensorPointerAndType[T: ClassTag](
      t: T
  ): Option[(Pointer, org.bytedeco.ngraph.Type)] = {

    t match {
      case tensorOpt: Option[Tensor[Any]] => {
        tensorOpt match {
          case Some(y: Tensor[Any]) => Some(tensorToPointerAndType(y))
          case None                 => None
        }
      }
    }
  }

  override def close(): Unit = {
    scope.close
  }
}
