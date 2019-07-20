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
import org.emergentorder.union.UnionType._
import org.bytedeco.javacpp._;
import org.bytedeco.onnx.ModelProto;
import org.bytedeco.onnx.global.onnx.ParseProtoFromBytes;
import org.bytedeco.onnx.MessageLite;
import org.bytedeco.onnx.NodeProto;
import org.bytedeco.onnx.GraphProto
import org.bytedeco.ngraph.global.ngraph.import_onnx_model
import org.bytedeco.ngraph.Backend
import org.bytedeco.ngraph.global.ngraph.f32
import org.bytedeco.ngraph.global.ngraph.f64
import org.bytedeco.ngraph.global.ngraph.i64
import org.bytedeco.ngraph.global.ngraph.i32
import org.bytedeco.onnx.global.onnx.check_model

//TODO: Extract ModelProto modifications into generic layer, then do each layer-wise
// op in a lazy fashion, while doing the same for generating a single overall ModelProto, via ModelProto.MergeFrom. 
// Use one path for speed and dynamic graph tracing at runtime (the default), the other for sanity/type/shape/AxisType/control flow checking at compile time 
//TODO: ONNX-JS backend for both JS and JVM
//TODO: ONNX Runtime backend for JVM (and Native?)
//TODO: Find and squash memory leaks
class NGraphBackend(onnxHelper: ONNXHelper)
    extends Add
    with DataSource
    with Constant
    with ArgMin
    with ArgMax
    with Equal
    with GlobalAveragePool
    with Log
    with Softmax
    with Max
    with Min
    with Less
    with Greater
    with Abs
    with Conv
    with Sigmoid
    with Gemm
    with Gather
    with Mul
    with Relu
    with MaxPool
    with Concat
    with Dropout
    with AveragePool
    with Reshape {
//with DataSource
val ngraphBackend = Backend.create("CPU")

  def paramsMap[T: spire.math.Numeric: ClassTag] =
    onnxHelper.params
      .map(x => x._1 -> (x._2, x._3.asInstanceOf[Array[T]], x._4))
      .toMap

  override def getParams[T: Numeric: ClassTag](name: String)(
      implicit ev: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): Tensor[T] = {
    val params = paramsMap.get(name)
    params match {
      case Some(x) => TensorFactory.getTensor(x._2, x._3.map(z => z: XInt)) 
      case None =>
        throw new Exception("No params found for param name: " + name)
    }
  }

  def Abs1[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[T]) = ???

  def Abs6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[T]) = {
    (trinaryOpNoAttrs(name, "Abs", X, None, None))
  }

  def Add1[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      consumed_inputs: Option[(Array[Int])] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[T]) = ???

  def Add6[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[T]) = ???

  def Add7[@sp T: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[T]) = {
    (trinaryOpNoAttrs(name, "Add", A, B, None))
  }

  def ArgMax1[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      keepdims: Option[(Int)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[Long]) = {
    val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
    // (trinaryOp(name, "ArgMax", data, None, None, map))
    ???
  }

  def ArgMin1[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      keepdims: Option[(Int)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[Long]) = {
    val map: Map[String, Any] = Map("axis" -> axis, "keepdims" -> keepdims)
    //(trinaryOp(name, "ArgMin", data, None, None, map))
    ???
  }

  def Constant1[@sp T: Numeric: ClassTag](
      name: String,
      value: Option[(Tensor[T])]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): (Tensor[T]) = ???

  def Constant9[@sp T: Numeric: ClassTag](
      name: String,
      value: Option[(Tensor[T])]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): (Tensor[T]) = {
    (trinaryOpNoAttrs(name, "Constant", value, None, None))
  }

  def Equal1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T1]) = ???

  def Equal7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Boolean TypeOr Int TypeOr Long)#check[T],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T1]) = {
    //(trinaryOpNoAttrs(name, "Equal", A, B, None))
    ???
  }

  def GlobalAveragePool1[@sp T: Numeric: ClassTag](
      name: String,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {
    (trinaryOpNoAttrs(name, "GlobalAveragePool", X, None, None))
  }

  def Greater1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T1]) = ???

  def Greater7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T1]) = ???

  def Greater9[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T1]) = {
    //(trinaryOpNoAttrs(name, "Greater", A, B, None))
    ???
  }

  def Less1[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T1]) = ???

  def Less7[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T1]) = ???

  def Less9[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T1]) = {
    //(trinaryOpNoAttrs(name, "Less", A, B, None))
    ???
  }

  def Log1[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])] = None,
      input: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = ???

  def Log6[@sp T: Numeric: ClassTag](name: String, input: Option[Tensor[T]])(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {
    (trinaryOpNoAttrs(name, "Log", input, None, None))
  }

  def Max6[@sp T: Numeric: ClassTag](
      name: String,
      data_0: Seq[Option[Tensor[T]]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = ???

  def Max8[@sp T: Numeric: ClassTag](
      name: String,
      data_0: Seq[Option[Tensor[T]]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {
    ???
    //(trinaryOpNoAttrs(name, "Max", data_0, None, None))
  }

  def Min6[@sp T: Numeric: ClassTag](
      name: String,
      data_0: Seq[Option[Tensor[T]]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = ???

  def Min8[@sp T: Numeric: ClassTag](
      name: String,
      data_0: Seq[Option[Tensor[T]]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {
    //(trinaryOpNoAttrs(name, "Min", data_0, None, None))
    ???
  }

  def Softmax1[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      input: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {
    val map: Map[String, Any] = Map("axis" -> axis)
    (trinaryOp(name, "Softmax", input, None, None, map))
  }

  def Mul1[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      consumed_inputs: Option[(Array[Int])] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[T]) = ???

  def Mul6[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[T]) = ???

  def Mul7[@sp T: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): (Tensor[T]) = {
    (trinaryOpNoAttrs(name, "Mul", A, B, None))
  }

  def Sigmoid1[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = ???

  def Sigmoid6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {
    (trinaryOpNoAttrs(name, "Sigmoid", X, None, None))
  }
  def Gemm1[@sp T: Numeric: ClassTag](
      name: String,
      alpha: Option[(Float)] = None,
      beta: Option[(Float)] = None,
      broadcast: Option[(Int)] = None,
      transA: Option[(Int)] = None,
      transB: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]],
      C: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
        T
      ]
  ): (Tensor[T]) = ???

  def Gemm6[@sp T: Numeric: ClassTag](
      name: String,
      alpha: Option[(Float)] = None,
      beta: Option[(Float)] = None,
      broadcast: Option[(Int)] = None,
      transA: Option[(Int)] = None,
      transB: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]],
      C: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
        T
      ]
  ): (Tensor[T]) = ???

  def Gemm7[@sp T: Numeric: ClassTag](
      name: String,
      alpha: Option[(Float)] = None,
      beta: Option[(Float)] = None,
      transA: Option[(Int)] = None,
      transB: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]],
      C: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
        T
      ]
  ): (Tensor[T]) = ???

  def Gemm9[@sp T: Numeric: ClassTag](
      name: String,
      alpha: Option[(Float)] = None,
      beta: Option[(Float)] = None,
      transA: Option[(Int)] = None,
      transB: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]],
      C: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
        T
      ]
  ): (Tensor[T]) = {
    val map: Map[String, Any] = Map(
      "alpha"  -> alpha,
      "beta"   -> beta,
      "transA" -> transA,
      "transB" -> transB
    )
    (trinaryOp(name, "Gemm", A, B, C, map))
  }

  def Gather1[@sp T: Numeric: ClassTag, @sp Tind: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)],
      data: Option[Tensor[T]],
      indices: Option[Tensor[Tind]]
  )(
      implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T],
      evTind: (UNil TypeOr Int TypeOr Long)#check[Tind]
  ): (Tensor[T]) = {
    (trinaryOpNoAttrs(name, "Gather", data, indices, None: Option[Tensor[T]]))
  }

  def Conv1[@sp T: Numeric: ClassTag](
      name: String,
      auto_pad: Option[(String)] = None,
      dilations: Option[(Array[Int])] = None,
      group: Option[(Int)] = None,
      kernel_shape: Option[(Array[Int])] = None,
      pads: Option[(Array[Int])] = None,
      strides: Option[(Array[Int])] = None,
      X: Option[Tensor[T]],
      W: Option[Tensor[T]],
      B: Option[Tensor[T]] = None
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {

    val map: Map[String, Any] = Map(
      "auto_pad"     -> auto_pad,
      "dilations"    -> dilations,
      "group"        -> group,
      "kernel_shape" -> kernel_shape,
      "pads"         -> pads,
      "strides"      -> strides
    )

    trinaryOp(name, "Conv", X, W, B, map)
  }

  /*
    val inputs: Seq[String] = node.input
          assert (inputs.size == 2 || inputs.size == 3, s"number of inputs of a conv node should always be 2 or 3, got ${inputs.size}")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of output of a conv node should always be 1")

          val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
          convNode(inputs, outputs.head, getConvMaxPAvPAttr(attributes))
   */

  def Dropout7[@sp T: Numeric: ClassTag](
      name: String,
      ratio: Option[(Float)],
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T], Tensor[T]) = {
    val map: Map[String, Any] = Map("ratio" -> ratio)
    (trinaryOp(name, "Dropout", data, None, None, map), null) //TODO: optional output
  }

  def Dropout10[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): (Tensor[T], Tensor[T1]) = ???

  def Relu1[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])] // = None //Default args don't work
      ,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = ???

  def Relu6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {
    trinaryOpNoAttrs(name, "Relu", X, None, None)
  }

  def trinaryOpNoAttrs[@sp T: ClassTag, T1: ClassTag, T2: ClassTag](
      name: String,
      opName: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T1]],
      C: Option[Tensor[T2]]
  ): (Tensor[T]) =
    trinaryOp(name, opName, A, B, C, Map())

  def trinaryOpNode[@sp T: ClassTag, T1: ClassTag, T2: ClassTag](
      name: String,
      opName: String,
      A: Option[Tensor[T]],
      aName: String,
      B: Option[Tensor[T1]],
      bName: String,
      C: Option[Tensor[T2]],
      cName: String,
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
    def addInput[A](input: Option[Tensor[A]], inputName: String) {

      input match {
        case Some(tens) => {
          node.add_input(inputName)
        }
        case None =>
      }

    }


    addInput(A, aName)
    addInput(B, bName)
    addInput(C, cName)

    handleAttrs

    return node
      }

  def addInputToGraph[A](input: Option[Tensor[A]], inputName: String, graph: GraphProto) {

      input match {
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
          dims.rawShape.foreach { x =>
            val inputDim = inputValueInfo.`type`.tensor_type.shape.add_dim

//              inputDim.set_dim_param("NAME?")
            inputDim.set_dim_value(x)

          }
        }
        case None =>
      }

    }


      
  def trinaryOpModel[@sp T: ClassTag, T1: ClassTag, T2: ClassTag](
      name: String,
      opName: String,
      A: Option[Tensor[T]],
      aName: String,
      B: Option[Tensor[T1]],
      bName: String,
      C: Option[Tensor[T2]],
      cName: String,
      outName: String,
      attrs: Map[String, Any]
    ): (ModelProto) = {
//TODO: Refactor op method sigs to return this
  

    val model = (new ModelProto).New()
    val graph = new org.bytedeco.onnx.GraphProto  
    model.set_producer_name("ONNX-Scala")
    graph.set_name(name)
 
    //TODO: pass real names
    val origNode = trinaryOpNode(name, opName, A, aName, B, bName, C, cName, outName, attrs)

    val node  = graph.add_node
    node.MergeFrom(origNode)

      
    model.set_allocated_graph(graph)
    model.set_ir_version(3)

    model.add_opset_import
    model.opset_import(0).set_version(8) 

    val outputValueInfo = graph.add_output

    outputValueInfo.set_name(outName)

    outputValueInfo.mutable_type
    outputValueInfo.`type`.mutable_tensor_type
    outputValueInfo.`type`.tensor_type.set_elem_type(1)

    addInputToGraph(A, aName, graph)
    addInputToGraph(B, bName, graph)
    addInputToGraph(C, cName, graph)

    //TODO: ensure the outer model is the last merged 
    (model)
    }
   

  def trinaryOp[@sp T: ClassTag, T1: ClassTag, T2: ClassTag, T3: ClassTag](
      name: String,
      opName: String,
      A: Option[Tensor[T]],
  //    aName: String,
      B: Option[Tensor[T1]],
  //    bName: String,
      C: Option[Tensor[T2]],
  //    cName: String,
  //    outName: String,
      attrs: Map[String, Any]
    ) = {
    val opModel = trinaryOpModel(name, opName, A, "aName", B, "bName", C, "cName", "outName", attrs)
    opFromModel[T, T1, T2, T3](opModel, A, B, C)
    }

    def tensorToPointerAndType[T: ClassTag](tens: Option[Tensor[T]]): (Pointer, org.bytedeco.ngraph.Type) = tens.map {
      case (x, y) => (x, y.rawShape)
    } match {
      case Some((data: Array[Int], shape: Array[XInt])) => {

        (new IntPointer(data.asInstanceOf[Array[Int]]: _*), i32)
      }

      case Some((data: Array[Long], shape: Array[XInt])) => {

        (new LongPointer(data.asInstanceOf[Array[Long]]: _*), i64)
      }

      case Some((data: Array[Float], shape: Array[XInt])) => {

        (new FloatPointer(data.asInstanceOf[Array[Float]]: _*), f32)
      }
      case None => (new IntPointer, f32)
    }

    
    def tensorToInputShape[T: ClassTag](tens: Option[Tensor[T]]): org.bytedeco.ngraph.Shape = tens match {
      case Some(tens) => {
        val dims = tens._2
        val s    = new org.bytedeco.ngraph.Shape(tens._2.rawShape.size)
        s.resize(tens._2.rawShape.size)
        val longShape = tens._2.rawShape.map { x =>
          x.toLong
        }
        s.put(longShape: _*)
        s
      }
      case None => new org.bytedeco.ngraph.Shape

    }

  def tensorVectorToOutputTensor[T3: ClassTag](tensVec: org.bytedeco.ngraph.TensorVector, outputShape: org.bytedeco.ngraph.Shape) = {
    val arraySize = (0 until outputShape.size.toInt)
      .map { x =>
        outputShape.get(x).toInt
      }
      .reduceLeft(_ * _)

    val fp = new FloatPointer(arraySize)
    tensVec.get(0).read(fp, 0, arraySize * 4)

    val fb = fp.asByteBuffer.asFloatBuffer
    val fa = new Array[T3](arraySize.toInt)
    (0 until fb.capacity).map { x =>
      fa.update(x, fb.get(x).asInstanceOf[T3]) //unsafe : asInstanceOf
    }

    val shapeArray = new Array[Int](outputShape.size.toInt)
    (0 until outputShape.size.toInt).map { x =>
      shapeArray.update(x, outputShape.get(x).toInt)
    }

    val result: Tensor[T3]= TensorFactory.getTensor(fa, shapeArray.map(z => z: XInt))

    (result)
  }

  def opFromModel[@sp T: ClassTag, T1: ClassTag, T2: ClassTag, T3: ClassTag](
    opModel: ModelProto,
    A: Option[Tensor[T]],
    B: Option[Tensor[T1]],
    C: Option[Tensor[T2]]): (Tensor[T3]) = {
    val modelString = opModel.SerializeAsString

    //println(modelString.getString)
    //TODO: Pull this as far forward as possible
    val ngraphFunc = import_onnx_model(modelString)

    val inputShape: org.bytedeco.ngraph.Shape = tensorToInputShape(A)
    val secondInputShape: org.bytedeco.ngraph.Shape = tensorToInputShape(B)
    val thirdInputShape: org.bytedeco.ngraph.Shape = tensorToInputShape(C)

    val outputShape = ngraphFunc.get_output_shape(0)
    val outputType = ngraphFunc.get_output_element_type(0)
    val inputTens: (Pointer, org.bytedeco.ngraph.Type) = tensorToPointerAndType(A)

    val secondInputTens: (Pointer, org.bytedeco.ngraph.Type) = tensorToPointerAndType(B)

    val thirdInputTens: (Pointer, org.bytedeco.ngraph.Type) = tensorToPointerAndType(C) 

    val input  = ngraphBackend.create_tensor(inputTens._2, inputShape, inputTens._1)
    val output = ngraphBackend.create_tensor(outputType, outputShape)

    val inputVector = B match {
      case Some(_) => {
        val tens2 = ngraphBackend.create_tensor(
          secondInputTens._2,
          secondInputShape,
          secondInputTens._1
        )
        C match {
          case Some(_) =>
            new org.bytedeco.ngraph.TensorVector(
              input,
              tens2,
              ngraphBackend.create_tensor(thirdInputTens._2, thirdInputShape, thirdInputTens._1)
            )
          case None => new org.bytedeco.ngraph.TensorVector(input, tens2)
        }
      }
      case None => new org.bytedeco.ngraph.TensorVector(input)
    }

    val outputVector = new org.bytedeco.ngraph.TensorVector(output)

    //println(outputShape)
    //println(ngraphFunc)
    //println(ngraphFunc.get_output_shape(0))
    val executable = ngraphBackend.compile(ngraphFunc)

    def t = {
      val before = System.nanoTime
      executable.call(outputVector, inputVector)
      val after = System.nanoTime
 
      println("Elapsed per Op: " + "  : " + (after - before))
    }


    t

    //convert result to onnx-scala Tensor


    (tensorVectorToOutputTensor(outputVector, outputShape))
  }

 
  def fullModel[@sp T: ClassTag, T1: ClassTag, T2: ClassTag, T3: ClassTag](
      A: Option[Tensor[T]],
      B: Option[Tensor[T1]],
      C: Option[Tensor[T2]]): (Tensor[T3]) = {
      val opModel = onnxHelper.model
      println(opModel.graph.input(0).name.getString)
      opFromModel[T, T1, T2, T3](opModel, A, B, C)
    }

  def MaxPool1[@sp T: Numeric: ClassTag](
      name: String,
      auto_pad: Option[(String)] = None,
      kernel_shape: Option[(Array[Int])],
      pads: Option[(Array[Int])] = None,
      strides: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = ???

  def MaxPool8[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
      name: String,
      auto_pad: Option[(String)] = None,
      kernel_shape: Option[(Array[Int])],
      pads: Option[(Array[Int])] = None,
      storage_order: Option[(Int)] = None,
      strides: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
      evI: (UNil TypeOr Long)#check[I]
  ): (Tensor[T], Tensor[I]) = {
    val map: Map[String, Any] = Map(
      "auto_pad"      -> auto_pad,
      "kernel_shape"  -> kernel_shape,
      "pads"          -> pads,
      "storage_order" -> storage_order,
      "strides"       -> strides
    )

    (trinaryOp[T, T, T, T](name, "MaxPool", X, None, None, map), null) //TODO:optional output
  }

  def MaxPool10[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
      name: String,
      auto_pad: Option[(String)] = None,
      ceil_mode: Option[(Int)] = None,
      dilations: Option[(Array[Int])] = None,
      kernel_shape: Option[(Array[Int])],
      pads: Option[(Array[Int])] = None,
      storage_order: Option[(Int)] = None,
      strides: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
      evI: (UNil TypeOr Long)#check[I]
  ): (Tensor[T], Tensor[I]) = ???

  def Concat4[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)],
      inputs: Seq[Option[Tensor[T]]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): (Tensor[T]) = {
    val map: Map[String, Any] = Map("axis" -> axis)
    val X                     = inputs(0)
    val Y                     = inputs(1)
    val Z                     = if (inputs.size > 2) inputs(2) else None
    trinaryOp(name, "Concat", X, Y, Z, map)
    //TODO: > 3 inputs
  }

  def Dropout1[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])] = None,
      is_test: Option[(Int)] = None,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T], Tensor[T]) = ???

  def Dropout6[@sp T: Numeric: ClassTag](
      name: String,
      is_test: Option[(Int)] = None,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T], Tensor[T]) = ???

  def AveragePool1[@sp T: Numeric: ClassTag](
      name: String,
      auto_pad: Option[(String)] = None,
      kernel_shape: Option[(Array[Int])],
      pads: Option[(Array[Int])] = None,
      strides: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = ???

  def AveragePool7[@sp T: Numeric: ClassTag](
      name: String,
      auto_pad: Option[(String)] = None,
      count_include_pad: Option[(Int)] = None,
      kernel_shape: Option[(Array[Int])],
      pads: Option[(Array[Int])] = None,
      strides: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = {
    val map: Map[String, Any] = Map(
      "auto_pad"          -> auto_pad,
      "count_include_pad" -> count_include_pad,
      "kernel_shape"      -> kernel_shape,
      "pads"              -> pads,
      "strides"           -> strides
    )

    trinaryOp(name, "AveragePool", X, None, None, map)
  }

  def AveragePool10[@sp T: Numeric: ClassTag](
      name: String,
      auto_pad: Option[(String)] = None,
      ceil_mode: Option[(Int)] = None,
      count_include_pad: Option[(Int)] = None,
      kernel_shape: Option[(Array[Int])],
      pads: Option[(Array[Int])] = None,
      strides: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): (Tensor[T]) = ???

  def Reshape1[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])] = None,
      shape: Option[(Array[Int])] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): (Tensor[T]) = ???

  def Reshape5[@sp T: Numeric: ClassTag](
      name: String,
      data: Option[Tensor[T]],
      shape: Option[Tensor[Long]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): (Tensor[T]) = {
    val map: Map[String, Any] = Map("shape" -> shape)
    trinaryOp(name, "Reshape", data, None, None, map)
  }

}
