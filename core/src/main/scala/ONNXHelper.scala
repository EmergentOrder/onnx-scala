/*
 * ONNXHelper
 * Copyright (c) 2018 Alexander Merritt
 * All rights reserved.
 * This program is free software: you can redistribute it and/or modify
 *
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.emergentorder.onnx

import java.nio.file._
import java.nio.ByteBuffer
import collection.JavaConverters._
import scala.reflect.ClassTag

import java.io.File
import scala.reflect.io.Streamable
import org.bytedeco.javacpp._
import org.bytedeco.onnx._
import org.bytedeco.onnx.global.onnx.ParseProtoFromBytes

class ONNXHelper(val byteArray: Array[Byte]) extends AutoCloseable {

//  val scope = new PointerScope()
//  val loaded =
//    org.bytedeco.javacpp.Loader.load(classOf[org.bytedeco.onnx.global.onnx])

  // println(Pointer.physicalBytes)
  lazy val model = {
    val r     = (new ModelProto).New()
    val bytes = new BytePointer(byteArray: _*)
    ParseProtoFromBytes(
      r,
      bytes,
      byteArray.length.toLong
    )
//    bytes.close
    r
  }

  private val graph = model.graph

  val maxOpsetVersion =
    try {
      model.opset_import(0).version
    } catch {
      case e: Exception => { 1 }
    }

  private def dimsToArray[VV: spire.math.Numeric: ClassTag](
      dimsCount: Int,
      dimsList: List[Long]
  ): Array[VV] = {
    val dimsArrayInt = dimsList.map(x => x.toInt).toArray
    val arrX = dimsCount match {
      case 0 => Array[VV]()
      case 1 => Array.ofDim[VV](dimsArrayInt(0))
      case 2 => Array.ofDim[VV](dimsArrayInt(0) * dimsArrayInt(1))
      case 3 =>
        Array
          .ofDim[VV](dimsArrayInt(0) * dimsArrayInt(1) * dimsArrayInt(2))
      case 4 =>
        Array
          .ofDim[VV](
            dimsArrayInt(0) *
              dimsArrayInt(1) *
              dimsArrayInt(2) *
              dimsArrayInt(3)
          )
      case 5 =>
        Array
          .ofDim[VV](
            dimsArrayInt(0) *
              dimsArrayInt(1) *
              dimsArrayInt(2) *
              dimsArrayInt(3) *
              dimsArrayInt(4)
          )
    }
    arrX
  }

  def onnxTensorProtoToArray(tensorProto: TensorProto) = {

    //TODEFER: Get dim and type denotations, encode into types here in 2.13 / earlier if possible
//    val scope = new PointerScope()

    val onnxDataType = tensorProto.data_type
    val dimsCount    = tensorProto.dims_size
    val dimsList =
      (0 until dimsCount.toInt).map(x => tensorProto.dims(x)).toList

    val rawData = tensorProto.raw_data

    val TensProtoByte   = TensorProto.INT8
    val TensProtoShort  = TensorProto.INT16
    val TensProtoInt    = TensorProto.INT32
    val TensProtoLong   = TensorProto.INT64
    val TensProtoFloat  = TensorProto.FLOAT
    val TensProtoDouble = TensorProto.DOUBLE

    val array = onnxDataType match {
      case TensProtoByte => {
        val arrX = dimsToArray[Byte](dimsCount, dimsList)
        (0 until arrX.length).foreach {
          if (rawData == null) { x => arrX(x) = tensorProto.int32_data(x).toByte }
          else { x => arrX(x) = rawData.get(x) }
        }
        arrX.toArray
      }
      case TensProtoShort => {
        val arrX = dimsToArray[Short](dimsCount, dimsList)
        (0 until arrX.length).foreach {
          if (rawData == null) { x => arrX(x) = tensorProto.int32_data(x).toShort }
          else { x => arrX(x) = rawData.getShort(x * 2) }
        }
        arrX.toArray
      }
      case TensProtoInt => {
        val arrX = dimsToArray[Int](dimsCount, dimsList)
        (0 until arrX.length).foreach {
          if (rawData == null) { x => arrX(x) = tensorProto.int32_data(x) }
          else { x => arrX(x) = rawData.getInt(x * 4) }
        }
        arrX.toArray
      }
      case TensProtoLong => {
        val arrX = dimsToArray[Long](dimsCount, dimsList)
        (0 until arrX.length).foreach {
          if (rawData == null) { x => arrX(x) = tensorProto.int64_data(x) }
          else { x => arrX(x) = rawData.getLong(x * 8) }
        }
        arrX.toArray
      }
      case TensProtoFloat => {
        val arrX = dimsToArray[Float](dimsCount, dimsList)
        (0 until arrX.length).foreach {
          if (rawData == null) { x => arrX(x) = tensorProto.float_data(x) }
          else { x => arrX(x) = rawData.getFloat(x * 4) }
        }
        arrX.toArray
      }
      case TensProtoDouble => {
        val arrX = dimsToArray[Double](dimsCount, dimsList)
        (0 until arrX.length).foreach {
          if (rawData == null) { x => arrX(x) = tensorProto.double_data(x) }
          else { x => arrX(x) = rawData.getDouble(x * 8) }
        }
        arrX.toArray
      }
    }
    if (rawData != null) {
//      rawData.close
    }
//    tensorProto.close
//    scope.close
    array
  }

  private val nodeCount = graph.node_size.toInt
  private val node      = (0 until nodeCount).map(x => graph.node(x)).toList

  val attributes =
    node.map { x =>
      val attributeCount = x.attribute_size.toInt
      val attribute      = (0 until attributeCount).map(y => x.attribute(y)).toArray
      attribute
    }.toArray

  val ops = node.map(x => x.op_type.getString).toArray

  private val tensorElemTypeMap = Map(
    org.bytedeco.onnx.TensorProto.UNDEFINED  -> "Undefined",
    org.bytedeco.onnx.TensorProto.FLOAT      -> "Float",
    org.bytedeco.onnx.TensorProto.UINT8      -> "UByte",
    org.bytedeco.onnx.TensorProto.INT8       -> "Byte",
    org.bytedeco.onnx.TensorProto.UINT16     -> "UShort",
    org.bytedeco.onnx.TensorProto.INT16      -> "Short",
    org.bytedeco.onnx.TensorProto.INT32      -> "Int",
    org.bytedeco.onnx.TensorProto.INT64      -> "Long",
    org.bytedeco.onnx.TensorProto.STRING     -> "String",
    org.bytedeco.onnx.TensorProto.BOOL       -> "Boolean",
    org.bytedeco.onnx.TensorProto.FLOAT16    -> "Float16",
    org.bytedeco.onnx.TensorProto.DOUBLE     -> "Double",
    org.bytedeco.onnx.TensorProto.UINT32     -> "UInt",
    org.bytedeco.onnx.TensorProto.UINT64     -> "ULong",
    org.bytedeco.onnx.TensorProto.COMPLEX64  -> "Complex[Float]",
    org.bytedeco.onnx.TensorProto.COMPLEX128 -> "Complex[Double]",
    org.bytedeco.onnx.TensorProto.BFLOAT16   -> "???"
  )

  val nodeInputs =
    node
      .map { x =>
        val inputCount = x.input_size.toInt
        val input      = (0 until inputCount).map(y => x.input(y)).toList

        input
      }
      .toArray
      .map { x =>
        x.toArray
          .map(y =>
            y.getString
              .asInstanceOf[String]
              .replaceAll("-", "_")
              .replaceAll("/", "_")
          )
      }

  val nodeOutputs =
    node
      .map { x =>
        val outputCount = x.output_size.toInt
        val output      = (0 until outputCount).map(y => x.output(y)).toList

        output
      }
      .toArray
      .map { x =>
        x.toArray.map(y =>
          y.getString
            .asInstanceOf[String]
            .replaceAll("-", "_")
            .replaceAll("/", "_")
        )
      }

  val globalOutputCount = graph.output_size.toInt
  val globalOutput =
    (0 until globalOutputCount).map(x => graph.output(x)).toList

  val inputCount = graph.input_size.toInt
  val input      = (0 until inputCount).map(x => graph.input(x)).toList

  private val initializerCount = graph.initializer_size
  private val initializer =
    (0 until initializerCount).map(x => graph.initializer(x)).toList

  lazy val params =
    initializer.map { x =>
      val dimsCount      = x.dims_size
      val dimsList       = (0 until dimsCount.toInt).map(y => x.dims(y)).toList
      val name           = x.name.getString.replaceAll("-", "_").replaceAll("/", "_")
      val tensorElemType = tensorElemTypeMap(x.data_type)
      val arrX           = onnxTensorProtoToArray(x)
      (name, tensorElemType, arrX, dimsList.map(y => y.toInt).toArray)
    }

  lazy val nodes = {
    val someNodes = input.map { x =>
      val name = x.name.getString
      if (params exists (_._1.equals(name)))
        ("param_" + name)
      else ("input_" + name)
    } ++ nodeOutputs.flatten.map(y => ("output_" + y))
    someNodes
  }

  lazy val outputs = {
    val outputArray = globalOutput.toArray
    outputArray
      .map(x => x.name.getString.replaceAll("-", "_").replaceAll("/", "_"))
      .filter(x => nodes.contains("output_" + x))
  }

  lazy val graphInputs = {
    val inputCount = graph.input_size.toInt
    val input      = (0 until inputCount).map(y => graph.input(y)).toList
    input.toArray
      .map { y =>
        (
          y.name.getString
            .asInstanceOf[String]
            .replaceAll("-", "_")
            .replaceAll("/", "_"),
          tensorElemTypeMap(y.`type`.tensor_type.elem_type)
        )
      }
      .filter(z => !(params exists (_._1.equals(z._1))))
  }

  lazy val graphOutputs = {
    val outputCount = graph.output_size.toInt
    val output      = (0 until outputCount).map(y => graph.output(y)).toList
    output.toArray
      .map(y =>
        (
          y.name.getString
            .asInstanceOf[String]
            .replaceAll("-", "_")
            .replaceAll("/", "_"),
          tensorElemTypeMap(y.`type`.tensor_type.elem_type)
        )
      )
      .filter(z => !(params exists (_._1.equals(z._1))))
  }

  override def close(): Unit = {
//    model.close
//    scope.close
  }

}
