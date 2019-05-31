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

import org.bytedeco.javacpp._
import org.bytedeco.onnx._
import org.bytedeco.onnx.global.onnx._

class ONNXHelper(modelFileName: String) {



  //TODO: Add the rest of the types
  type ValidTensorProtoTypes = Array[Float]

  val byteArray = Files.readAllBytes(Paths.get(modelFileName))

  val res = new ModelProto()
  ParseProtoFromBytes(res.asInstanceOf[MessageLite],
                      new BytePointer(byteArray: _*),
                      byteArray.length.toLong)
  val graph = res.graph

  def maxOpsetVersion =
    try {
      res.opset_import(0).version
    } catch {
      case e: Exception => { 1 }
    }

  def dimsToArray[VV: spire.math.Numeric: ClassTag](
      dimsCount: Int,
      dimsList: List[Long]): Array[VV] = {
    val dimsArrayInt = dimsList.map(x => x.toInt).toArray
    val arrX = dimsCount match {
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
              dimsArrayInt(3))
      case 5 =>
        Array
          .ofDim[VV](
            dimsArrayInt(0) *
              dimsArrayInt(1) *
              dimsArrayInt(2) *
              dimsArrayInt(3) *
              dimsArrayInt(4))
    }
    arrX
  }

  def onnxTensorProtoToArray(
      tensorProto: TensorProto): ValidTensorProtoTypes = {

    val onnxDataType = tensorProto.data_type
    val dimsCount = tensorProto.dims_size
    val dimsList =
      (0 until dimsCount.toInt).map(x => tensorProto.dims(x)).toList

    val bytesBuffer = tensorProto.raw_data.asByteBuffer

    val TensProtoInt = TensorProto.INT32

    val TensProtoFloat = TensorProto.FLOAT

    val array: ValidTensorProtoTypes = onnxDataType match {
//      case TensProtoInt => {
//        val arrX = dimsToArray[Int](dimsCount, dimsList)
//        bytesBuffer.asIntBuffer.get(arrX)
//        arrX.toArray
//        //arrX.map(x => x.asInstanceOf[VV])
//      }
      case TensProtoFloat => {
        val arrX = dimsToArray[Float](dimsCount, dimsList)
        bytesBuffer.asFloatBuffer.get(arrX)
        arrX.toArray
      }
    }
    array
  }

  val nodeCount = graph.node_size.toInt
  val node = (0 until nodeCount).map(x => graph.node(x)).toList

  def attributes =
    node.map { x =>
      val attributeCount = x.attribute_size.toInt
      val attribute = (0 until attributeCount).map(y => x.attribute(y)).toArray
      attribute
    }.toArray

  def ops = node.map(x => x.op_type.getString).toArray

  def graphInputs = {
    val inputCount = graph.input_size.toInt
    val input = (0 until inputCount).map(y => graph.input(y)).toList
    input.toArray.map( y =>
        (y.name.getString
          .asInstanceOf[String]
          .replaceAll("-", "_")
          .replaceAll("/", "_"), tensorElemTypeMap(y.`type`.tensor_type.elem_type)))
          .filter(z => !(params exists (_._1.equals(z._1))))
  }
 

  def graphOutputs = {
    val outputCount = graph.output_size.toInt
    val output = (0 until outputCount).map(y => graph.output(y)).toList
    output.toArray.map( y =>
        (y.name.getString
          .asInstanceOf[String]
          .replaceAll("-", "_")
          .replaceAll("/", "_"), tensorElemTypeMap(y.`type`.tensor_type.elem_type)))
          .filter(z => !(params exists (_._1.equals(z._1))))
  }


  val tensorElemTypeMap = Map(org.bytedeco.onnx.TensorProto.UNDEFINED ->"Undefined",
                      org.bytedeco.onnx.TensorProto.FLOAT -> "Float",
                      org.bytedeco.onnx.TensorProto.UINT8 -> "UByte",
                      org.bytedeco.onnx.TensorProto.INT8 -> "Byte",
                      org.bytedeco.onnx.TensorProto.UINT16 -> "UShort",
                      org.bytedeco.onnx.TensorProto.INT16 -> "Short",
                      org.bytedeco.onnx.TensorProto.INT32 -> "Int",
                      org.bytedeco.onnx.TensorProto.INT64 -> "Long",
                      org.bytedeco.onnx.TensorProto.STRING -> "String",
                      org.bytedeco.onnx.TensorProto.BOOL -> "Boolean",
                      org.bytedeco.onnx.TensorProto.FLOAT16 -> "Float16",

                      org.bytedeco.onnx.TensorProto.DOUBLE -> "Double",

                      org.bytedeco.onnx.TensorProto.UINT32 -> "UInt",
                      org.bytedeco.onnx.TensorProto.UINT64 -> "ULong",
                      org.bytedeco.onnx.TensorProto.COMPLEX64 -> "Complex[Float]",
                      org.bytedeco.onnx.TensorProto.COMPLEX128 -> "Complex[Double]",
                      org.bytedeco.onnx.TensorProto.BFLOAT16 -> "???")

  def nodeInputs =
    node
      .map { x =>
        val inputCount = x.input_size.toInt
        val input = (0 until inputCount).map(y => x.input(y)).toList

        input
      }
      .toArray
      .map { x =>
        x.toArray
          .map(
            y =>
              y.getString
                .asInstanceOf[String]
                .replaceAll("-", "_")
                .replaceAll("/", "_"))
      }

  def nodeOutputs =
    node
      .map { x =>
        val outputCount = x.output_size.toInt
        val output = (0 until outputCount).map(y => x.output(y)).toList

        output
      }
      .toArray
      .map { x =>
        x.toArray.map(
          y =>
            y.getString
              .asInstanceOf[String]
              .replaceAll("-", "_")
              .replaceAll("/", "_"))
      }

  val globalOutputCount = graph.output_size.toInt
  val globalOutput =
    (0 until globalOutputCount).map(x => graph.output(x)).toList

  def outputs = {
    val outputArray = globalOutput.toArray
    outputArray
      .map(x => x.name.getString.replaceAll("-", "_").replaceAll("/", "_"))
      .filter(x => nodes.contains("output_" + x))
  }

  val inputCount = graph.input_size.toInt
  val input = (0 until inputCount).map(x => graph.input(x)).toList

  def nodes = {
    val someNodes = input.map { x =>
      val name = x.name.getString
      if (params exists (_._1.equals(name)))
        ("param_" + name) 
      else ("input_" + name)
    } ++ nodeOutputs.flatten.map(y => ("output_" + y)) 
    someNodes
  }

  val initializerCount = graph.initializer_size
  val initializer =
    (0 until initializerCount).map(x => graph.initializer(x)).toList

  val params =
    initializer.map { x =>
      val dimsCount = x.dims_size
      val dimsList = (0 until dimsCount.toInt).map(y => x.dims(y)).toList
      def arrX: ValidTensorProtoTypes = onnxTensorProtoToArray(x)
      val tensorElemType = tensorElemTypeMap(x.data_type)
      (x.name.getString.replaceAll("-", "_").replaceAll("/", "_"), tensorElemType, arrX, dimsList.map(y => y.toInt).toArray)
    }

  def paramsMap[T : spire.math.Numeric : ClassTag]  = params.map( x => x._1 -> (x._2, x._3.asInstanceOf[Array[T]], x._4)).toMap
}
