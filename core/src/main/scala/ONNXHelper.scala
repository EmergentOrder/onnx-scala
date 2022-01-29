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
import scala.language.implicitConversions

import java.io.File
//import org.bytedeco.javacpp._
import onnx.onnx._
import onnx.onnx.TensorProto.DataType._

class ONNXHelper(val byteArray: Array[Byte]) {
   lazy val model = ModelProto.parseFrom(byteArray)
   //  check_model(model.toProtoString)
   private val graph = model.graph

   val maxOpsetVersion: Long =
      try {
         model.opsetImport(0).version.getOrElse(0L)
      } catch {
         case e: Exception => { 1 }
      }

   def onnxTensorProtoToArray(tensorProto: TensorProto) = {

      // TODEFER: Get dim and type denotations, encode into types here

      val onnxDataType = tensorProto.dataType
      val dimsCount    = tensorProto.dims.size
      val dims =
         (0 until dimsCount.toInt).map(x => tensorProto.dims(x)).toArray

      val rawData = tensorProto.rawData

      val TensProtoByte   = INT8.index
      val TensProtoShort  = INT16.index
      val TensProtoInt    = INT32.index
      val TensProtoLong   = INT64.index
      val TensProtoFloat  = FLOAT.index
      val TensProtoDouble = DOUBLE.index

      // TODO: asXBuffer then put
      val array = onnxDataType.getOrElse(1) match {
         case TensProtoByte => {
            rawData.map(x => x.toByteArray())
         }
         case TensProtoShort => {
            tensorProto.int32Data.toArray
         }
         case TensProtoInt => {
            tensorProto.int32Data.toArray
         }
         case TensProtoLong => {
            tensorProto.int64Data.toArray
         }
         case TensProtoFloat => {
            tensorProto.floatData.toArray
         }
         case TensProtoDouble => {
            tensorProto.doubleData.toArray
         }
      }

      array
   }

   private val nodeCount = graph.map(x => x.node.size.toInt).getOrElse(0)
   private val node      = (0 until nodeCount).map(x => graph.map(y => y.node(x)))

   val attributes =
      node
         .map { nodeOpt =>
            nodeOpt.map { x =>
               val attributeCount = x.attribute.size.toInt
               val attribute      = (0 until attributeCount).map(y => x.attribute(y)).toArray
               attribute
            }
         }
         .toArray
         .flatten

   val ops = node.map(x => x.map(y => y.opType).flatten).toArray

   private val tensorElemTypeMap = Map(
     UNDEFINED.index  -> "Undefined",
     FLOAT.index      -> "Float",
     UINT8.index      -> "UByte",
     INT8.index       -> "Byte",
     UINT16.index     -> "UShort",
     INT16.index      -> "Short",
     INT32.index      -> "Int",
     INT64.index      -> "Long",
     STRING.index     -> "String",
     BOOL.index       -> "Boolean",
     FLOAT16.index    -> "Float16",
     DOUBLE.index     -> "Double",
     UINT32.index     -> "UInt",
     UINT64.index     -> "ULong",
     COMPLEX64.index  -> "Complex[Float]",
     COMPLEX128.index -> "Complex[Double]",
     BFLOAT16.index   -> "???"
   )

   val nodeInputs =
      node
         .map { nodeOpt =>
            nodeOpt.map { x =>
               val inputCount = x.input.size.toInt
               val input      = (0 until inputCount).map(y => x.input(y))

               input
            }
         }
         .toIndexedSeq
         .flatten
         .map { x =>
            x.toArray
               .map(y =>
                  y
                     .replaceAll("-", "_")
                     .replaceAll("/", "_")
               )
         }

   val nodeOutputs =
      node
         .map { nodeOpt =>
            nodeOpt.map { x =>
               val outputCount = x.output.size.toInt
               val output      = (0 until outputCount).map(y => x.output(y))

               output
            }
         }
         .toIndexedSeq
         .flatten
         .map { x =>
            x.toArray.map(y =>
               y.replaceAll("-", "_")
                  .replaceAll("/", "_")
            )
         }

//  val globalOutputCount: Int = graph.map(x => x.output.size.toInt).getOrElse(0)
   val globalOutput =
      (0 until graph.map(x => x.output.size.toInt).getOrElse(0)).map(x =>
         graph.map(y => y.output(x))
      )

   val inputCount = graph.map(x => x.input.size.toInt).getOrElse(0)
   val input      = (0 until inputCount).map(x => graph.map(y => y.input(x)))

   private val initializerCount = graph.map(x => x.initializer.size).getOrElse(0)
   private val initializer =
      (0 until initializerCount).map(y => graph.map(z => z.initializer(y))).toIndexedSeq.flatten

   lazy val params =
      initializer.map { x =>
         val dimsCount = x.dims.size
         val dimsList  = (0 until dimsCount.toInt).map(y => x.dims(y))
         val name = x.name.map(y => y.replaceAll("-", "_").replaceAll("/", "_")).getOrElse("none")
         val tensorElemType = x.dataType.map(y => tensorElemTypeMap(y)).getOrElse("none")
         val arrX           = onnxTensorProtoToArray(x)
         (name, tensorElemType, arrX, dimsList.map(y => y.toInt).toArray)
      }

   lazy val nodes = {
      val someNodes = input.map { inputOpt =>
         inputOpt.map { x =>
            val name = x.name.getOrElse("MissingName")
            if params exists (_._1.equals(name)) then ("param_" + name)
            else ("input_" + name)
         } ++ nodeOutputs.flatten.map(y => ("output_" + y))
      }
      someNodes
   }

   lazy val outputs = {
      val outputArray = globalOutput.toArray
      outputArray.map { valueinfoOpt =>
         valueinfoOpt
            .map(x =>
               x.name.map(y => y.replaceAll("-", "_").replaceAll("/", "_")).getOrElse("MissingName")
            )
            .filter(x => nodes.contains("output_" + x))
      }
   }

   lazy val graphInputs = {
      val inputCount = graph.map(x => x.input.size.toInt).getOrElse(0)
      val input      = (0 until inputCount).map(y => graph.map(z => z.input(y)))
      input.toArray
         .map { valueinfoOpt =>
            valueinfoOpt.map { y =>
               (
                 y.name
                    .map(
                      _.replaceAll("-", "_")
                         .replaceAll("/", "_")
                    )
                    .getOrElse("MissingName"),
                 tensorElemTypeMap(y.`type`.map(q => q.getTensorType.getElemType).getOrElse(0))
               )
            }
         }
         .filter(opt => opt.map(z => !(params exists (_._1.equals(z._1)))).getOrElse(false))
         .flatten
   }

   lazy val graphOutputs = {
      val outputCount = graph.map(x => x.output.size.toInt).getOrElse(0)
      val output      = (0 until outputCount).map(y => graph.map(z => z.output(y)))
      output.toArray
         .map(valueInfoOpt =>
            valueInfoOpt.map(y =>
               (
                 y.name
                    .map(z =>
                       z.replaceAll("-", "_")
                          .replaceAll("/", "_")
                    )
                    .getOrElse("MissingName"),
                 tensorElemTypeMap(y.`type`.map(q => q.getTensorType.getElemType).getOrElse(0))
               )
            )
         )
         .filter(opt => opt.map(z => !(params exists (_._1.equals(z._1)))).getOrElse(false))
         .flatten
   }
}
