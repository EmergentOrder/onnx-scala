/*
 * ONNXFreestyleProgramGenerator
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
import scala.meta._
import org.bytedeco.javacpp.onnx.TensorProto
import collection.JavaConverters._
import spire.math.Number

import scala.reflect.ClassTag

object ONNXProgramGenerator {
 def main(args: Array[String]): Unit = {

  val FS = false
  val useDotty = false
  val unionTypeOperator = (if(useDotty) " | " else " TypeOr ")
  //TODO: Get input types from first node
  val inputTypes = "T " + (if(useDotty) "<: " else ": ") + (if(useDotty) "" else "(UNil TypeOr ") +"Float16" + unionTypeOperator + "Float" + unionTypeOperator + "Double" + (if(useDotty) "" else ")#check") + ":Numeric:ClassTag:Field"


  //TODO: Fix output for the benchmark models shown here: https://github.com/onnx/backend-scoreboard
  //TODO: run time benchmarks on the same models
  val fileName = args(0)
  val programName = fileName.stripSuffix(".onnx").capitalize + (if(FS) "Free" else "")
  val path = Paths.get("programGenerator/src/main/scala/generatedprograms/" + programName + ".scala");

  //TODO: Be explicit about model version, metadata
  //Notes, from the standard:
  //"Each model MUST explicitly name the operator sets that it relies on for its functionality."
  //"An implementation must support all operators in the set or reject the model" - This can happen at runtime via Freestyle implicits, possibly mixing backends
  //"Operator sets other than the default operator set MUST specify its domain and SHOULD use reverse domain names based on the responsible organization's identity, the same convention that is used for naming Java packages." - - "Must be unique among all sets."  - Do not support custom opsets initially, backlog
  //"Models MUST specify a domain and use reverse domain names based on the responsible organization's identity, the same convention that is traditionally used for naming Java packages." - Encode this
  //"Note: As of the publication of this document, no ONNX implementation is known to process operator set documents." - backlog

  val onnxHelper = new ONNXHelper(fileName)

  def fullSource = {
    val params = onnxHelper.params
//    val nodes = onnxHelper.nodes
    val nodeInputs = onnxHelper.nodeInputs
    val nodeOutputs = onnxHelper.nodeOutputs
    val outputs = onnxHelper.outputs
    val attributes = onnxHelper.attributes
  //val sortedParamNames = params.keys.toSeq.sorted.map(x => "param_" + x)
    val ops = onnxHelper.ops
    val distinctOps = ops.distinct

    val nodesInputsOpsAndOutputs = (nodeInputs zip ops) zip nodeOutputs

    "package org.emergentorder.onnx" + (if(FS) "Free" else "") + "\n\n" +
    (if(FS) 
      "import freestyle.free._\n" +
      "import freestyle.free.implicits._\n" +
      "import cats.free.{ Free, FreeApplicative } \n" +
      "import cats.implicits._ \n" +
      "import cats.effect.IO\n" +
      "import org.emergentorder.onnx._\n"
      else ""
      )  +
    (if(useDotty) "" else
      "import org.emergentorder.onnx.UnionType._\n"
    ) +
    "import scala.reflect.ClassTag\n" +
    "import spire.implicits._\n" +
    "import spire.math.UByte\n" +
    "import spire.math.UShort\n" +
    "import spire.math.Complex\n" +
    "import spire.algebra.Field\n" +
    "import spire.math.Numeric\n" +
    "import singleton.ops._\n" +
    "import scala.language.higherKinds\n\n" +
    (if(FS) "@module " else "") + "trait " + programName + " {\n" +
    distinctOps
      .map { x =>
        "  val " + x + (if(FS) "Free" else "") + ": " + x.capitalize + (if(FS) "Free" else "") + "\n"
      }
      .mkString("") +
    "  val dataSource: DataSource" + (if(FS) "Free" else "") + "\n" +
//    "  import cats.implicits._\n" +
    //Omit return type here for now
    //TODO: Don't re-use T for all inputData / params
    "  def program[" + inputTypes + "]" + (if(FS) ": FS.Seq[Tensor[T]] " else ": List[Tensor[T]] ") + " = \n" +
    //Body of program generated here
    "    for {\n" +
    //TODO: Assumes one output for now, enable multiple outputs for full computation graph
    "      node" +
    nodeInputs(0)(0) +
    " <- " + (if(FS) "" else "List(") + "dataSource.inputData" +(if(FS) "Free" else "") +"[T]" + (if(FS) "" else ")") + "\n" +
    params
      .map(x =>
        "      node" + x + " <- "
          + (if(FS) "" else "List(") + " dataSource.getParams" + (if(FS) "Free" else "") + "[T](\"" + x + "\")" + (if(FS) "" else ")" ) + "\n")
      .mkString("") +
    (nodesInputsOpsAndOutputs zip attributes)
      .map { x =>
        val nodesOrParams = x._1._1._1.map(y => "node" + y + """, """" + y + """"""")
        val nodesOrParamsRaw = x._1._1._1.map(y => "node" + y)
          val longFields = x._2
          .filter { y => y.has_i
          }
          .map { y =>

            val field = y.i.asInstanceOf[Long]
            y.name.getString + """ = Some((""" + field.toInt + """))"""
          }

          val longListFields = x._2
          .filter { y =>
            val longListCount = y.ints_size
            val longListList = (0 until longListCount.toInt).map(z => y.ints(z)).toList
            !longListList.isEmpty  //|| longList(0).isInstanceOf[Long]
          }
          .map { y =>
            val longListCount = y.ints_size
            val longListList = (0 until longListCount.toInt).map(z => y.ints(z)).toList
            val field = longListList.toVector.asInstanceOf[Vector[Long]]
            y.name.getString + """ = Some((Array(""" + field.mkString(",") + """)))""" 
          }
        val stringFields = x._2
          .filter { y =>
            val stringCount = y.strings_size
            val stringList = (0 until stringCount.toInt).map(z => y.strings(z)).toList
            !stringList.isEmpty //stringList(1).isInstanceOf[String]
          }
          .map { y =>
            val stringCount = y.strings_size
            val stringList = (0 until stringCount.toInt).map(z => y.strings(z)).toList
            val field = stringList.asInstanceOf[String]
            y.name.getString + """ = Some(Array(""" + field + """))"""
          }
        val tensorProtoFields = x._2
          .filter { y =>
            val tensorCount = y.tensors_size
            val tensorList = (0 until tensorCount.toInt).map(z => y.tensors(z)).toList
            //fields(1)._2.isInstanceOf[TensorProto]
            !tensorList.isEmpty //tensorList(1).isInstanceOf[TensorProto]
          }
          .map { y =>
            val tensorCount = y.tensors_size
            val tensorList = (0 until tensorCount.toInt).map(z => y.tensors(z)).toList
            val field = onnxHelper.onnxTensorProtoToArray(
              tensorList.asInstanceOf[TensorProto])
            field match {
              case array: Array[_] => y.name.getString + " = Some((Array(" + array.mkString(",") + ")))"
          
            }
          }
       
        val opName = x._1._1._2
        val nodeName = x._1._2(0) 
        //TODO: Select correct op version instead of 1
        "      node" + nodeName + " <- " + (if(FS) ""
          else "List(") + opName + (if(FS) "Free" else "") + "." + opName + "1" + (if(FS) "Free" else "")  + "[T]" +
        "(" +
        """"""" + nodeName + """", """ + //assumes > 0 args
          nodesOrParams.mkString(",") +
          (if (tensorProtoFields.size > 0) "," else "") +
          tensorProtoFields.mkString(",") +
         (if (longListFields.size > 0) "," else "") +
          longListFields.mkString(",") +
          (if (stringFields.size > 0) "," else "") +
          stringFields.mkString(",") +
          (if (longFields.size > 0) "," else "") +
          longFields.mkString(",") +
          ")" + (if(FS) ""
           // "}" 
            else ")") + "\n"
      }
      .mkString("") +
    "    } yield (" +
    outputs.map(x => "node" + x.name.getString).mkString(",") +
    ")\n" +
    "}\n"
  }
//pw.write("for {\n")

  def generate() = {
    println(fullSource)
    //Seems to not catch some things it should
    val onnxSource = fullSource.parse[Source].get

    Files.write(path, onnxSource.syntax.getBytes("UTF-8"));
  }

  generate()
 }
}
