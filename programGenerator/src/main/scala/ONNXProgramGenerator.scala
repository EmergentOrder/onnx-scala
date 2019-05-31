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
import org.bytedeco.onnx.TensorProto
import collection.JavaConverters._
import spire.math.Number

import scala.reflect.ClassTag

object ONNXProgramGenerator {
  def main(args: Array[String]): Unit = {

    @SuppressWarnings(Array("org.wartremover.warts.Equals"))
    implicit final class AnyOps[A](self: A) {
      def ===(other: A): Boolean = self == other
    }

    val fileName = args(0)
    val onnxHelper = new ONNXHelper(fileName)

    val maxOpsetVersion = onnxHelper.maxOpsetVersion

    val schemas =
      org.bytedeco.onnx.OpSchemaRegistry.get_all_schemas_with_history
    val schemasSize = schemas.size

    val scalaCollSchemas = (0 until schemasSize.toInt).map(x => schemas.get(x))
    val schemaMap = scalaCollSchemas
      .filter(x => x.since_version <= maxOpsetVersion)
      .map(x =>
        x.Name.getString ->
          (x.inputs, x.since_version))
      .toMap

    val FS = false
    val useDotty = false
    val unionTypeOperator = (if (useDotty) " | " else " TypeOr ")


    //TODO: Fix output for the benchmark models shown here: https://github.com/onnx/backend-scoreboard
    //TODO: run time benchmarks on the same models

    val programName = fileName.stripSuffix(".onnx").capitalize + (if (FS) "Free"
                                                                  else "")
    val path = Paths.get(
      "programGenerator/src/main/scala/generatedprograms/" + programName + ".scala");

    //TODO: Be explicit about model version, metadata
    //Notes, from the standard:
    //"Each model MUST explicitly name the operator sets that it relies on for its functionality."
    //"An implementation must support all operators in the set or reject the model" - This can happen at runtime via Freestyle implicits, possibly mixing backends
    //"Operator sets other than the default operator set MUST specify its domain and SHOULD use reverse domain names based on the responsible organization's identity, the same convention that is used for naming Java packages." - - "Must be unique among all sets."  - Do not support custom opsets initially, backlog
    //"Models MUST specify a domain and use reverse domain names based on the responsible organization's identity, the same convention that is traditionally used for naming Java packages." - Encode this
    //"Note: As of the publication of this document, no ONNX implementation is known to process operator set documents." - backlog

    def fullSource = {
      val params = onnxHelper.params
//    val nodes = onnxHelper.nodes
      val nodeInputs = onnxHelper.nodeInputs
      val graphInputs = onnxHelper.graphInputs
      val graphOutputs = onnxHelper.graphOutputs

      val nodeOutputs = onnxHelper.nodeOutputs
      val outputs = onnxHelper.outputs
      val attributes = onnxHelper.attributes

      //val sortedParamNames = params.keys.toSeq.sorted.map(x => "param_" + x)
      val ops = onnxHelper.ops
      val distinctOps = ops.distinct

       def replaceTypeStrings(s: String) = s.replaceAll("uint64", "ULong")
            .replaceAll("uint32", "UInt")
            .replaceAll("uint16", "UShort")
            .replaceAll("uint8", "UByte")
            .replaceAll("int64", "Long")
            .replaceAll("Int64", "Long")
            .replaceAll("int32", "Int")
            .replaceAll("Int32", "Int")
            .replaceAll("int16", "Short")
            .replaceAll("int8", "Byte")
            .replaceAll("string", "String")
            .replaceAll("float", "Float")
            .replaceAll("double", "Double")
            .replaceAll("Bool", "Boolean")
            .replaceAll("bool", "Boolean")
            .replaceAll("complex64", "Complex[Float]")
.replaceAll("complex128", "Complex[Double]")

     val graphOutputType = replaceTypeStrings(graphOutputs(0)._2)

      val nodesInputsOpsAndOutputs = (nodeInputs zip ops) zip nodeOutputs

      "package org.emergentorder.onnx" + (if (FS) "Free" else "") + "\n\n" +
        (if (FS)
           "import freestyle.free._\n" +
             "import freestyle.free.implicits._\n" +
             "import cats.free.{ Free, FreeApplicative } \n" +
             "import cats.implicits._ \n" +
             "import cats.effect.IO\n" +
             "import org.emergentorder.onnx._\n"
         else "") +
        (if (useDotty) ""
         else
           "import org.emergentorder.onnx.UnionType._\n") +
        "import scala.reflect.ClassTag\n" +
        "import spire.implicits._\n" +
        "import spire.math.UByte\n" +
        "import spire.math.UShort\n" +
        "import spire.math.Complex\n" +
        "import spire.algebra.Field\n" +
        "import spire.math.Numeric\n" +
        "import singleton.ops._\n" +
        "import scala.language.higherKinds\n\n" +
        (if (FS) "@module " else "") + "trait " + programName + " {\n" +
        distinctOps
          .map { x =>
            "  val " + x + (if (FS) "Free" else "") + ": " + x.capitalize + (if (FS)
                                                                               "Free"
                                                                             else
                                                                               "") + "\n"
          }
          .mkString("") +
        "  val dataSource: DataSource" + (if (FS) "Free" else "") + "\n" +
//    "  import cats.implicits._\n" +
        //Omit return type here for now
        "  def program" + (if (FS) ": FS.Seq[Tensor[" + graphOutputType + "]] "
                                               else
                                                 ": List[Tensor[" + graphOutputType + "]] ") + " = \n" +
        //Body of program generated here
        "    for {\n" +
        //TODO: Assumes one output for now, enable multiple outputs for full computation graph
        graphInputs.map{ x =>
          "      node" + x._1.replaceAll("\\.","") +
        " <- " + (if (FS) "" else "List(") + "dataSource.inputData" + (if (FS)
                                                                         "Free"
                                                                       else
                                                                         "") + "[" + replaceTypeStrings(x._2) + "]" + //"[T]" +
                                                                       (if (FS)
                                                                                          ""
                                                                                        else
                                                                                          ")")
        }.mkString("\n") +
        "\n" +
        params
          .map(x =>
            "      node" + x._1.replaceAll("\\.", "") + " <- "
              + (if (FS) "" else "List(") + " dataSource.getParams" + (if (FS)
                                                                         "Free"
                                                                       else
                                                                         "") + "[" + x._2 + "]" + "(\"" + x._1 + "\")" + (if (FS) ""
                                                                                                            else
                                                                                                              ")") + "\n")
          .mkString("") +
        (nodesInputsOpsAndOutputs zip attributes)
          .map { x =>
            //TODO: handle multiple outputs
            val nodesOrParams = x._1._1._1.map{
              y =>
                "Some(node" + y.replaceAll("\\.", "") + (if (y.contains("dropout") || y.contains(
                                         "bn_1") || y.contains(
                                         "pool5_7x7_s1_2")) "._1"
                                   else "") + ")"
            }// ,""" + y.name.getString + "name" + " = " + """ Some("""" + y + """")""")

            val nodesOrParamsRaw = x._1._1._1.map(y => "node" + y)
            val longFields = x._2
              .filter { y =>
                y.has_i
              }
              .map { y =>
                val field = y.i.asInstanceOf[Long]
                y.name.getString + """ = Some((""" + field.toInt + """))"""
              }

            val longListFields = x._2
              .filter { y =>
                val longListCount = y.ints_size
                val longListList =
                  (0 until longListCount.toInt).map(z => y.ints(z)).toList
                !longListList.isEmpty //|| longList(0).isInstanceOf[Long]
              }
              .map { y =>
                val longListCount = y.ints_size
                val longListList =
                  (0 until longListCount.toInt).map(z => y.ints(z)).toList
                val field = longListList.toVector.asInstanceOf[Vector[Long]]
                y.name.getString + """ = Some((Array(""" + field.mkString(",") + """)))"""
              }
            val stringFields = x._2
              .filter { y =>
                val stringCount = y.strings_size
                val stringList =
                  (0 until stringCount.toInt).map(z => y.strings(z)).toList
                !stringList.isEmpty //stringList(1).isInstanceOf[String]
              }
              .map { y =>
                val stringCount = y.strings_size
                val stringList =
                  (0 until stringCount.toInt).map(z => y.strings(z)).toList
                val field = stringList.asInstanceOf[String]
                y.name.getString + """ = Some(Array(""" + field + """))"""
              }
            val tensorProtoFields = x._2
              .filter { y =>
                val tensorCount = y.tensors_size
                val tensorList =
                  (0 until tensorCount.toInt).map(z => y.tensors(z)).toList
                //fields(1)._2.isInstanceOf[TensorProto]
                !tensorList.isEmpty //tensorList(1).isInstanceOf[TensorProto]
              }
              .map { y =>
                val tensorCount = y.tensors_size
                val tensorList =
                  (0 until tensorCount.toInt).map(z => y.tensors(z)).toList
                val field = onnxHelper.onnxTensorProtoToArray(
                  tensorList.asInstanceOf[TensorProto])
                field match {
                  case array: Array[_] =>
                    y.name.getString + " = Some((Array(" + array.mkString(",") + ")))"

                }
              }

            val opName = x._1._1._2

            val opInputsNames = (0 until schemaMap(opName)._1.size.toInt).map {
              b =>
                schemaMap(opName)._1.get(b).GetName.getString
            }

            val opInputsIsVariadic =
              (0 until schemaMap(opName)._1.size.toInt).map { b =>
                schemaMap(opName)._1.get(b).GetOption === 2
              }

            val sinceVersion = schemaMap(opName)._2.toString 

            val groupedNodesOrParams: Array[String] = nodesOrParams.take(opInputsNames.size - 1) ++ Seq(nodesOrParams.drop(opInputsNames.size -1).mkString(","))

            val opInputs = (opInputsNames zip opInputsIsVariadic) zip groupedNodesOrParams 

            val opInputsCleaned = opInputs.map{
              t =>
                ((t._1._1, t._1._2),
                 (if (opName.contains("Reshape") && sinceVersion
                        .equals("5") && t._1._1.equals("shape"))
                    t._2 + ".asInstanceOf[Option[Tensor[Long]]]"
                  else t._2))
            }

            val namedNodesOrParams = opInputsCleaned.map(
              t =>
                t._1._1
                  .replaceAll("var", "someVar") + " = " + (if (t._1._2)
                                                             t._2
                                                               .replaceFirst(
                                                                 "Some",
                                                                 "Seq(Some")
                                                                + ")"
                                                           else t._2))

            val nodeName = x._1._2(0)

            "      node" + nodeName.replaceAll("\\.", "") + " <- " + (if (FS) ""
                                                else
                                                  "List(") + opName + (if (FS)
                                                                         "Free"
                                                                       else
                                                                         "") + "." + opName + sinceVersion + (if (FS)
                                                                                                                "Free"
                                                                                                              else
                                                                                                                "") + 
                                                                                                              (if( nodeName.contains("output")) "[" + graphOutputType + "]" else "") +
              "(" +
              """"""" + nodeName + """" """ + //assumes > 0 args
              (if (tensorProtoFields.size > 0) "," else "") +
              tensorProtoFields.mkString(",") +
              (if (longListFields.size > 0) "," else "") +
              longListFields.mkString(",") +
              (if (stringFields.size > 0) "," else "") +
              stringFields.mkString(",") +
              (if (longFields.size > 0) "," else "") +
              longFields.mkString(",") +
              "," +
              namedNodesOrParams.mkString(",") +
              ")" + (if (FS) ""
                     // "}"
                     else ")") + "\n"
          }
          .mkString("") +
        "    } yield (" +
        outputs.map(x => "node" + x).mkString(",") +
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
