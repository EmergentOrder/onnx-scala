package org.emergentorder.onnx

//import scala.language.implicitConversions
import onnx.onnx.AttributeProto
import onnx.onnx.GraphProto
import onnx.onnx.ModelProto
import onnx.onnx.NodeProto
import onnx.onnx.OperatorSetIdProto
import onnx.onnx.TensorProto.DataType.*
import onnx.onnx.ValueInfoProto
import org.emergentorder.compiletime.*
import org.emergentorder.io.kjaer.compiletime.Shape
import org.emergentorder.onnx.Tensors.*

import scala.collection.immutable.ArraySeq
import scala.compiletime.asMatchable

trait OpToONNXBytesConverter {

   protected def opToNode[
       T <: Supported
   ](
       name: String,
       opName: String,
       outName: String,
       attrs: Map[String, Any],
       domain: String
   ): NodeProto = {
      val node = NodeProto(
        name = Some(name),
        opType = Some(opName),
        output = (Seq(outName)),
        domain = Some(domain)
      )

      /*
      def createFloatTensorAttr[
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](tens: Tensor[Float, Tuple3[Tt, Td, S]], key: String): AttributeProto = {
         val data  = tens.data
         val shape = tens.shape

         data.flatMap { x =>
            shape.map { y =>
               AttributeProto(
                 name = Some(key),
                 `type` = Some(AttributeProto.AttributeType.TENSOR),
                 t = Some(
                   TensorProto()
                      .withDataType(TensorProto.DataType.FLOAT.value)
                      .withDims(ArraySeq.unsafeWrapArray(y.map(_.toLong)))
                      .withFloatData(ArraySeq.unsafeWrapArray(x))
                 )
               )
            }
         }
      }
       */
      def createFloatAttr(x: Float, key: String): AttributeProto = {
         AttributeProto(
           name = Some(key),
           `type` = Some(AttributeProto.AttributeType.FLOAT),
           f = Some(x)
         )
      }

      def createFloatArrayAttr(x: Array[Float], key: String): AttributeProto = {
         AttributeProto(
           name = Some(key),
           `type` = Some(AttributeProto.AttributeType.FLOATS),
           floats = ArraySeq.unsafeWrapArray(x)
         )
      }

      def createIntAttr(x: Int, key: String): AttributeProto = {
         AttributeProto(
           name = Some(key),
           `type` = Some(AttributeProto.AttributeType.INT),
           i = Some(x.toLong)
         )
      }

      def createIntArrayAttr(x: Array[Int], key: String): AttributeProto = {
         AttributeProto(
           name = Some(key),
           `type` = Some(AttributeProto.AttributeType.INTS),
           ints = ArraySeq.unsafeWrapArray(x.map(_.toLong))
         )
      }

      def createStrAttr(x: String, key: String): AttributeProto = {
         AttributeProto(
           name = Some(key),
           `type` = Some(AttributeProto.AttributeType.STRING),
           s = Some(com.google.protobuf.ByteString.copyFromUtf8(x))
         )
      }

      def createStrArrayAttr(x: Array[String], key: String): AttributeProto = {
         AttributeProto(
           name = Some(key),
           `type` = Some(AttributeProto.AttributeType.STRINGS),
           strings = ArraySeq.unsafeWrapArray(x.map(com.google.protobuf.ByteString.copyFromUtf8(_)))
         )
      }

      // TODO: more attr typs
      def attrProtos[Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape]
          : Array[AttributeProto] =
         attrs
            .map { case (key: String, value) =>
               value.asMatchable match {
//                  case x: Tensor[Float, Tuple3[Tt, Td, S]] => {
//                     Some(createFloatTensorAttr(x, key))
//                  }
//                  case Some(x: Tensor[Float, Tuple3[Tt, Td, S]]) => {
//                     Some(createFloatTensorAttr(x, key))
                  //                 }
                  case x: Int => {
                     Some(createIntAttr(x, key))
                  }
                  case Some(x: Int): Some[Int] => {
                     Some(createIntAttr(x, key))
                  }
                  case x: Array[Int] => {
                     Some(createIntArrayAttr(x, key))
                  }
                  case Some(x: Array[Int]): Some[Array[Int]] => {
                     Some(createIntArrayAttr(x, key))
                  }
                  case x: Float => {
                     Some(createFloatAttr(x, key))
                  }
                  case Some(x: Float): Some[Float] => {
                     Some(createFloatAttr(x, key))
                  }
                  case x: Array[Float] => {
                     Some(createFloatArrayAttr(x, key))
                  }
                  case Some(x: Array[Float]): Some[Array[Float]] => {
                     Some(createFloatArrayAttr(x, key))
                  }
                  case x: String => {
                     Some(createStrAttr(x, key))
                  }
                  case Some(x: String): Some[String] => {
                     Some(createStrAttr(x, key))
                  }
                  case x: Array[String] => {
                     Some(createStrArrayAttr(x, key))
                  }
                  case Some(x: Array[String]): Some[Array[String]] => {
                     Some(createStrArrayAttr(x, key))
                  }
                  case None => None
               }
            }
            .toArray
            .flatten

      val newNode = node.withAttribute(ArraySeq.unsafeWrapArray(attrProtos))

      return newNode
   }

   protected def createInputValueInfoProto[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](elemTypeIn: Int, shape: Array[Int], inputName: String): ValueInfoProto = {

      {

         val elemType = elemTypeIn match {
            case 3  => INT8.index
            case 5  => INT16.index
            case 11 => DOUBLE.index
            case 1  => FLOAT.index
            case 6  => INT32.index
            case 7  => INT64.index
            case 9  => BOOL.index
            case _  => INT64.index // In case of Scala.js BigInt
         }
//         tens.shape.map { y =>
         ValueInfoProto(
           name = { Some(inputName) },
           `type` = Some(
             onnx.onnx
                .TypeProto()
                .withTensorType(
                  onnx.onnx.TypeProto.Tensor(
                    shape = Some(
                      onnx.onnx.TensorShapeProto(
                        dim = ArraySeq.unsafeWrapArray(
                          shape.map(onnx.onnx.TensorShapeProto.Dimension().withDimValue(_))
                        )
                      )
                    ),
                    elemType = Some(elemType)
                  )
                )
           )
         ).copy()
//         }
      }
   }

   def opToModelProto[
       T <: Supported
   ](
       opName: String,
       inputs: Array[Tuple2[Int, Array[Int]]],
       attrs: Map[String, Any]
   ): ModelProto = {

      val thisDomain = if opName.equals("Inverse") then "com.microsoft" else "ai.onnx"

      def nodeWithAddedInputs(inputNames: List[String], node: NodeProto): NodeProto = {
         inputNames match {
            case x :: tail => {
               nodeWithAddedInputs(tail, node.addInput(x).copy())
            }
            case Nil => node
         }
      }

      // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
//      @annotation.nowarn
      val inputValueInfosAndExistingInputs: List[Tuple2[ValueInfoProto, String]] =
         inputs.zipWithIndex.map { x =>
            (createInputValueInfoProto(x._1._1, x._1._2, x._2.toString), x._2.toString)
         }.toList
      //       .sequence

      val outName = inputs.size.toString

      val newGraph = {

         val node = nodeWithAddedInputs(
           inputValueInfosAndExistingInputs.map(_._2),
           opToNode(inputs.toString.hashCode.toString, opName, outName, attrs, thisDomain)
         )

         //        node.map { y =>
         GraphProto(
           name = Some(inputs.toString),
           output = Seq(ValueInfoProto(name = Some(outName))),
           input = inputValueInfosAndExistingInputs.map(_._1),
           node = Seq(node)
         )
         //      }
      }

      val thisOpset = if opName.equals("Inverse") then 1 else 17
      val model     = {
         val mod = ModelProto(
           producerName = Some("ONNX-Scala"),
           graph = Some(newGraph),
           domain = Some(thisDomain),
           irVersion = Some(8),
           opsetImport = Seq(OperatorSetIdProto(version = Some(thisOpset)))
         )
         mod
      }
      model
   }
}
