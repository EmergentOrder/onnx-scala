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
        name = name,
        opType = opName,
        output = Array(outName),
        domain = domain
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
           name = key,
           `type` = AttributeProto.AttributeType.FLOAT,
           f = x
         )
      }

      def createFloatArrayAttr(x: Array[Float], key: String): AttributeProto = {
         AttributeProto(
           name = key,
           `type` = AttributeProto.AttributeType.FLOATS,
           floats = x
         )
      }

      def createIntAttr(x: Int, key: String): AttributeProto = {
         AttributeProto(
           name = key,
           `type` = AttributeProto.AttributeType.INT,
           i = x.toLong
         )
      }

      def createIntArrayAttr(x: Array[Int], key: String): AttributeProto = {
         AttributeProto(
           name = key,
           `type` = AttributeProto.AttributeType.INTS,
           ints = x.map(_.toLong)
         )
      }

      def createStrAttr(x: String, key: String): AttributeProto = {
         AttributeProto(
           name = key,
           `type` = AttributeProto.AttributeType.STRING,
           s = com.google.protobuf.ByteString.copyFromUtf8(x)
         )
      }

      def createStrArrayAttr(x: Array[String], key: String): AttributeProto = {
         AttributeProto(
           name = key,
           `type` = AttributeProto.AttributeType.STRINGS,
           strings = x.map(com.google.protobuf.ByteString.copyFromUtf8(_))
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
                  case Some(x: Int) => {
                     Some(createIntAttr(x, key))
                  }
                  case x: Array[Int] => {
                     Some(createIntArrayAttr(x, key))
                  }
                  case Some(x: Array[Int]) => {
                     Some(createIntArrayAttr(x, key))
                  }
                  case x: Float => {
                     Some(createFloatAttr(x, key))
                  }
                  case Some(x: Float) => {
                     Some(createFloatAttr(x, key))
                  }
                  case x: Array[Float] => {
                     Some(createFloatArrayAttr(x, key))
                  }
                  case Some(x: Array[Float]) => {
                     Some(createFloatArrayAttr(x, key))
                  }
                  case x: String => {
                     Some(createStrAttr(x, key))
                  }
                  case Some(x: String) => {
                     Some(createStrAttr(x, key))
                  }
                  case x: Array[String] => {
                     Some(createStrArrayAttr(x, key))
                  }
                  case Some(x: Array[String]) => {
                     Some(createStrArrayAttr(x, key))
                  }
                  case _ => None // None => None
               }
            }
            .toArray
            .flatten

      val newNode = node.withAttribute(attrProtos)

      return newNode
   }

   protected def createInputValueInfoProto[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](elemTypeIn: Int, shape: Array[Int], inputName: String): ValueInfoProto = {

      {

         val elemTypeORT: Int = elemTypeIn match {
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
           name = { inputName },
           `type` = Some(
             onnx.onnx
                .TypeProto()
                .withTensorType(
                  onnx.onnx.TypeProto.Tensor(
                    shape = Some(
                      onnx.onnx.TensorShapeProto(
                        dim = shape.map(onnx.onnx.TensorShapeProto.Dimension().withDimValue(_))
                      )
                    ),
                    elemType = elemTypeORT
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
      val inputValueInfosAndExistingInputs: Array[Tuple2[ValueInfoProto, String]] =
         inputs.zipWithIndex.map { x =>
            (createInputValueInfoProto(x._1._1, x._1._2, x._2.toString), x._2.toString)
         }
      //       .sequence

      val outName = inputs.size.toString

      val newGraph = {

         val node = nodeWithAddedInputs(
           inputValueInfosAndExistingInputs.map(_._2).toList,
           opToNode(inputs.toString.hashCode.toString, opName, outName, attrs, thisDomain)
         )

         //        node.map { y =>
         GraphProto(
           name = inputs.toString,
           output = Array(ValueInfoProto(name = outName)),
           input = inputValueInfosAndExistingInputs.map(_._1),
           node = Array(node)
         )
         //      }
      }

      val thisOpset = if opName.equals("Inverse") then 1 else 17
      val model     = {
         val mod = ModelProto(
           producerName = "ONNX-Scala",
           graph = Some(newGraph),
           domain = thisDomain,
           irVersion = 8,
           opsetImport = Array(OperatorSetIdProto(version = thisOpset))
         )
         mod
      }
      model
   }
}
