package org.emergentorder.onnx

//import scala.language.implicitConversions
import scala.collection.immutable.ArraySeq

import onnx.onnx.ModelProto
import onnx.onnx.NodeProto
import onnx.onnx.GraphProto
import onnx.onnx.TensorProto
import onnx.onnx.AttributeProto
import onnx.onnx.ValueInfoProto
import onnx.onnx.OperatorSetIdProto
import onnx.onnx.TensorProto.DataType._

import io.kjaer.compiletime.Shape
import org.emergentorder.compiletime._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._

trait OpToONNXBytesConverter extends AutoCloseable {

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

      def createFloatTensorAttr[
          Tt <: TensorTypeDenotation,
          Td <: TensorShapeDenotation,
          S <: Shape
      ](x: Tensor[Float, Tuple3[Tt, Td, S]], key: String): AttributeProto = {
         AttributeProto(
           name = Some(key),
           `type` = Some(AttributeProto.AttributeType.TENSOR),
           t = Some(
             TensorProto()
                .withDataType(TensorProto.DataType.FLOAT.value)
                .withDims(ArraySeq.unsafeWrapArray(x.shape.map(_.toLong)))
                .withFloatData(ArraySeq.unsafeWrapArray(x.data))
           )
         )
      }

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

      // TODO: more attr types
      def attrProtos[Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape]
          : Array[AttributeProto] =
         attrs
            .map { case (key: String, value) =>
               value match {
                  case x: Tensor[Float, Tuple3[Tt, Td, S]] => {
                     Some(createFloatTensorAttr(x, key))
                  }
                  case Some(x: Tensor[Float, Tuple3[Tt, Td, S]]) => {
                     Some(createFloatTensorAttr(x, key))
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
   ](tens: Tensor[T, (Tt, Td, S)], inputName: String): ValueInfoProto = {

      val elemType = tens.data match {
         case b: Array[Byte]    => INT8.index
         case s: Array[Short]   => INT16.index
         case d: Array[Double]  => DOUBLE.index
         case f: Array[Float]   => FLOAT.index
         case i: Array[Int]     => INT32.index
         case l: Array[Long]    => INT64.index
         case b: Array[Boolean] => BOOL.index
      }

      ValueInfoProto(
        name = Some(inputName),
        `type` = Some(
          onnx.onnx
             .TypeProto()
             .withTensorType(
               onnx.onnx.TypeProto.Tensor(
                 shape = Some(
                   onnx.onnx.TensorShapeProto(dim =
                      ArraySeq.unsafeWrapArray(
                        tens.shape.map(onnx.onnx.TensorShapeProto.Dimension().withDimValue(_))
                      )
                   )
                 ),
                 elemType = Some(elemType)
               )
             )
        )
      )
   }

   def opToModelProto[
       T <: Supported
   ](
       opName: String,
       inputs: Tuple,
       attrs: Map[String, Any]
   ): ModelProto = {

      val thisDomain = if (opName.equals("Inverse")) "com.microsoft" else "ai.onnx"

      def nodeWithAddedInputs(inputNames: List[String], node: NodeProto): NodeProto = {
         inputNames match {
            case x :: tail => { nodeWithAddedInputs(tail, node.addInput(x)) }
            case Nil       => node
         }
      }

      // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
      val inputValueInfosAndExistingInputs: IndexedSeq[Tuple2[ValueInfoProto, String]] =
         (0 until inputs.size).map { i =>
            val t = inputs.drop(i).take(1)
            t match {
               case tup: Tuple1[_] => // TODO: union type
                  tup(0) match {
                     case opt: Option[Tensor[T, Axes]] =>
                        opt match {
                           case Some(in) => {
                              val incr: String =
                                 if (inputs.toArray.distinct.size == inputs.size) "" else i.toString
                              val name = (opt.toString + incr).hashCode.toString
                              Some((createInputValueInfoProto(in, name), name))
                           }
                           case None => None
                        }
                     case tens: Tensor[T, Axes] => {
                        val incr: String =
                           if (inputs.toArray.distinct.size == inputs.size) "" else i.toString
                        val name = ((tens.toString + incr).hashCode).toString
                        Some((createInputValueInfoProto(tens, name), name))
                     }
                  }
            }
         }.flatten

      val outName = inputs.toArray.map(_.toString).toList.toString.hashCode.toString

      val node = nodeWithAddedInputs(
        inputValueInfosAndExistingInputs.map(_._2).toList,
        opToNode(inputs.toString.hashCode.toString, opName, outName, attrs, thisDomain)
      )

      val newGraph = GraphProto(
        name = Some(inputs.toString),
        output = Seq(ValueInfoProto(name = Some(outName))),
        input = inputValueInfosAndExistingInputs.map(_._1),
        node = Seq(node)
      )

      val thisOpset = if (opName.equals("Inverse")) 1 else 13
      val model = ModelProto(
        producerName = Some("ONNX-Scala"),
        graph = Some(newGraph),
        domain = Some(thisDomain),
        irVersion = Some(7),
        opsetImport = Seq(OperatorSetIdProto(version = Some(thisOpset)))
      )
      model
   }
}
