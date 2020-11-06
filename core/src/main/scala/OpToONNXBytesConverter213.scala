package org.emergentorder.onnx

//import scala.language.implicitConversions

import onnx.onnx.ModelProto
import onnx.onnx.NodeProto
import onnx.onnx.GraphProto
import onnx.onnx.TensorProto
import onnx.onnx.AttributeProto
import onnx.onnx.ValueInfoProto
import onnx.onnx.OperatorSetIdProto
import onnx.onnx.TensorProto.DataType._

import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._

trait OpToONNXBytesConverter extends AutoCloseable {

  protected def opToNode[
      T
  ](
      name: String,
      opName: String,
//      inputs: Tuple,
      outName: String,
      attrs: Map[String, Any]
  ) 
      : NodeProto = {
    val node = NodeProto(name=Some(name),opType=Some(opName), output=(Seq(outName)))
    
    def createIntAttr(x: Int, key: String): AttributeProto = {
      AttributeProto(name=Some(key),`type`=Some(AttributeProto.AttributeType.INT),i=Some(x.toLong))    
    }

    def createIntArrayAttr(x: Array[Int], key: String): AttributeProto = {
      AttributeProto(name=Some(key),`type`=Some(AttributeProto.AttributeType.INTS),ints=x.map(_.toLong))  
    }

    //TODO: more attr types
    val attrProtos: Array[AttributeProto] =
      attrs.map {
        case (key:String, value) =>
          value match {
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
            case None => None
          }
      }.toArray.flatten
 
    val newNode = node.withAttribute(attrProtos)

    return newNode
  }

  //TODO: prevent passing the inputs all the way down here
  protected def createInputValueInfoProto[T, Ax <: Axes](tens: Tensor[T, Ax], inputName: String): ValueInfoProto = {
//    node.addInput(inputName)
    val elemType = tens._1 match {
          case b: Array[Byte]   => INT8.index
          case s: Array[Short]  => INT16.index
          case d: Array[Double] => DOUBLE.index
          case f: Array[Float]  => FLOAT.index
          case i: Array[Int]    => INT32.index
          case l: Array[Long]   => INT64.index
          case b: Array[Boolean] => BOOL.index
        }

      ValueInfoProto(name=Some(inputName), `type`=Some(onnx.onnx.TypeProto().withTensorType(onnx.onnx.TypeProto.Tensor(shape=Some(onnx.onnx.TensorShapeProto(dim=shape(tens).map(onnx.onnx.TensorShapeProto.Dimension().withDimValue(_)))),
          elemType=Some(elemType)))
            //.asInstanceOf[onnx.onnx.TypeProto]
            ))
/*



//              inputDim.set_dim_param("NAME?")
 */

  }

  def opToONNXBytes[
      T
  ](
      name: String,
      opName: String,
      inputs: Seq[_],
      outName: String,
      attrs: Map[String, Any]
  ): Array[Byte] = {

    def nodeWithAddedInputs(inputNames: List[String], node: NodeProto, name: String): NodeProto = {
      inputNames match { 
        case x :: tail => {nodeWithAddedInputs(tail, node.addInput(x), name)}
        case Nil => node
      }
    }

    //Dummy names

    val inputValueInfosAndExistingInputs: IndexedSeq[Tuple2[ValueInfoProto, String]] = (0 until inputs.size).map { i =>
          val t = inputs(i)
          t match {
//            case tup: Tuple1[_] => 
//              tup(0) match {
                case opt: Option[Tensor[T, _]] =>
                opt match {
                  case Some(in) => {Some((createInputValueInfoProto(in, i.toString), i.toString))}
                  case None => None
                }
                case tens: Tensor[T, _] =>  {Some((createInputValueInfoProto(tens, i.toString), i.toString)
                                    )}  
              }
//          } 
        }.flatten

    val node = nodeWithAddedInputs(inputValueInfosAndExistingInputs.map(_._2).toList, opToNode(name, opName, outName, attrs), name) 

    val newGraph = GraphProto(name = Some(name), output=Seq(ValueInfoProto(name=Some(outName))), input = inputValueInfosAndExistingInputs.map(_._1), node = Seq(node)) 
  
    val model = ModelProto(producerName=Some("ONNX-Scala"), graph=Some(newGraph), irVersion=Some(6), opsetImport=Seq(OperatorSetIdProto(version=Some(12))))
  
    (model.toByteArray)
  }

  override def close(): Unit = {
//    scope.close
  }

}
