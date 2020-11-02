package org.emergentorder.onnx

//import scala.language.implicitConversions

import org.bytedeco.onnx.ModelProto
import org.bytedeco.onnx.NodeProto
import org.bytedeco.onnx.GraphProto
import org.bytedeco.onnx.TensorProto
import org.bytedeco.onnx.AttributeProto
import org.bytedeco.javacpp.PointerScope
import org.bytedeco.javacpp.BytePointer
import org.emergentorder.onnx.Tensors._


trait OpToONNXBytesConverter extends AutoCloseable {

//  private val scope = new PointerScope()

  protected def opToNode[
      T <: Supported
  ](
      name: String,
      opName: String,
//      inputs: Tuple,
      outName: String,
      attrs: Map[String, Any]
  )
  //(
//        implicit evT:  (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
  //       Float] TypeOr Complex[Double])#check[T])
      : NodeProto = {
    val node = new NodeProto
    //(new NodeProto).New()

    node.set_name(name)
    node.set_op_type(opName)
    node.add_output(outName)

    def handleIntAttrs(x: Int, key: String): Unit = {
      val attr     = node.add_attribute
      val attrName = new BytePointer(key)
      attr.set_name(attrName)
      attr.set_type(AttributeProto.INT)
      val longVal = x.toLong
      attr.set_i(longVal)
   
    }

    def handleIntArrayAttrs(x: Array[Int], key: String): Unit = {
      val attr     = node.add_attribute
      val attrName = new BytePointer(key)
      attr.set_name(attrName)
      attr.set_type(AttributeProto.INTS)
      (0 until x.size).foreach(y => attr.add_ints(x(y).toLong))
  
    }

    def handleAttrs: Unit =
      attrs.foreach {
        case (key, value) =>
          value match {
            case x: Int => {
              handleIntAttrs(x, key)
            }
            case Some(x: Int) => {
              handleIntAttrs(x, key)
            }
            case x: Array[Int] => {
              handleIntArrayAttrs(x, key)
            }
            case Some(x: Array[Int]) => {
              handleIntArrayAttrs(x, key)
            }
            case None =>
          }
      }

//    def addInput(inputName: String): Unit = { 
//          node.add_input(inputName)
//    }
    //Dummy names
  
//        (0 until inputs.size).foreach { i => addInput(i.toString) }
    

 
    handleAttrs

    return node
  }

  //TODO: prevent passing the inputs all the way down here
  protected def addInputToGraph[T <: Supported](tens: Tensor[T, ?], inputName: String, graph: GraphProto, node: NodeProto): Unit = {
    node.add_input(inputName)
    val elemType = tens._1 match {
          case b: Array[Byte]   => TensorProto.INT8
          case s: Array[Short]  => TensorProto.INT16
          case d: Array[Double] => TensorProto.DOUBLE
          case f: Array[Float]  => TensorProto.FLOAT
          case i: Array[Int]    => TensorProto.INT32
          case l: Array[Long]   => TensorProto.INT64
          case b: Array[Boolean] => TensorProto.BOOL
        }

        val inputValueInfo = graph.add_input

        inputValueInfo.set_name(inputName)
        inputValueInfo.mutable_type
        inputValueInfo.`type`.mutable_tensor_type
        inputValueInfo.`type`.tensor_type.set_elem_type(elemType)

        val dims = tens._2 
        inputValueInfo.`type`.tensor_type.mutable_shape
        dims.foreach { x =>
          val inputDim = inputValueInfo.`type`.tensor_type.shape.add_dim

//              inputDim.set_dim_param("NAME?")
          inputDim.set_dim_value(x)
        }
 
  }

  def opToONNXBytes[
      T <: Supported
  ](
      name: String,
      opName: String,
      inputs: Tuple,
      outName: String,
      attrs: Map[String, Any]
  ): Array[Byte] = {
    val model = (new ModelProto).New()
    val graph = new org.bytedeco.onnx.GraphProto
    model.set_producer_name("ONNX-Scala")
    graph.set_name(name)

    val origNode = opToNode(name, opName, outName, attrs)

    val node = graph.add_node
    node.MergeFrom(origNode)

    origNode.close
    model.set_allocated_graph(graph)
    model.set_ir_version(6)

    model.add_opset_import
    model.opset_import(0).set_version(12)

    val outputValueInfo = graph.add_output

    outputValueInfo.set_name(outName)

    //Dummy names

        (0 until inputs.size).foreach { i =>
          val t = inputs.drop(i).take(1)
          t match {
            case tup: Tuple1[_] => 
              tup(0) match {
                case opt: Option[Tensor[T, ?]] =>
                opt match {
                  case Some(in) => addInputToGraph(in, i.toString, graph, node)
                  case None =>
                }
               case tens: Tensor[T, ?] => {
                 addInputToGraph(tens, i.toString, graph, node)
               } 
              }
          } 
        }

    val modelString = model.SerializeAsString


    node.close
    model.close
    graph.close
    val modelStringBytes = modelString.getStringBytes
    modelString.close

    (modelStringBytes)
  }

  override def close(): Unit = {
//    scope.close
  }

}
