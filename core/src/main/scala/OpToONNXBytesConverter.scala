package org.emergentorder.onnx

import scala.reflect.ClassTag

import org.bytedeco.onnx.ModelProto
import org.bytedeco.onnx.NodeProto
import org.bytedeco.onnx.GraphProto
import org.bytedeco.onnx.TensorProto
import org.bytedeco.onnx.AttributeProto
import org.bytedeco.javacpp.PointerScope
import org.bytedeco.javacpp.BytePointer

trait OpToONNXBytesConverter extends AutoCloseable {

  private val scope = new PointerScope()

  protected def opToNode[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag
  ](
      name: String,
      opName: String,
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
      outName: String,
      attrs: Map[String, Any]
  )
  //(
//        implicit evT:  (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
  //       Float] TypeOr Complex[Double])#check[T])
      : NodeProto = {
    val node = (new NodeProto).New()

    node.set_name(name)
    node.set_op_type(opName)
    node.add_output(outName)

    def handleAttrs = attrs.foreach {
      case (key, value) =>
        val res = value match {
          case Some(x: Int) => {
            val attr = node.add_attribute
            //val attr     = node.mutable_attribute(0)
            val attrName = new BytePointer(key)
            attr.set_name(attrName)
            attr.set_type(AttributeProto.INT)
            val longVal = x.toLong

            attr.set_i(longVal)
          }
          case Some(x: Array[Int]) => {
            val attr = node.add_attribute
            //val attr = node.mutable_attribute(0)
            val attrName = new BytePointer(key)
            attr.set_name(attrName)
            attr.set_type(AttributeProto.INTS)
            (0 until x.size).foreach(y => attr.add_ints(x(y).toLong))
          }
          case None =>
        }
    }

    //TODO: Don't take tensor here
    def addInput[A](input: A, inputName: String): Unit = {
      input match {
        case tensorOpt: Option[Tensor[Any]] => {
          tensorOpt match {
            case Some(y) => node.add_input(inputName)
            case None    =>
          }
        }
        /*
        case tensorOpt: Seq[Option[Tensor[Any]]] => {
          tensorOpt.foreach { x =>
            x match {
              case tensorOpt: Option[Tensor[Any]] => {
                tensorOpt match {
                  case Some(y) => node.add_input(inputName)
                  case None    =>
                }
              }
            }
          }
        }
        */
        case _ => ??? //TODO: Handle non-tensors / don't assume tensor here

      }

    }
    //TODO: fix names
    addInput(inputs._1, "A")
    addInput(inputs._2, "B")
    addInput(inputs._3, "C")
    addInput(inputs._4, "D")
    addInput(inputs._5, "E")
    addInput(inputs._6, "F")
    addInput(inputs._7, "G")
    addInput(inputs._8, "H")
    addInput(inputs._9, "I")

    handleAttrs

    return node
  }

  protected def addInputToGraph[A](input: A, inputName: String, graph: GraphProto): Unit = {

    input match {
      case tensorOpt: Option[Tensor[_]] => {
        tensorOpt match {
          case Some(tens) => {

            val elemType = tens._1 match {
              case f: Array[Float] => TensorProto.FLOAT
              case i: Array[Int]   => TensorProto.INT32
              case l: Array[Long]  => TensorProto.INT64
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
          case None =>
        }

      }
      /*
      case tensorOpt: Seq[Option[Tensor[_]]] => {
        tensorOpt.foreach { x =>
          x match {
            //duplicated
            case Some(tens) => {

              val elemType = tens._1 match {
                case f: Array[Float] => TensorProto.FLOAT
                case i: Array[Int]   => TensorProto.INT32
                case l: Array[Long]  => TensorProto.INT64
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

                inputDim.set_dim_value(x)

              }
            }
            case None =>
          }
        }
      }
      */
    }
  }

  def opToONNXBytes[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag
  ](
      name: String,
      opName: String,
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
      outName: String,
      attrs: Map[String, Any]
  ): Array[Byte] = {

    val model = (new ModelProto).New()
    val graph = new org.bytedeco.onnx.GraphProto
    model.set_producer_name("ONNX-Scala")
    graph.set_name(name)

    //TODO: pass real names
    val origNode = opToNode(name, opName, inputs, outName, attrs)

    val node = graph.add_node
    node.MergeFrom(origNode)

    origNode.close
    model.set_allocated_graph(graph)
    model.set_ir_version(3)

    model.add_opset_import
    model.opset_import(0).set_version(8)

    val outputValueInfo = graph.add_output

    outputValueInfo.set_name(outName)

    outputValueInfo.mutable_type
    outputValueInfo.`type`.mutable_tensor_type
    outputValueInfo.`type`.tensor_type.set_elem_type(1)

    addInputToGraph(inputs._1, "A", graph)
    addInputToGraph(inputs._2, "B", graph)
    addInputToGraph(inputs._3, "C", graph)
    addInputToGraph(inputs._4, "D", graph)
    addInputToGraph(inputs._5, "E", graph)
    addInputToGraph(inputs._6, "F", graph)
    addInputToGraph(inputs._7, "G", graph)
    addInputToGraph(inputs._8, "H", graph)
    addInputToGraph(inputs._9, "I", graph)

    val modelString = model.SerializeAsString

    model.close
    val modelStringBytes = modelString.getStringBytes
    modelString.close

    (modelStringBytes)
  }

  override def close(): Unit = {
    scope.close
  }

}
