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
      T: ClassTag
      ](
      name: String,
      opName: String,
      inputs: Option[NonEmptyTuple],
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
    //Dummy names
    inputs match{
      case Some(x) => {
        addInput(x(0), "A")
        addInput(x(1), "B")
        addInput(x(2), "C")
        addInput(x(3), "D")
        addInput(x(4), "E")
        addInput(x(5), "F")
        addInput(x(6), "G")
        addInput(x(7), "H")
       addInput(x(8), "I")
      }
      case None =>
    }
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
      T: ClassTag
  ](
      name: String,
      opName: String,
      inputs: Option[NonEmptyTuple],
      outName: String,
      attrs: Map[String, Any]
  ): Array[Byte] = {

    val model = (new ModelProto).New()
    val graph = new org.bytedeco.onnx.GraphProto
    model.set_producer_name("ONNX-Scala")
    graph.set_name(name)

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

    //Dummy names
    inputs match{
      case Some(x) => {
        addInputToGraph(x(0), "A", graph)
        addInputToGraph(x(1), "B", graph)
        addInputToGraph(x(2), "C", graph)
        addInputToGraph(x(3), "D", graph)
        addInputToGraph(x(4), "E", graph) 
        addInputToGraph(x(5), "F", graph)
        addInputToGraph(x(6), "G", graph)
        addInputToGraph(x(7), "H", graph)
        addInputToGraph(x(8), "I", graph)
      }
      case None =>
    }

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
