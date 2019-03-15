package org.emergentorder.onnx.backends

import java.io.PrintWriter;
import java.io.File;
import java.io.FileInputStream;
import java.nio.file._

import scala.{specialized => sp}
import scala.collection.mutable.{Map => MMap};
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.Numeric
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex


import org.emergentorder.onnx._
import org.emergentorder.onnx.UnionType._
import org.bytedeco.javacpp._;
import org.bytedeco.javacpp.onnx.ModelProto;
import org.bytedeco.javacpp.onnx.ParseProtoFromBytes;
import org.bytedeco.javacpp.onnx.MessageLite;
import org.bytedeco.javacpp.onnx.NodeProto;
import org.bytedeco.javacpp.ngraph.import_onnx_model
import org.bytedeco.javacpp.ngraph.Backend
import org.bytedeco.javacpp.ngraph.f32


class NGraphBackend extends Conv with Relu with MaxPool with Concat with Dropout with AveragePool with Reshape{
//with DataSource
   
  def Conv1[@sp T: Numeric: ClassTag](name: String,
                                        auto_pad: Option[(String)] = None,
                                        dilations: Option[(Array[Int])] = None,
                                        group: Option[(Int)] = None,
                                        kernel_shape: Option[(Array[Int])] =
                                          None,
                                        pads: Option[(Array[Int])] = None,
                                        strides: Option[(Array[Int])] = None,
                                        X: Option[Tensor[T]],
                                        W: Option[Tensor[T]],
                                        B: Option[Tensor[T]] = None)(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : (Tensor[T]) = {

 val map: Map[String, Any] = Map("auto_pad" -> auto_pad,
   "dilations" -> dilations,
   "group" -> group,
   "kernel_shape" -> kernel_shape,
   "pads" -> pads,
   "strides" -> strides)
   
    unaryOrBinaryOp(name, "Conv", X, W, map) //TODO: B
      }

 
/*
    val inputs: Seq[String] = node.input
          assert (inputs.size == 2 || inputs.size == 3, s"number of inputs of a conv node should always be 2 or 3, got ${inputs.size}")

          val outputs: Seq[String] = node.output
          assert (outputs.size == 1, "number of output of a conv node should always be 1")

          val attributes: Seq[onnx_ml.AttributeProto] = node.attribute
          convNode(inputs, outputs.head, getConvMaxPAvPAttr(attributes))
          */


  def Dropout7[@sp T : Numeric:ClassTag](name: String,ratio : Option[(Float)],data: Option[Tensor[T]])
//      (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    
      : (Tensor[T], Tensor[T]) = {
        val map: Map[String, Any] = Map("ratio" -> ratio)
        (unaryOrBinaryOp(name, "Dropout", data, None, map), null) //TODO: optional output
      }


  def Relu1[@sp T : Numeric : ClassTag](name: String,
                                        consumed_inputs: Option[(Array[Int])] // = None //Default args don't work
                                        ,
                                        X: Option[Tensor[T]])
  (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : (Tensor[T]) = ???

  def Relu6[@sp T : Numeric : ClassTag](name: String, X: Option[Tensor[T]])
//  (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : (Tensor[T]) = {
        unaryOrBinaryOpNoAttrs(name, "Relu", X, None)
      }

  def unaryOrBinaryOpNoAttrs[@sp T : ClassTag](name: String, opName: String, X: Option[Tensor[T]], W: Option[Tensor[T]]) : (Tensor[T]) =
    unaryOrBinaryOp(name, opName, X, W, Map())
 
  def unaryOrBinaryOp[@sp T : ClassTag](name: String, opName: String, X: Option[Tensor[T]], W: Option[Tensor[T]], attrs: Map[String, Any]) 
  //(
//        implicit evT:  (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
   //       Float] TypeOr Complex[Double])#check[T])
      : (Tensor[T]) = {

        val model = new ModelProto()
        val graph = new org.bytedeco.javacpp.onnx.GraphProto
        val node = graph.add_node
    
//        println("name : " + name)

        model.set_producer_name("backend-test")
        graph.set_name(name)
        node.set_name(name)
        node.set_op_type(opName)
        node.add_output("Y")

        //TODO: handle attrs here


          def addInput(input: Option[Tensor[T]], inputName: String){

    input match {
          case Some(tens) => {
            node.add_input(inputName)

            val inputValueInfo = graph.add_input

            inputValueInfo.set_name(inputName)
            inputValueInfo.mutable_type
            inputValueInfo.`type`.mutable_tensor_type
            inputValueInfo.`type`.tensor_type.set_elem_type(1) //missing : type

            val dims = tens._2
            inputValueInfo.`type`.tensor_type.mutable_shape
            dims.foreach{x =>
              val inputDim = inputValueInfo.`type`.tensor_type.shape.add_dim

//              inputDim.set_dim_param("NAME?")
              inputDim.set_dim_value(x)
//              println("in dim val " + x)
            }
          }
          case None =>

          }

  }


        addInput(W, "w")

        addInput(X, "X")

        val outputValueInfo = graph.add_output


        outputValueInfo.set_name("Y")

 
        outputValueInfo.mutable_type
        outputValueInfo.`type`.mutable_tensor_type
        outputValueInfo.`type`.tensor_type.set_elem_type(1)

     
        X match {
          case Some(tens) => {
            val dims = tens._2
            outputValueInfo.`type`.tensor_type.mutable_shape
            dims.foreach{x =>  
              val outputDim = outputValueInfo.`type`.tensor_type.shape.add_dim
//              outputDim.set_dim_param("NAME?")
              outputDim.set_dim_value(x)
//               println("Out dim val " +x)
            }
          }
          case None =>
           
          }


        model.set_allocated_graph(graph)
        model.set_ir_version(3)

        model.add_opset_import
        model.opset_import(0).set_version(6)
        val modelString = model.SerializeAsString.getString
//        println(modelString)
//        println(modelString.size)
//        println("graph string size  " + model.graph.SerializeAsString.getString.size)
//        println("node string size " + model.graph.node(0).SerializeAsString.getString.size)
//        println("value info DIM size" + inputValueInfo.`type`.tensor_type.shape.dim_size)
//        println("value info DIM size" + outputValueInfo.`type`.tensor_type.shape.dim_size)
//        println("value info tring size" + outputValueInfo.`type`.tensor_type.SerializeAsString.getString.size)
        val ngraphFunc = import_onnx_model(modelString)


        val ngraphBackend = Backend.create("CPU") 


        val shape:org.bytedeco.javacpp.ngraph.Shape = X match {
          case Some(tens) => {
            val dims = tens._2
            val s =   new org.bytedeco.javacpp.ngraph.Shape(tens._2.size)
            s.resize(tens._2.size)
            val longShape = tens._2.map{x => 
//            println("Shape val: " + x)
            x.toLong}
            s.put(longShape: _*)
            s
          }
          case None =>  new org.bytedeco.javacpp.ngraph.Shape

          }
       
        val inputTens: FloatPointer =  X match {
          case Some(tens) => {
            new FloatPointer(tens._1.asInstanceOf[Array[Float]]: _*)
          }
          case None => new FloatPointer
        }
        val input = ngraphBackend.create_tensor(f32, shape, inputTens)
        val output = ngraphBackend.create_tensor(f32, shape)
        val inputVector = new org.bytedeco.javacpp.ngraph.NgraphTensorVector(input)
        val outputVector = new org.bytedeco.javacpp.ngraph.NgraphTensorVector(output)

//        println("sizes " + shape)
          //+ inputVector.size + " " + outputVector.size)
        ngraphBackend.compile(ngraphFunc)
        ngraphBackend.call(ngraphFunc, outputVector, inputVector)
        //convert result to onnx-scala Tensor
        
        val arraySize = (0 until shape.size.toInt).map{ x =>
          shape.get(x).toInt}.reduceLeft(_ * _)

        val fp = new FloatPointer(arraySize)
        outputVector.get(0).read(fp, 0, arraySize*4)

        val fb = fp.asByteBuffer.asFloatBuffer
        val fa = new Array[T](arraySize.toInt)
        (0 until fb.capacity).map{ x =>
          fa.update(x,fb.get(x).asInstanceOf[T]) //unsafe : asInstanceOf
        }
       
//        println(shape)
//        println(fa(2))
        val shapeArray = new Array[Int](shape.size.toInt)
        (0 until shape.size.toInt).map{ x =>
          shapeArray.update(x,shape.get(x).toInt)
        }

        val result: Tensor[T] = (fa,shapeArray) 
      
        result
      }

      /*
  def addInput(input: Option[Tensor[T]], inputName: String){

    input match {
          case Some(tens) => {
            node.add_input(inputName)

            val inputValueInfo = graph.add_input

            inputValueInfo.set_name(inputName)
            inputValueInfo.mutable_type
            inputValueInfo.`type`.mutable_tensor_type
            inputValueInfo.`type`.tensor_type.set_elem_type(1) //missing : type

            val dims = tens._2
            inputValueInfo.`type`.tensor_type.mutable_shape
            dims.foreach{x =>
              val inputDim = inputValueInfo.`type`.tensor_type.shape.add_dim

//              inputDim.set_dim_param("NAME?")
              inputDim.set_dim_value(x)
//              println("in dim val " + x)
            }
          }
          case None =>

          }

  }

*/

  def MaxPool1[@sp T: Numeric: ClassTag](name: String,
                                           auto_pad: Option[(String)] = None,
                                           kernel_shape: Option[(Array[Int])],
                                           pads: Option[(Array[Int])] = None,
                                           strides: Option[(Array[Int])] = None,
                                           X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : (Tensor[T]) = ???

  def MaxPool8[@sp T: Numeric: ClassTag, @sp I: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        storage_order: Option[(Int)] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
        evI: (UNil TypeOr Long)#check[I]): (Tensor[T], Tensor[I]) = {
          val map: Map[String, Any] = Map("auto_pad" -> auto_pad, "kernel_shape" -> kernel_shape,
            "pads" -> pads,
            "storage_order" -> storage_order,
            "strides" -> strides)

          (unaryOrBinaryOp(name, "MaxPool", X, None, map), null) //TODO:optional output
        }


  def Concat4[@sp T: Numeric: ClassTag](name: String,
                                          axis: Option[(Int)],
                                          inputs: Seq[Option[Tensor[T]]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float] TypeOr Complex[Double])#check[T]): (Tensor[T]) = {
            val map: Map[String, Any] = Map("axis" -> axis)
            val X = inputs(0)
            val W = inputs(1)
            unaryOrBinaryOp(name, "Concat", X, W, map)
                //TODO: > 2 inputs
          }


  def Dropout1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        is_test: Option[(Int)] = None,
        ratio: Option[(Float)] = None,
        data: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : (Tensor[T], Tensor[T]) = ???

    def Dropout6[@sp T: Numeric: ClassTag](name: String,
                                           is_test: Option[(Int)] = None,
                                           ratio: Option[(Float)] = None,
                                           data: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : (Tensor[T], Tensor[T]) = ???


    def AveragePool1[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : (Tensor[T]) = ???

    def AveragePool7[@sp T: Numeric: ClassTag](
        name: String,
        auto_pad: Option[(String)] = None,
        count_include_pad: Option[(Int)] = None,
        kernel_shape: Option[(Array[Int])],
        pads: Option[(Array[Int])] = None,
        strides: Option[(Array[Int])] = None,
        X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : (Tensor[T]) = {
        val map: Map[String, Any] = Map("auto_pad" -> auto_pad,
          "count_include_pad" -> count_include_pad,
          "kernel_shape" -> kernel_shape,
          "pads" -> pads,
          "strides" -> strides)

        unaryOrBinaryOp(name, "AveragePool", X, None, map)
        }
          def Reshape1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        shape: Option[(Array[Int])] = None,
        data: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float] TypeOr Complex[Double])#check[T]): (Tensor[T]) = ???

    def Reshape5[@sp T: Numeric: ClassTag](name: String,
                                           data: Option[Tensor[T]],
                                           shape: Option[Tensor[Long]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
          Float] TypeOr Complex[Double])#check[T]): (Tensor[T]) = {
            val map: Map[String, Any] = Map("shape" -> shape)
            unaryOrBinaryOp(name, "Reshape", data, None, map) 
          }


/*
                                            abstract class Node
    case class convNode(inputs: Seq[String], output: String, attributes: Map[String, Seq[Int]]) extends Node
    case class bnNode(inputs: Seq[String], output: String, attributeMap: Map[String, Float]) extends Node
    case class sumNode(inputs: Seq[String], output: String) extends Node
    case class reluNode(input: String, output: String) extends Node
    case class maxpoolNode(input: String, output: String, attributes: Map[String, Seq[Int]]) extends Node
    case class averagePoolNode(input: String, output: String, attributes: Map[String, Seq[Int]]) extends Node
    case class concatNode(inputs: Seq[String], output: String, axis: Int) extends Node
    case class dropoutNode(input: String, outputs: Seq[String], ratio: Float) extends Node
    case class globalAveragePoolNode(input: String, output: String) extends Node
    case class softmaxNode(input: String, output: String) extends Node
    case class reshapeNode(inputs: Seq[String], output: String) extends Node
    case class gemmNode(inputs: Seq[String], output: String, attInts: Map[String, Int], attFloats: Map[String, Float]) extends Node
    case class flattenNode(input: String, output: String, axis: Int) extends Node
    case class addNode(inputs: Seq[String], output: String) extends Node
    case class padNode(input: String, output: String, mode: String, pads: List[Int], value: Float) extends Node
    case class shapeNode(input: String, output: String) extends Node
    case class sliceNode(input: String, output: String, attMap: Map[String, List[Int]]) extends Node
    case class squeezeNode(input: String, output: String, axes: List[Int]) extends Node
    case class unsqueezeNode(input: String, output: String, axes: List[Int]) extends Node
    case class constantNode(output: String, data: Float) extends Node
*/

}
