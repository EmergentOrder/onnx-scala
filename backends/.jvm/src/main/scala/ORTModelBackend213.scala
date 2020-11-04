package org.emergentorder.onnx.backends

import scala.reflect.ClassTag
import scala.language.existentials
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.indexer.FloatIndexer
import ai.onnxruntime._
import scala.jdk.CollectionConverters._
import scala.language.existentials
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx._

//TODO: Clean up, remove asInstaceOf, multiple inputs, etc.
class ORTModelBackend(onnxBytes: Array[Byte])
    extends Model(onnxBytes)
    with ORTOperatorBackend
    with AutoCloseable {

  def getInputAndOutputNodeNamesAndDims(sess: OrtSession) = {
    val input_node_names = session.getInputNames

    val inputNodeDims = session.getInputInfo.values.asScala.map(_.getInfo.asInstanceOf[TensorInfo].getShape)

    val output_node_names = session.getOutputNames
   
    //Warn: conversion from unordered set to ordered list
      (input_node_names.asScala.toList, inputNodeDims.toArray, output_node_names.asScala.toList)
  }

  val session = getSession(onnxBytes)

  val allNodeNamesAndDims = getInputAndOutputNodeNamesAndDims(session)

  override def fullModel[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag,
      T9: ClassTag,
      T10: ClassTag,
      T11: ClassTag,
      T12: ClassTag,
      T13: ClassTag,
      T14: ClassTag,
      T15: ClassTag,
      T16: ClassTag,
      T17: ClassTag
  ](
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8]
  ): (T9) = {

    val inputTensors = Array(
      getInput(inputs._1),
      getInput(inputs._2),
      getInput(inputs._3),
      getInput(inputs._4),
      getInput(inputs._5),
      getInput(inputs._6),
      getInput(inputs._7),
      getInput(inputs._8),
      getInput(inputs._9)
    ).flatten

    val output = runModel(
      session,
      inputTensors,
      allNodeNamesAndDims._1,
      allNodeNamesAndDims._3
    )
//    val outputPointer = out.get(0).GetTensorMutableDataFloat().capacity(inputs.GetTensorTypeAndShapeInfo().GetElementCount());

//    println(outputPointer.get(0).IsTensor())

    output.asInstanceOf[T9]
  }

  def getInput[T: ClassTag](
      input: T
  ): Option[OnnxTensor] = {
    input match {
      case tensorOpt: Option[Tensor[Any]] => getTensor(tensorOpt)
      case tensor: Tensor[Any] => {
        getTensor(Some(tensor))
      }
    }
  }

  override def close(): Unit = {
//    executable.close
//    super.close
  }
}
