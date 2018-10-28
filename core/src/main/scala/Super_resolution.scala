package org.emergentorder.onnx

import org.emergentorder.onnx.UnionType._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import singleton.ops._
import scala.language.higherKinds

trait Super_resolution {
  val Conv: Conv
  val Add: Add
  val Relu: Relu
  val Reshape: Reshape
  val Transpose: Transpose
  val dataSource: DataSource
  def program[T : (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check:Numeric:ClassTag:Field]: List[Tensor[T]]  = 
    for {
      node1 <- List(dataSource.inputData[T])
      node2 <- List( dataSource.getParams[T]("2"))
      node3 <- List( dataSource.getParams[T]("3"))
      node4 <- List( dataSource.getParams[T]("4"))
      node5 <- List( dataSource.getParams[T]("5"))
      node6 <- List( dataSource.getParams[T]("6"))
      node7 <- List( dataSource.getParams[T]("7"))
      node8 <- List( dataSource.getParams[T]("8"))
      node9 <- List( dataSource.getParams[T]("9"))
      node11 <- List(Conv.Conv1[T]("11", Some(node1), Some("1"),Some(node2), Some("2"),kernel_shape = Some((Array(5,5))),strides = Some((Array(1,1))),pads = Some((Array(2,2,2,2))),dilations = Some((Array(1,1))),group = Some((1))))
      node12 <- List(Add.Add1[T]("12", Some(node11), Some("11"),Some(node3), Some("3"),broadcast = Some((1)),axis = Some((1))))
      node13 <- List(Relu.Relu1[T]("13", Some(node12), Some("12")))
      node15 <- List(Conv.Conv1[T]("15", Some(node13), Some("13"),Some(node4), Some("4"),kernel_shape = Some((Array(3,3))),strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),dilations = Some((Array(1,1))),group = Some((1))))
      node16 <- List(Add.Add1[T]("16", Some(node15), Some("15"),Some(node5), Some("5"),broadcast = Some((1)),axis = Some((1))))
      node17 <- List(Relu.Relu1[T]("17", Some(node16), Some("16")))
      node19 <- List(Conv.Conv1[T]("19", Some(node17), Some("17"),Some(node6), Some("6"),kernel_shape = Some((Array(3,3))),strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),dilations = Some((Array(1,1))),group = Some((1))))
      node20 <- List(Add.Add1[T]("20", Some(node19), Some("19"),Some(node7), Some("7"),broadcast = Some((1)),axis = Some((1))))
      node21 <- List(Relu.Relu1[T]("21", Some(node20), Some("20")))
      node23 <- List(Conv.Conv1[T]("23", Some(node21), Some("21"),Some(node8), Some("8"),kernel_shape = Some((Array(3,3))),strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),dilations = Some((Array(1,1))),group = Some((1))))
      node24 <- List(Add.Add1[T]("24", Some(node23), Some("23"),Some(node9), Some("9"),broadcast = Some((1)),axis = Some((1))))
      node25 <- List(Reshape.Reshape1[T]("25", Some(node24), Some("24"),shape = Some((Array(1,1,3,3,224,224)))))
      node26 <- List(Transpose.Transpose1[T]("26", Some(node25), Some("25"),perm = Some((Array(0,1,4,2,5,3)))))
      node27 <- List(Reshape.Reshape1[T]("27", Some(node26), Some("26"),shape = Some((Array(1,1,672,672)))))
    } yield (node27)
}
