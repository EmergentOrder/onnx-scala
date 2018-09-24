package org.emergentorder.onnx

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
  def program[T <: Float16 | Float | Double:Numeric:ClassTag:Field]: List[Tensor[T]]  = 
    for {
      node1 <- List(dataSource.inputData[T])
      node8 <- List( dataSource.getParams[T]("8"))
      node4 <- List( dataSource.getParams[T]("4"))
      node9 <- List( dataSource.getParams[T]("9"))
      node5 <- List( dataSource.getParams[T]("5"))
      node6 <- List( dataSource.getParams[T]("6"))
      node2 <- List( dataSource.getParams[T]("2"))
      node7 <- List( dataSource.getParams[T]("7"))
      node3 <- List( dataSource.getParams[T]("3"))
      node11 <- List(Conv.Conv1[T]("11", node1, "1",node2, "2",kernel_shape = Some((Array(5,5))),strides = Some((Array(1,1))),pads = Some((Array(2,2,2,2))),dilations = Some((Array(1,1))),group = Some((1))))
      node12 <- List(Add.Add1[T]("12", node11, "11",node3, "3",broadcast = Some((1)),axis = Some((1))))
      node13 <- List(Relu.Relu1[T]("13", node12, "12"))
      node15 <- List(Conv.Conv1[T]("15", node13, "13",node4, "4",kernel_shape = Some((Array(3,3))),strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),dilations = Some((Array(1,1))),group = Some((1))))
      node16 <- List(Add.Add1[T]("16", node15, "15",node5, "5",broadcast = Some((1)),axis = Some((1))))
      node17 <- List(Relu.Relu1[T]("17", node16, "16"))
      node19 <- List(Conv.Conv1[T]("19", node17, "17",node6, "6",kernel_shape = Some((Array(3,3))),strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),dilations = Some((Array(1,1))),group = Some((1))))
      node20 <- List(Add.Add1[T]("20", node19, "19",node7, "7",broadcast = Some((1)),axis = Some((1))))
      node21 <- List(Relu.Relu1[T]("21", node20, "20"))
      node23 <- List(Conv.Conv1[T]("23", node21, "21",node8, "8",kernel_shape = Some((Array(3,3))),strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),dilations = Some((Array(1,1))),group = Some((1))))
      node24 <- List(Add.Add1[T]("24", node23, "23",node9, "9",broadcast = Some((1)),axis = Some((1))))
      node25 <- List(Reshape.Reshape1[T]("25", node24, "24",shape = Some((Array(1,1,3,3,224,224)))))
      node26 <- List(Transpose.Transpose1[T]("26", node25, "25",perm = Some((Array(0,1,4,2,5,3)))))
      node27 <- List(Reshape.Reshape1[T]("27", node26, "26",shape = Some((Array(1,1,672,672)))))
    } yield (node27)
}
