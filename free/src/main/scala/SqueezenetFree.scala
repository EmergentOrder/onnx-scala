package org.emergentorder.onnxFree

import freestyle.free._
import freestyle.free.implicits._
import cats.free.{ Free, FreeApplicative } 
import cats.implicits._ 
import cats.effect.IO
import org.emergentorder.onnx._
import org.emergentorder.onnx.UnionType._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import scala.language.higherKinds

@module trait SqueezenetFree {
  val ConvFree: ConvFree
  val ReluFree: ReluFree
  val MaxPoolFree: MaxPoolFree
  val ConcatFree: ConcatFree
  val DropoutFree: DropoutFree
  val GlobalAveragePoolFree: GlobalAveragePoolFree
  val SoftmaxFree: SoftmaxFree
  val dataSource: DataSourceFree
  def program: FS.Seq[Tensor[Float]]  = 
    for {
      nodedata_0 <- dataSource.inputDataFree[Float]
      nodeconv10_b_0 <-  dataSource.getParamsFree[Float]("conv10_b_0")
      nodeconv10_w_0 <-  dataSource.getParamsFree[Float]("conv10_w_0")
      nodeconv1_b_0 <-  dataSource.getParamsFree[Float]("conv1_b_0")
      nodeconv1_w_0 <-  dataSource.getParamsFree[Float]("conv1_w_0")
      nodefire2_expand1x1_b_0 <-  dataSource.getParamsFree[Float]("fire2_expand1x1_b_0")
      nodefire2_expand1x1_w_0 <-  dataSource.getParamsFree[Float]("fire2_expand1x1_w_0")
      nodefire2_expand3x3_b_0 <-  dataSource.getParamsFree[Float]("fire2_expand3x3_b_0")
      nodefire2_expand3x3_w_0 <-  dataSource.getParamsFree[Float]("fire2_expand3x3_w_0")
      nodefire2_squeeze1x1_b_0 <-  dataSource.getParamsFree[Float]("fire2_squeeze1x1_b_0")
      nodefire2_squeeze1x1_w_0 <-  dataSource.getParamsFree[Float]("fire2_squeeze1x1_w_0")
      nodefire3_expand1x1_b_0 <-  dataSource.getParamsFree[Float]("fire3_expand1x1_b_0")
      nodefire3_expand1x1_w_0 <-  dataSource.getParamsFree[Float]("fire3_expand1x1_w_0")
      nodefire3_expand3x3_b_0 <-  dataSource.getParamsFree[Float]("fire3_expand3x3_b_0")
      nodefire3_expand3x3_w_0 <-  dataSource.getParamsFree[Float]("fire3_expand3x3_w_0")
      nodefire3_squeeze1x1_b_0 <-  dataSource.getParamsFree[Float]("fire3_squeeze1x1_b_0")
      nodefire3_squeeze1x1_w_0 <-  dataSource.getParamsFree[Float]("fire3_squeeze1x1_w_0")
      nodefire4_expand1x1_b_0 <-  dataSource.getParamsFree[Float]("fire4_expand1x1_b_0")
      nodefire4_expand1x1_w_0 <-  dataSource.getParamsFree[Float]("fire4_expand1x1_w_0")
      nodefire4_expand3x3_b_0 <-  dataSource.getParamsFree[Float]("fire4_expand3x3_b_0")
      nodefire4_expand3x3_w_0 <-  dataSource.getParamsFree[Float]("fire4_expand3x3_w_0")
      nodefire4_squeeze1x1_b_0 <-  dataSource.getParamsFree[Float]("fire4_squeeze1x1_b_0")
      nodefire4_squeeze1x1_w_0 <-  dataSource.getParamsFree[Float]("fire4_squeeze1x1_w_0")
      nodefire5_expand1x1_b_0 <-  dataSource.getParamsFree[Float]("fire5_expand1x1_b_0")
      nodefire5_expand1x1_w_0 <-  dataSource.getParamsFree[Float]("fire5_expand1x1_w_0")
      nodefire5_expand3x3_b_0 <-  dataSource.getParamsFree[Float]("fire5_expand3x3_b_0")
      nodefire5_expand3x3_w_0 <-  dataSource.getParamsFree[Float]("fire5_expand3x3_w_0")
      nodefire5_squeeze1x1_b_0 <-  dataSource.getParamsFree[Float]("fire5_squeeze1x1_b_0")
      nodefire5_squeeze1x1_w_0 <-  dataSource.getParamsFree[Float]("fire5_squeeze1x1_w_0")
      nodefire6_expand1x1_b_0 <-  dataSource.getParamsFree[Float]("fire6_expand1x1_b_0")
      nodefire6_expand1x1_w_0 <-  dataSource.getParamsFree[Float]("fire6_expand1x1_w_0")
      nodefire6_expand3x3_b_0 <-  dataSource.getParamsFree[Float]("fire6_expand3x3_b_0")
      nodefire6_expand3x3_w_0 <-  dataSource.getParamsFree[Float]("fire6_expand3x3_w_0")
      nodefire6_squeeze1x1_b_0 <-  dataSource.getParamsFree[Float]("fire6_squeeze1x1_b_0")
      nodefire6_squeeze1x1_w_0 <-  dataSource.getParamsFree[Float]("fire6_squeeze1x1_w_0")
      nodefire7_expand1x1_b_0 <-  dataSource.getParamsFree[Float]("fire7_expand1x1_b_0")
      nodefire7_expand1x1_w_0 <-  dataSource.getParamsFree[Float]("fire7_expand1x1_w_0")
      nodefire7_expand3x3_b_0 <-  dataSource.getParamsFree[Float]("fire7_expand3x3_b_0")
      nodefire7_expand3x3_w_0 <-  dataSource.getParamsFree[Float]("fire7_expand3x3_w_0")
      nodefire7_squeeze1x1_b_0 <-  dataSource.getParamsFree[Float]("fire7_squeeze1x1_b_0")
      nodefire7_squeeze1x1_w_0 <-  dataSource.getParamsFree[Float]("fire7_squeeze1x1_w_0")
      nodefire8_expand1x1_b_0 <-  dataSource.getParamsFree[Float]("fire8_expand1x1_b_0")
      nodefire8_expand1x1_w_0 <-  dataSource.getParamsFree[Float]("fire8_expand1x1_w_0")
      nodefire8_expand3x3_b_0 <-  dataSource.getParamsFree[Float]("fire8_expand3x3_b_0")
      nodefire8_expand3x3_w_0 <-  dataSource.getParamsFree[Float]("fire8_expand3x3_w_0")
      nodefire8_squeeze1x1_b_0 <-  dataSource.getParamsFree[Float]("fire8_squeeze1x1_b_0")
      nodefire8_squeeze1x1_w_0 <-  dataSource.getParamsFree[Float]("fire8_squeeze1x1_w_0")
      nodefire9_expand1x1_b_0 <-  dataSource.getParamsFree[Float]("fire9_expand1x1_b_0")
      nodefire9_expand1x1_w_0 <-  dataSource.getParamsFree[Float]("fire9_expand1x1_w_0")
      nodefire9_expand3x3_b_0 <-  dataSource.getParamsFree[Float]("fire9_expand3x3_b_0")
      nodefire9_expand3x3_w_0 <-  dataSource.getParamsFree[Float]("fire9_expand3x3_w_0")
      nodefire9_squeeze1x1_b_0 <-  dataSource.getParamsFree[Float]("fire9_squeeze1x1_b_0")
      nodefire9_squeeze1x1_w_0 <-  dataSource.getParamsFree[Float]("fire9_squeeze1x1_w_0")
      nodeconv1_1 <- ConvFree.Conv1Free("conv1_1" ,strides = Some((Array(2,2))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(3,3))),X = Some(nodedata_0),W = Some(nodeconv1_w_0),B = Some(nodeconv1_b_0))
      nodeconv1_2 <- ReluFree.Relu6Free("conv1_2" ,X = Some(nodeconv1_1))
      nodepool1_1 <- MaxPoolFree.MaxPool1Free("pool1_1" ,strides = Some((Array(2,2))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(3,3))),X = Some(nodeconv1_2))
      nodefire2_squeeze1x1_1 <- ConvFree.Conv1Free("fire2_squeeze1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodepool1_1),W = Some(nodefire2_squeeze1x1_w_0),B = Some(nodefire2_squeeze1x1_b_0))
      nodefire2_squeeze1x1_2 <- ReluFree.Relu6Free("fire2_squeeze1x1_2" ,X = Some(nodefire2_squeeze1x1_1))
      nodefire2_expand1x1_1 <- ConvFree.Conv1Free("fire2_expand1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire2_squeeze1x1_2),W = Some(nodefire2_expand1x1_w_0),B = Some(nodefire2_expand1x1_b_0))
      nodefire2_expand1x1_2 <- ReluFree.Relu6Free("fire2_expand1x1_2" ,X = Some(nodefire2_expand1x1_1))
      nodefire2_expand3x3_1 <- ConvFree.Conv1Free("fire2_expand3x3_1" ,strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),kernel_shape = Some((Array(3,3))),X = Some(nodefire2_squeeze1x1_2),W = Some(nodefire2_expand3x3_w_0),B = Some(nodefire2_expand3x3_b_0))
      nodefire2_expand3x3_2 <- ReluFree.Relu6Free("fire2_expand3x3_2" ,X = Some(nodefire2_expand3x3_1))
      nodefire2_concat_1 <- ConcatFree.Concat4Free("fire2_concat_1" ,axis = Some((1)),inputs = Seq(Some(nodefire2_expand1x1_2),Some(nodefire2_expand3x3_2)))
      nodefire3_squeeze1x1_1 <- ConvFree.Conv1Free("fire3_squeeze1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire2_concat_1),W = Some(nodefire3_squeeze1x1_w_0),B = Some(nodefire3_squeeze1x1_b_0))
      nodefire3_squeeze1x1_2 <- ReluFree.Relu6Free("fire3_squeeze1x1_2" ,X = Some(nodefire3_squeeze1x1_1))
      nodefire3_expand1x1_1 <- ConvFree.Conv1Free("fire3_expand1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire3_squeeze1x1_2),W = Some(nodefire3_expand1x1_w_0),B = Some(nodefire3_expand1x1_b_0))
      nodefire3_expand1x1_2 <- ReluFree.Relu6Free("fire3_expand1x1_2" ,X = Some(nodefire3_expand1x1_1))
      nodefire3_expand3x3_1 <- ConvFree.Conv1Free("fire3_expand3x3_1" ,strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),kernel_shape = Some((Array(3,3))),X = Some(nodefire3_squeeze1x1_2),W = Some(nodefire3_expand3x3_w_0),B = Some(nodefire3_expand3x3_b_0))
      nodefire3_expand3x3_2 <- ReluFree.Relu6Free("fire3_expand3x3_2" ,X = Some(nodefire3_expand3x3_1))
      nodefire3_concat_1 <- ConcatFree.Concat4Free("fire3_concat_1" ,axis = Some((1)),inputs = Seq(Some(nodefire3_expand1x1_2),Some(nodefire3_expand3x3_2)))
      nodepool3_1 <- MaxPoolFree.MaxPool1Free("pool3_1" ,strides = Some((Array(2,2))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(3,3))),X = Some(nodefire3_concat_1))
      nodefire4_squeeze1x1_1 <- ConvFree.Conv1Free("fire4_squeeze1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodepool3_1),W = Some(nodefire4_squeeze1x1_w_0),B = Some(nodefire4_squeeze1x1_b_0))
      nodefire4_squeeze1x1_2 <- ReluFree.Relu6Free("fire4_squeeze1x1_2" ,X = Some(nodefire4_squeeze1x1_1))
      nodefire4_expand1x1_1 <- ConvFree.Conv1Free("fire4_expand1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire4_squeeze1x1_2),W = Some(nodefire4_expand1x1_w_0),B = Some(nodefire4_expand1x1_b_0))
      nodefire4_expand1x1_2 <- ReluFree.Relu6Free("fire4_expand1x1_2" ,X = Some(nodefire4_expand1x1_1))
      nodefire4_expand3x3_1 <- ConvFree.Conv1Free("fire4_expand3x3_1" ,strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),kernel_shape = Some((Array(3,3))),X = Some(nodefire4_squeeze1x1_2),W = Some(nodefire4_expand3x3_w_0),B = Some(nodefire4_expand3x3_b_0))
      nodefire4_expand3x3_2 <- ReluFree.Relu6Free("fire4_expand3x3_2" ,X = Some(nodefire4_expand3x3_1))
      nodefire4_concat_1 <- ConcatFree.Concat4Free("fire4_concat_1" ,axis = Some((1)),inputs = Seq(Some(nodefire4_expand1x1_2),Some(nodefire4_expand3x3_2)))
      nodefire5_squeeze1x1_1 <- ConvFree.Conv1Free("fire5_squeeze1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire4_concat_1),W = Some(nodefire5_squeeze1x1_w_0),B = Some(nodefire5_squeeze1x1_b_0))
      nodefire5_squeeze1x1_2 <- ReluFree.Relu6Free("fire5_squeeze1x1_2" ,X = Some(nodefire5_squeeze1x1_1))
      nodefire5_expand1x1_1 <- ConvFree.Conv1Free("fire5_expand1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire5_squeeze1x1_2),W = Some(nodefire5_expand1x1_w_0),B = Some(nodefire5_expand1x1_b_0))
      nodefire5_expand1x1_2 <- ReluFree.Relu6Free("fire5_expand1x1_2" ,X = Some(nodefire5_expand1x1_1))
      nodefire5_expand3x3_1 <- ConvFree.Conv1Free("fire5_expand3x3_1" ,strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),kernel_shape = Some((Array(3,3))),X = Some(nodefire5_squeeze1x1_2),W = Some(nodefire5_expand3x3_w_0),B = Some(nodefire5_expand3x3_b_0))
      nodefire5_expand3x3_2 <- ReluFree.Relu6Free("fire5_expand3x3_2" ,X = Some(nodefire5_expand3x3_1))
      nodefire5_concat_1 <- ConcatFree.Concat4Free("fire5_concat_1" ,axis = Some((1)),inputs = Seq(Some(nodefire5_expand1x1_2),Some(nodefire5_expand3x3_2)))
      nodepool5_1 <- MaxPoolFree.MaxPool1Free("pool5_1" ,strides = Some((Array(2,2))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(3,3))),X = Some(nodefire5_concat_1))
      nodefire6_squeeze1x1_1 <- ConvFree.Conv1Free("fire6_squeeze1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodepool5_1),W = Some(nodefire6_squeeze1x1_w_0),B = Some(nodefire6_squeeze1x1_b_0))
      nodefire6_squeeze1x1_2 <- ReluFree.Relu6Free("fire6_squeeze1x1_2" ,X = Some(nodefire6_squeeze1x1_1))
      nodefire6_expand1x1_1 <- ConvFree.Conv1Free("fire6_expand1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire6_squeeze1x1_2),W = Some(nodefire6_expand1x1_w_0),B = Some(nodefire6_expand1x1_b_0))
      nodefire6_expand1x1_2 <- ReluFree.Relu6Free("fire6_expand1x1_2" ,X = Some(nodefire6_expand1x1_1))
      nodefire6_expand3x3_1 <- ConvFree.Conv1Free("fire6_expand3x3_1" ,strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),kernel_shape = Some((Array(3,3))),X = Some(nodefire6_squeeze1x1_2),W = Some(nodefire6_expand3x3_w_0),B = Some(nodefire6_expand3x3_b_0))
      nodefire6_expand3x3_2 <- ReluFree.Relu6Free("fire6_expand3x3_2" ,X = Some(nodefire6_expand3x3_1))
      nodefire6_concat_1 <- ConcatFree.Concat4Free("fire6_concat_1" ,axis = Some((1)),inputs = Seq(Some(nodefire6_expand1x1_2),Some(nodefire6_expand3x3_2)))
      nodefire7_squeeze1x1_1 <- ConvFree.Conv1Free("fire7_squeeze1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire6_concat_1),W = Some(nodefire7_squeeze1x1_w_0),B = Some(nodefire7_squeeze1x1_b_0))
      nodefire7_squeeze1x1_2 <- ReluFree.Relu6Free("fire7_squeeze1x1_2" ,X = Some(nodefire7_squeeze1x1_1))
      nodefire7_expand1x1_1 <- ConvFree.Conv1Free("fire7_expand1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire7_squeeze1x1_2),W = Some(nodefire7_expand1x1_w_0),B = Some(nodefire7_expand1x1_b_0))
      nodefire7_expand1x1_2 <- ReluFree.Relu6Free("fire7_expand1x1_2" ,X = Some(nodefire7_expand1x1_1))
      nodefire7_expand3x3_1 <- ConvFree.Conv1Free("fire7_expand3x3_1" ,strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),kernel_shape = Some((Array(3,3))),X = Some(nodefire7_squeeze1x1_2),W = Some(nodefire7_expand3x3_w_0),B = Some(nodefire7_expand3x3_b_0))
      nodefire7_expand3x3_2 <- ReluFree.Relu6Free("fire7_expand3x3_2" ,X = Some(nodefire7_expand3x3_1))
      nodefire7_concat_1 <- ConcatFree.Concat4Free("fire7_concat_1" ,axis = Some((1)),inputs = Seq(Some(nodefire7_expand1x1_2),Some(nodefire7_expand3x3_2)))
      nodefire8_squeeze1x1_1 <- ConvFree.Conv1Free("fire8_squeeze1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire7_concat_1),W = Some(nodefire8_squeeze1x1_w_0),B = Some(nodefire8_squeeze1x1_b_0))
      nodefire8_squeeze1x1_2 <- ReluFree.Relu6Free("fire8_squeeze1x1_2" ,X = Some(nodefire8_squeeze1x1_1))
      nodefire8_expand1x1_1 <- ConvFree.Conv1Free("fire8_expand1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire8_squeeze1x1_2),W = Some(nodefire8_expand1x1_w_0),B = Some(nodefire8_expand1x1_b_0))
      nodefire8_expand1x1_2 <- ReluFree.Relu6Free("fire8_expand1x1_2" ,X = Some(nodefire8_expand1x1_1))
      nodefire8_expand3x3_1 <- ConvFree.Conv1Free("fire8_expand3x3_1" ,strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),kernel_shape = Some((Array(3,3))),X = Some(nodefire8_squeeze1x1_2),W = Some(nodefire8_expand3x3_w_0),B = Some(nodefire8_expand3x3_b_0))
      nodefire8_expand3x3_2 <- ReluFree.Relu6Free("fire8_expand3x3_2" ,X = Some(nodefire8_expand3x3_1))
      nodefire8_concat_1 <- ConcatFree.Concat4Free("fire8_concat_1" ,axis = Some((1)),inputs = Seq(Some(nodefire8_expand1x1_2),Some(nodefire8_expand3x3_2)))
      nodefire9_squeeze1x1_1 <- ConvFree.Conv1Free("fire9_squeeze1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire8_concat_1),W = Some(nodefire9_squeeze1x1_w_0),B = Some(nodefire9_squeeze1x1_b_0))
      nodefire9_squeeze1x1_2 <- ReluFree.Relu6Free("fire9_squeeze1x1_2" ,X = Some(nodefire9_squeeze1x1_1))
      nodefire9_expand1x1_1 <- ConvFree.Conv1Free("fire9_expand1x1_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire9_squeeze1x1_2),W = Some(nodefire9_expand1x1_w_0),B = Some(nodefire9_expand1x1_b_0))
      nodefire9_expand1x1_2 <- ReluFree.Relu6Free("fire9_expand1x1_2" ,X = Some(nodefire9_expand1x1_1))
      nodefire9_expand3x3_1 <- ConvFree.Conv1Free("fire9_expand3x3_1" ,strides = Some((Array(1,1))),pads = Some((Array(1,1,1,1))),kernel_shape = Some((Array(3,3))),X = Some(nodefire9_squeeze1x1_2),W = Some(nodefire9_expand3x3_w_0),B = Some(nodefire9_expand3x3_b_0))
      nodefire9_expand3x3_2 <- ReluFree.Relu6Free("fire9_expand3x3_2" ,X = Some(nodefire9_expand3x3_1))
      nodefire9_concat_1 <- ConcatFree.Concat4Free("fire9_concat_1" ,axis = Some((1)),inputs = Seq(Some(nodefire9_expand1x1_2),Some(nodefire9_expand3x3_2)))
      nodefire9_concat_2 <- DropoutFree.Dropout7Free("fire9_concat_2" ,data = Some(nodefire9_concat_1))
      nodeconv10_1 <- ConvFree.Conv1Free("conv10_1" ,strides = Some((Array(1,1))),pads = Some((Array(0,0,0,0))),kernel_shape = Some((Array(1,1))),X = Some(nodefire9_concat_2._1),W = Some(nodeconv10_w_0),B = Some(nodeconv10_b_0))
      nodeconv10_2 <- ReluFree.Relu6Free("conv10_2" ,X = Some(nodeconv10_1))
      nodepool10_1 <- GlobalAveragePoolFree.GlobalAveragePool1Free("pool10_1" ,X = Some(nodeconv10_2))
      nodesoftmaxout_1 <- SoftmaxFree.Softmax1Free("softmaxout_1" ,input = Some(nodepool10_1))
    } yield (nodesoftmaxout_1)
}
