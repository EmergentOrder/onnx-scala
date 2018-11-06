package org.emergentorder.onnxFree

import freestyle.free._
import freestyle.free.implicits._
import cats.free.{Free, FreeApplicative}
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
import singleton.ops._
import scala.language.higherKinds

@module trait SqueezenetFree {
  val ConvFree: ConvFree
  val ReluFree: ReluFree
  val MaxPoolFree: MaxPoolFree
  val ConcatFree: ConcatFree
  val DropoutFree: DropoutFree
  val AveragePoolFree: AveragePoolFree
  val ReshapeFree: ReshapeFree
  val dataSource: DataSourceFree
  def program[
      T: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check: Numeric: ClassTag]
    : FS.Seq[Tensor[T]] =
    for {
      nodedata <- dataSource.inputDataFree[T]
      nodesqueezenet0_conv0_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv0_weight")
      nodesqueezenet0_conv0_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv0_bias")
      nodesqueezenet0_conv1_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv1_weight")
      nodesqueezenet0_conv1_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv1_bias")
      nodesqueezenet0_conv2_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv2_weight")
      nodesqueezenet0_conv2_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv2_bias")
      nodesqueezenet0_conv3_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv3_weight")
      nodesqueezenet0_conv3_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv3_bias")
      nodesqueezenet0_conv4_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv4_weight")
      nodesqueezenet0_conv4_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv4_bias")
      nodesqueezenet0_conv5_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv5_weight")
      nodesqueezenet0_conv5_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv5_bias")
      nodesqueezenet0_conv6_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv6_weight")
      nodesqueezenet0_conv6_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv6_bias")
      nodesqueezenet0_conv7_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv7_weight")
      nodesqueezenet0_conv7_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv7_bias")
      nodesqueezenet0_conv8_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv8_weight")
      nodesqueezenet0_conv8_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv8_bias")
      nodesqueezenet0_conv9_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv9_weight")
      nodesqueezenet0_conv9_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv9_bias")
      nodesqueezenet0_conv10_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv10_weight")
      nodesqueezenet0_conv10_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv10_bias")
      nodesqueezenet0_conv11_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv11_weight")
      nodesqueezenet0_conv11_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv11_bias")
      nodesqueezenet0_conv12_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv12_weight")
      nodesqueezenet0_conv12_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv12_bias")
      nodesqueezenet0_conv13_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv13_weight")
      nodesqueezenet0_conv13_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv13_bias")
      nodesqueezenet0_conv14_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv14_weight")
      nodesqueezenet0_conv14_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv14_bias")
      nodesqueezenet0_conv15_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv15_weight")
      nodesqueezenet0_conv15_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv15_bias")
      nodesqueezenet0_conv16_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv16_weight")
      nodesqueezenet0_conv16_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv16_bias")
      nodesqueezenet0_conv17_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv17_weight")
      nodesqueezenet0_conv17_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv17_bias")
      nodesqueezenet0_conv18_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv18_weight")
      nodesqueezenet0_conv18_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv18_bias")
      nodesqueezenet0_conv19_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv19_weight")
      nodesqueezenet0_conv19_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv19_bias")
      nodesqueezenet0_conv20_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv20_weight")
      nodesqueezenet0_conv20_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv20_bias")
      nodesqueezenet0_conv21_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv21_weight")
      nodesqueezenet0_conv21_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv21_bias")
      nodesqueezenet0_conv22_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv22_weight")
      nodesqueezenet0_conv22_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv22_bias")
      nodesqueezenet0_conv23_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv23_weight")
      nodesqueezenet0_conv23_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv23_bias")
      nodesqueezenet0_conv24_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv24_weight")
      nodesqueezenet0_conv24_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv24_bias")
      nodesqueezenet0_conv25_weight <- dataSource.getParamsFree[T](
        "squeezenet0_conv25_weight")
      nodesqueezenet0_conv25_bias <- dataSource.getParamsFree[T](
        "squeezenet0_conv25_bias")
      nodereshape_attr_tensor118 <- dataSource.getParamsFree[Long](
        "reshape_attr_tensor118")
      nodesqueezenet0_conv0_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv0_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(2, 2))),
        group = Some((1)),
        X = Some(nodedata),
        W = Some(nodesqueezenet0_conv0_weight),
        B = Some(nodesqueezenet0_conv0_bias)
      )
      nodesqueezenet0_relu0_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu0_fwd",
        X = Some(nodesqueezenet0_conv0_fwd))
      nodesqueezenet0_pool0_fwd <- MaxPoolFree.MaxPool1Free[T](
        "squeezenet0_pool0_fwd",
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(2, 2))),
        X = Some(nodesqueezenet0_relu0_fwd))
      nodesqueezenet0_conv1_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv1_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_pool0_fwd),
        W = Some(nodesqueezenet0_conv1_weight),
        B = Some(nodesqueezenet0_conv1_bias)
      )
      nodesqueezenet0_relu1_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu1_fwd",
        X = Some(nodesqueezenet0_conv1_fwd))
      nodesqueezenet0_conv2_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv2_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu1_fwd),
        W = Some(nodesqueezenet0_conv2_weight),
        B = Some(nodesqueezenet0_conv2_bias)
      )
      nodesqueezenet0_relu2_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu2_fwd",
        X = Some(nodesqueezenet0_conv2_fwd))
      nodesqueezenet0_conv3_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv3_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu1_fwd),
        W = Some(nodesqueezenet0_conv3_weight),
        B = Some(nodesqueezenet0_conv3_bias)
      )
      nodesqueezenet0_relu3_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu3_fwd",
        X = Some(nodesqueezenet0_conv3_fwd))
      nodesqueezenet0_concat0 <- ConcatFree.Concat4Free[T](
        "squeezenet0_concat0",
        axis = Some((1)),
        inputs = Seq(Some(nodesqueezenet0_relu2_fwd)))
      nodesqueezenet0_conv4_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv4_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_concat0),
        W = Some(nodesqueezenet0_conv4_weight),
        B = Some(nodesqueezenet0_conv4_bias)
      )
      nodesqueezenet0_relu4_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu4_fwd",
        X = Some(nodesqueezenet0_conv4_fwd))
      nodesqueezenet0_conv5_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv5_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu4_fwd),
        W = Some(nodesqueezenet0_conv5_weight),
        B = Some(nodesqueezenet0_conv5_bias)
      )
      nodesqueezenet0_relu5_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu5_fwd",
        X = Some(nodesqueezenet0_conv5_fwd))
      nodesqueezenet0_conv6_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv6_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu4_fwd),
        W = Some(nodesqueezenet0_conv6_weight),
        B = Some(nodesqueezenet0_conv6_bias)
      )
      nodesqueezenet0_relu6_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu6_fwd",
        X = Some(nodesqueezenet0_conv6_fwd))
      nodesqueezenet0_concat1 <- ConcatFree.Concat4Free[T](
        "squeezenet0_concat1",
        axis = Some((1)),
        inputs = Seq(Some(nodesqueezenet0_relu5_fwd)))
      nodesqueezenet0_pool1_fwd <- MaxPoolFree.MaxPool1Free[T](
        "squeezenet0_pool1_fwd",
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(2, 2))),
        X = Some(nodesqueezenet0_concat1))
      nodesqueezenet0_conv7_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv7_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_pool1_fwd),
        W = Some(nodesqueezenet0_conv7_weight),
        B = Some(nodesqueezenet0_conv7_bias)
      )
      nodesqueezenet0_relu7_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu7_fwd",
        X = Some(nodesqueezenet0_conv7_fwd))
      nodesqueezenet0_conv8_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv8_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu7_fwd),
        W = Some(nodesqueezenet0_conv8_weight),
        B = Some(nodesqueezenet0_conv8_bias)
      )
      nodesqueezenet0_relu8_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu8_fwd",
        X = Some(nodesqueezenet0_conv8_fwd))
      nodesqueezenet0_conv9_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv9_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu7_fwd),
        W = Some(nodesqueezenet0_conv9_weight),
        B = Some(nodesqueezenet0_conv9_bias)
      )
      nodesqueezenet0_relu9_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu9_fwd",
        X = Some(nodesqueezenet0_conv9_fwd))
      nodesqueezenet0_concat2 <- ConcatFree.Concat4Free[T](
        "squeezenet0_concat2",
        axis = Some((1)),
        inputs = Seq(Some(nodesqueezenet0_relu8_fwd)))
      nodesqueezenet0_conv10_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv10_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_concat2),
        W = Some(nodesqueezenet0_conv10_weight),
        B = Some(nodesqueezenet0_conv10_bias)
      )
      nodesqueezenet0_relu10_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu10_fwd",
        X = Some(nodesqueezenet0_conv10_fwd))
      nodesqueezenet0_conv11_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv11_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu10_fwd),
        W = Some(nodesqueezenet0_conv11_weight),
        B = Some(nodesqueezenet0_conv11_bias)
      )
      nodesqueezenet0_relu11_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu11_fwd",
        X = Some(nodesqueezenet0_conv11_fwd))
      nodesqueezenet0_conv12_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv12_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu10_fwd),
        W = Some(nodesqueezenet0_conv12_weight),
        B = Some(nodesqueezenet0_conv12_bias)
      )
      nodesqueezenet0_relu12_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu12_fwd",
        X = Some(nodesqueezenet0_conv12_fwd))
      nodesqueezenet0_concat3 <- ConcatFree.Concat4Free[T](
        "squeezenet0_concat3",
        axis = Some((1)),
        inputs = Seq(Some(nodesqueezenet0_relu11_fwd)))
      nodesqueezenet0_pool2_fwd <- MaxPoolFree.MaxPool1Free[T](
        "squeezenet0_pool2_fwd",
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(2, 2))),
        X = Some(nodesqueezenet0_concat3))
      nodesqueezenet0_conv13_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv13_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_pool2_fwd),
        W = Some(nodesqueezenet0_conv13_weight),
        B = Some(nodesqueezenet0_conv13_bias)
      )
      nodesqueezenet0_relu13_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu13_fwd",
        X = Some(nodesqueezenet0_conv13_fwd))
      nodesqueezenet0_conv14_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv14_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu13_fwd),
        W = Some(nodesqueezenet0_conv14_weight),
        B = Some(nodesqueezenet0_conv14_bias)
      )
      nodesqueezenet0_relu14_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu14_fwd",
        X = Some(nodesqueezenet0_conv14_fwd))
      nodesqueezenet0_conv15_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv15_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu13_fwd),
        W = Some(nodesqueezenet0_conv15_weight),
        B = Some(nodesqueezenet0_conv15_bias)
      )
      nodesqueezenet0_relu15_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu15_fwd",
        X = Some(nodesqueezenet0_conv15_fwd))
      nodesqueezenet0_concat4 <- ConcatFree.Concat4Free[T](
        "squeezenet0_concat4",
        axis = Some((1)),
        inputs = Seq(Some(nodesqueezenet0_relu14_fwd)))
      nodesqueezenet0_conv16_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv16_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_concat4),
        W = Some(nodesqueezenet0_conv16_weight),
        B = Some(nodesqueezenet0_conv16_bias)
      )
      nodesqueezenet0_relu16_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu16_fwd",
        X = Some(nodesqueezenet0_conv16_fwd))
      nodesqueezenet0_conv17_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv17_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu16_fwd),
        W = Some(nodesqueezenet0_conv17_weight),
        B = Some(nodesqueezenet0_conv17_bias)
      )
      nodesqueezenet0_relu17_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu17_fwd",
        X = Some(nodesqueezenet0_conv17_fwd))
      nodesqueezenet0_conv18_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv18_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu16_fwd),
        W = Some(nodesqueezenet0_conv18_weight),
        B = Some(nodesqueezenet0_conv18_bias)
      )
      nodesqueezenet0_relu18_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu18_fwd",
        X = Some(nodesqueezenet0_conv18_fwd))
      nodesqueezenet0_concat5 <- ConcatFree.Concat4Free[T](
        "squeezenet0_concat5",
        axis = Some((1)),
        inputs = Seq(Some(nodesqueezenet0_relu17_fwd)))
      nodesqueezenet0_conv19_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv19_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_concat5),
        W = Some(nodesqueezenet0_conv19_weight),
        B = Some(nodesqueezenet0_conv19_bias)
      )
      nodesqueezenet0_relu19_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu19_fwd",
        X = Some(nodesqueezenet0_conv19_fwd))
      nodesqueezenet0_conv20_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv20_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu19_fwd),
        W = Some(nodesqueezenet0_conv20_weight),
        B = Some(nodesqueezenet0_conv20_bias)
      )
      nodesqueezenet0_relu20_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu20_fwd",
        X = Some(nodesqueezenet0_conv20_fwd))
      nodesqueezenet0_conv21_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv21_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu19_fwd),
        W = Some(nodesqueezenet0_conv21_weight),
        B = Some(nodesqueezenet0_conv21_bias)
      )
      nodesqueezenet0_relu21_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu21_fwd",
        X = Some(nodesqueezenet0_conv21_fwd))
      nodesqueezenet0_concat6 <- ConcatFree.Concat4Free[T](
        "squeezenet0_concat6",
        axis = Some((1)),
        inputs = Seq(Some(nodesqueezenet0_relu20_fwd)))
      nodesqueezenet0_conv22_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv22_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_concat6),
        W = Some(nodesqueezenet0_conv22_weight),
        B = Some(nodesqueezenet0_conv22_bias)
      )
      nodesqueezenet0_relu22_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu22_fwd",
        X = Some(nodesqueezenet0_conv22_fwd))
      nodesqueezenet0_conv23_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv23_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu22_fwd),
        W = Some(nodesqueezenet0_conv23_weight),
        B = Some(nodesqueezenet0_conv23_bias)
      )
      nodesqueezenet0_relu23_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu23_fwd",
        X = Some(nodesqueezenet0_conv23_fwd))
      nodesqueezenet0_conv24_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv24_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_relu22_fwd),
        W = Some(nodesqueezenet0_conv24_weight),
        B = Some(nodesqueezenet0_conv24_bias)
      )
      nodesqueezenet0_relu24_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu24_fwd",
        X = Some(nodesqueezenet0_conv24_fwd))
      nodesqueezenet0_concat7 <- ConcatFree.Concat4Free[T](
        "squeezenet0_concat7",
        axis = Some((1)),
        inputs = Seq(Some(nodesqueezenet0_relu23_fwd)))
      nodesqueezenet0_dropout0_fwd <- DropoutFree.Dropout7Free[T](
        "squeezenet0_dropout0_fwd",
        data = Some(nodesqueezenet0_concat7))
      nodesqueezenet0_conv25_fwd <- ConvFree.Conv1Free[T](
        "squeezenet0_conv25_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = Some(nodesqueezenet0_dropout0_fwd._1),
        W = Some(nodesqueezenet0_conv25_weight),
        B = Some(nodesqueezenet0_conv25_bias)
      )
      nodesqueezenet0_relu25_fwd <- ReluFree.Relu6Free[T](
        "squeezenet0_relu25_fwd",
        X = Some(nodesqueezenet0_conv25_fwd))
      nodesqueezenet0_pool3_fwd <- AveragePoolFree.AveragePool7Free[T](
        "squeezenet0_pool3_fwd",
        kernel_shape = Some((Array(13, 13))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(13, 13))),
        X = Some(nodesqueezenet0_relu25_fwd))
      nodesqueezenet0_flatten0_reshape0 <- ReshapeFree.Reshape5Free[T](
        "squeezenet0_flatten0_reshape0",
        data = Some(nodesqueezenet0_pool3_fwd),
        shape = Some(nodereshape_attr_tensor118))
    } yield (nodesqueezenet0_flatten0_reshape0)
}
