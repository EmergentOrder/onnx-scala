package org.emergentorder.onnx

import org.emergentorder.onnx._
import org.emergentorder.onnx.backends._
import org.emergentorder.union._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.Complex
import spire.math.Numeric

class Squeezenet1dot1(byteArray: Array[Byte]) extends AutoCloseable {
  val backend                    = new ORTOperatorBackendAll()
  val bytesDataSource            = new ONNXBytesDataSource(byteArray)
  val Conv: ConvV1               = backend
  val Relu: ReluV6               = backend
  val MaxPool: MaxPoolV1         = backend
  val Concat: ConcatV4           = backend
  val Dropout: DropoutV7         = backend
  val AveragePool: AveragePoolV7 = backend
  val Reshape: ReshapeV5         = backend
  val dataSource: DataSource     = bytesDataSource
  def program(inputDatadata: Tensor[Float]): Tensor[Float] = {
    val nodedata                      = inputDatadata
    val nodesqueezenet0_conv0_weight  = dataSource.getParams[Float]("squeezenet0_conv0_weight")
    val nodesqueezenet0_conv0_bias    = dataSource.getParams[Float]("squeezenet0_conv0_bias")
    val nodesqueezenet0_conv1_weight  = dataSource.getParams[Float]("squeezenet0_conv1_weight")
    val nodesqueezenet0_conv1_bias    = dataSource.getParams[Float]("squeezenet0_conv1_bias")
    val nodesqueezenet0_conv2_weight  = dataSource.getParams[Float]("squeezenet0_conv2_weight")
    val nodesqueezenet0_conv2_bias    = dataSource.getParams[Float]("squeezenet0_conv2_bias")
    val nodesqueezenet0_conv3_weight  = dataSource.getParams[Float]("squeezenet0_conv3_weight")
    val nodesqueezenet0_conv3_bias    = dataSource.getParams[Float]("squeezenet0_conv3_bias")
    val nodesqueezenet0_conv4_weight  = dataSource.getParams[Float]("squeezenet0_conv4_weight")
    val nodesqueezenet0_conv4_bias    = dataSource.getParams[Float]("squeezenet0_conv4_bias")
    val nodesqueezenet0_conv5_weight  = dataSource.getParams[Float]("squeezenet0_conv5_weight")
    val nodesqueezenet0_conv5_bias    = dataSource.getParams[Float]("squeezenet0_conv5_bias")
    val nodesqueezenet0_conv6_weight  = dataSource.getParams[Float]("squeezenet0_conv6_weight")
    val nodesqueezenet0_conv6_bias    = dataSource.getParams[Float]("squeezenet0_conv6_bias")
    val nodesqueezenet0_conv7_weight  = dataSource.getParams[Float]("squeezenet0_conv7_weight")
    val nodesqueezenet0_conv7_bias    = dataSource.getParams[Float]("squeezenet0_conv7_bias")
    val nodesqueezenet0_conv8_weight  = dataSource.getParams[Float]("squeezenet0_conv8_weight")
    val nodesqueezenet0_conv8_bias    = dataSource.getParams[Float]("squeezenet0_conv8_bias")
    val nodesqueezenet0_conv9_weight  = dataSource.getParams[Float]("squeezenet0_conv9_weight")
    val nodesqueezenet0_conv9_bias    = dataSource.getParams[Float]("squeezenet0_conv9_bias")
    val nodesqueezenet0_conv10_weight = dataSource.getParams[Float]("squeezenet0_conv10_weight")
    val nodesqueezenet0_conv10_bias   = dataSource.getParams[Float]("squeezenet0_conv10_bias")
    val nodesqueezenet0_conv11_weight = dataSource.getParams[Float]("squeezenet0_conv11_weight")
    val nodesqueezenet0_conv11_bias   = dataSource.getParams[Float]("squeezenet0_conv11_bias")
    val nodesqueezenet0_conv12_weight = dataSource.getParams[Float]("squeezenet0_conv12_weight")
    val nodesqueezenet0_conv12_bias   = dataSource.getParams[Float]("squeezenet0_conv12_bias")
    val nodesqueezenet0_conv13_weight = dataSource.getParams[Float]("squeezenet0_conv13_weight")
    val nodesqueezenet0_conv13_bias   = dataSource.getParams[Float]("squeezenet0_conv13_bias")
    val nodesqueezenet0_conv14_weight = dataSource.getParams[Float]("squeezenet0_conv14_weight")
    val nodesqueezenet0_conv14_bias   = dataSource.getParams[Float]("squeezenet0_conv14_bias")
    val nodesqueezenet0_conv15_weight = dataSource.getParams[Float]("squeezenet0_conv15_weight")
    val nodesqueezenet0_conv15_bias   = dataSource.getParams[Float]("squeezenet0_conv15_bias")
    val nodesqueezenet0_conv16_weight = dataSource.getParams[Float]("squeezenet0_conv16_weight")
    val nodesqueezenet0_conv16_bias   = dataSource.getParams[Float]("squeezenet0_conv16_bias")
    val nodesqueezenet0_conv17_weight = dataSource.getParams[Float]("squeezenet0_conv17_weight")
    val nodesqueezenet0_conv17_bias   = dataSource.getParams[Float]("squeezenet0_conv17_bias")
    val nodesqueezenet0_conv18_weight = dataSource.getParams[Float]("squeezenet0_conv18_weight")
    val nodesqueezenet0_conv18_bias   = dataSource.getParams[Float]("squeezenet0_conv18_bias")
    val nodesqueezenet0_conv19_weight = dataSource.getParams[Float]("squeezenet0_conv19_weight")
    val nodesqueezenet0_conv19_bias   = dataSource.getParams[Float]("squeezenet0_conv19_bias")
    val nodesqueezenet0_conv20_weight = dataSource.getParams[Float]("squeezenet0_conv20_weight")
    val nodesqueezenet0_conv20_bias   = dataSource.getParams[Float]("squeezenet0_conv20_bias")
    val nodesqueezenet0_conv21_weight = dataSource.getParams[Float]("squeezenet0_conv21_weight")
    val nodesqueezenet0_conv21_bias   = dataSource.getParams[Float]("squeezenet0_conv21_bias")
    val nodesqueezenet0_conv22_weight = dataSource.getParams[Float]("squeezenet0_conv22_weight")
    val nodesqueezenet0_conv22_bias   = dataSource.getParams[Float]("squeezenet0_conv22_bias")
    val nodesqueezenet0_conv23_weight = dataSource.getParams[Float]("squeezenet0_conv23_weight")
    val nodesqueezenet0_conv23_bias   = dataSource.getParams[Float]("squeezenet0_conv23_bias")
    val nodesqueezenet0_conv24_weight = dataSource.getParams[Float]("squeezenet0_conv24_weight")
    val nodesqueezenet0_conv24_bias   = dataSource.getParams[Float]("squeezenet0_conv24_bias")
    val nodesqueezenet0_conv25_weight = dataSource.getParams[Float]("squeezenet0_conv25_weight")
    val nodesqueezenet0_conv25_bias   = dataSource.getParams[Float]("squeezenet0_conv25_bias")
    val nodereshape_attr_tensor118    = dataSource.getParams[Long]("reshape_attr_tensor118")
    val nodesqueezenet0_conv0_fwd = Conv
      .ConvV1(
        "squeezenet0_conv0_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(2, 2))),
        group = Some((1)),
        X = ((nodedata)),
        W = ((nodesqueezenet0_conv0_weight)),
        B = (Some((nodesqueezenet0_conv0_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu0_fwd =
      Relu.ReluV6("squeezenet0_relu0_fwd", X = ((nodesqueezenet0_conv0_fwd))).apply(0)
    val nodesqueezenet0_pool0_fwd = MaxPool
      .MaxPoolV1(
        "squeezenet0_pool0_fwd",
        kernel_shape = ((Array(3, 3))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(2, 2))),
        X = ((nodesqueezenet0_relu0_fwd))
      )
      .apply(0)
    val nodesqueezenet0_conv1_fwd = Conv
      .ConvV1(
        "squeezenet0_conv1_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_pool0_fwd)),
        W = ((nodesqueezenet0_conv1_weight)),
        B = (Some((nodesqueezenet0_conv1_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu1_fwd =
      Relu.ReluV6("squeezenet0_relu1_fwd", X = ((nodesqueezenet0_conv1_fwd))).apply(0)
    val nodesqueezenet0_conv2_fwd = Conv
      .ConvV1(
        "squeezenet0_conv2_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu1_fwd)),
        W = ((nodesqueezenet0_conv2_weight)),
        B = (Some((nodesqueezenet0_conv2_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu2_fwd =
      Relu.ReluV6("squeezenet0_relu2_fwd", X = ((nodesqueezenet0_conv2_fwd))).apply(0)
    val nodesqueezenet0_conv3_fwd = Conv
      .ConvV1(
        "squeezenet0_conv3_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu1_fwd)),
        W = ((nodesqueezenet0_conv3_weight)),
        B = (Some((nodesqueezenet0_conv3_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu3_fwd =
      Relu.ReluV6("squeezenet0_relu3_fwd", X = ((nodesqueezenet0_conv3_fwd))).apply(0)
    val nodesqueezenet0_concat0 = Concat
      .ConcatV4(
        "squeezenet0_concat0",
        axis = ((1)),
        inputs = (Seq(nodesqueezenet0_relu2_fwd, nodesqueezenet0_relu3_fwd))
      )
      .apply(0)
    val nodesqueezenet0_conv4_fwd = Conv
      .ConvV1(
        "squeezenet0_conv4_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_concat0)),
        W = ((nodesqueezenet0_conv4_weight)),
        B = (Some((nodesqueezenet0_conv4_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu4_fwd =
      Relu.ReluV6("squeezenet0_relu4_fwd", X = ((nodesqueezenet0_conv4_fwd))).apply(0)
    val nodesqueezenet0_conv5_fwd = Conv
      .ConvV1(
        "squeezenet0_conv5_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu4_fwd)),
        W = ((nodesqueezenet0_conv5_weight)),
        B = (Some((nodesqueezenet0_conv5_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu5_fwd =
      Relu.ReluV6("squeezenet0_relu5_fwd", X = ((nodesqueezenet0_conv5_fwd))).apply(0)
    val nodesqueezenet0_conv6_fwd = Conv
      .ConvV1(
        "squeezenet0_conv6_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu4_fwd)),
        W = ((nodesqueezenet0_conv6_weight)),
        B = (Some((nodesqueezenet0_conv6_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu6_fwd =
      Relu.ReluV6("squeezenet0_relu6_fwd", X = ((nodesqueezenet0_conv6_fwd))).apply(0)
    val nodesqueezenet0_concat1 = Concat
      .ConcatV4(
        "squeezenet0_concat1",
        axis = ((1)),
        inputs = (Seq(nodesqueezenet0_relu5_fwd, nodesqueezenet0_relu6_fwd))
      )
      .apply(0)
    val nodesqueezenet0_pool1_fwd = MaxPool
      .MaxPoolV1(
        "squeezenet0_pool1_fwd",
        kernel_shape = ((Array(3, 3))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(2, 2))),
        X = ((nodesqueezenet0_concat1))
      )
      .apply(0)
    val nodesqueezenet0_conv7_fwd = Conv
      .ConvV1(
        "squeezenet0_conv7_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_pool1_fwd)),
        W = ((nodesqueezenet0_conv7_weight)),
        B = (Some((nodesqueezenet0_conv7_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu7_fwd =
      Relu.ReluV6("squeezenet0_relu7_fwd", X = ((nodesqueezenet0_conv7_fwd))).apply(0)
    val nodesqueezenet0_conv8_fwd = Conv
      .ConvV1(
        "squeezenet0_conv8_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu7_fwd)),
        W = ((nodesqueezenet0_conv8_weight)),
        B = (Some((nodesqueezenet0_conv8_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu8_fwd =
      Relu.ReluV6("squeezenet0_relu8_fwd", X = ((nodesqueezenet0_conv8_fwd))).apply(0)
    val nodesqueezenet0_conv9_fwd = Conv
      .ConvV1(
        "squeezenet0_conv9_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu7_fwd)),
        W = ((nodesqueezenet0_conv9_weight)),
        B = (Some((nodesqueezenet0_conv9_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu9_fwd =
      Relu.ReluV6("squeezenet0_relu9_fwd", X = ((nodesqueezenet0_conv9_fwd))).apply(0)
    val nodesqueezenet0_concat2 = Concat
      .ConcatV4(
        "squeezenet0_concat2",
        axis = ((1)),
        inputs = (Seq(nodesqueezenet0_relu8_fwd, nodesqueezenet0_relu9_fwd))
      )
      .apply(0)
    val nodesqueezenet0_conv10_fwd = Conv
      .ConvV1(
        "squeezenet0_conv10_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_concat2)),
        W = ((nodesqueezenet0_conv10_weight)),
        B = (Some((nodesqueezenet0_conv10_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu10_fwd =
      Relu.ReluV6("squeezenet0_relu10_fwd", X = ((nodesqueezenet0_conv10_fwd))).apply(0)
    val nodesqueezenet0_conv11_fwd = Conv
      .ConvV1(
        "squeezenet0_conv11_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu10_fwd)),
        W = ((nodesqueezenet0_conv11_weight)),
        B = (Some((nodesqueezenet0_conv11_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu11_fwd =
      Relu.ReluV6("squeezenet0_relu11_fwd", X = ((nodesqueezenet0_conv11_fwd))).apply(0)
    val nodesqueezenet0_conv12_fwd = Conv
      .ConvV1(
        "squeezenet0_conv12_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu10_fwd)),
        W = ((nodesqueezenet0_conv12_weight)),
        B = (Some((nodesqueezenet0_conv12_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu12_fwd =
      Relu.ReluV6("squeezenet0_relu12_fwd", X = ((nodesqueezenet0_conv12_fwd))).apply(0)
    val nodesqueezenet0_concat3 = Concat
      .ConcatV4(
        "squeezenet0_concat3",
        axis = ((1)),
        inputs = (Seq(nodesqueezenet0_relu11_fwd, nodesqueezenet0_relu12_fwd))
      )
      .apply(0)
    val nodesqueezenet0_pool2_fwd = MaxPool
      .MaxPoolV1(
        "squeezenet0_pool2_fwd",
        kernel_shape = ((Array(3, 3))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(2, 2))),
        X = ((nodesqueezenet0_concat3))
      )
      .apply(0)
    val nodesqueezenet0_conv13_fwd = Conv
      .ConvV1(
        "squeezenet0_conv13_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_pool2_fwd)),
        W = ((nodesqueezenet0_conv13_weight)),
        B = (Some((nodesqueezenet0_conv13_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu13_fwd =
      Relu.ReluV6("squeezenet0_relu13_fwd", X = ((nodesqueezenet0_conv13_fwd))).apply(0)
    val nodesqueezenet0_conv14_fwd = Conv
      .ConvV1(
        "squeezenet0_conv14_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu13_fwd)),
        W = ((nodesqueezenet0_conv14_weight)),
        B = (Some((nodesqueezenet0_conv14_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu14_fwd =
      Relu.ReluV6("squeezenet0_relu14_fwd", X = ((nodesqueezenet0_conv14_fwd))).apply(0)
    val nodesqueezenet0_conv15_fwd = Conv
      .ConvV1(
        "squeezenet0_conv15_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu13_fwd)),
        W = ((nodesqueezenet0_conv15_weight)),
        B = (Some((nodesqueezenet0_conv15_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu15_fwd =
      Relu.ReluV6("squeezenet0_relu15_fwd", X = ((nodesqueezenet0_conv15_fwd))).apply(0)
    val nodesqueezenet0_concat4 = Concat
      .ConcatV4(
        "squeezenet0_concat4",
        axis = ((1)),
        inputs = (Seq(nodesqueezenet0_relu14_fwd, nodesqueezenet0_relu15_fwd))
      )
      .apply(0)
    val nodesqueezenet0_conv16_fwd = Conv
      .ConvV1(
        "squeezenet0_conv16_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_concat4)),
        W = ((nodesqueezenet0_conv16_weight)),
        B = (Some((nodesqueezenet0_conv16_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu16_fwd =
      Relu.ReluV6("squeezenet0_relu16_fwd", X = ((nodesqueezenet0_conv16_fwd))).apply(0)
    val nodesqueezenet0_conv17_fwd = Conv
      .ConvV1(
        "squeezenet0_conv17_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu16_fwd)),
        W = ((nodesqueezenet0_conv17_weight)),
        B = (Some((nodesqueezenet0_conv17_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu17_fwd =
      Relu.ReluV6("squeezenet0_relu17_fwd", X = ((nodesqueezenet0_conv17_fwd))).apply(0)
    val nodesqueezenet0_conv18_fwd = Conv
      .ConvV1(
        "squeezenet0_conv18_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu16_fwd)),
        W = ((nodesqueezenet0_conv18_weight)),
        B = (Some((nodesqueezenet0_conv18_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu18_fwd =
      Relu.ReluV6("squeezenet0_relu18_fwd", X = ((nodesqueezenet0_conv18_fwd))).apply(0)
    val nodesqueezenet0_concat5 = Concat
      .ConcatV4(
        "squeezenet0_concat5",
        axis = ((1)),
        inputs = (Seq(nodesqueezenet0_relu17_fwd, nodesqueezenet0_relu18_fwd))
      )
      .apply(0)
    val nodesqueezenet0_conv19_fwd = Conv
      .ConvV1(
        "squeezenet0_conv19_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_concat5)),
        W = ((nodesqueezenet0_conv19_weight)),
        B = (Some((nodesqueezenet0_conv19_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu19_fwd =
      Relu.ReluV6("squeezenet0_relu19_fwd", X = ((nodesqueezenet0_conv19_fwd))).apply(0)
    val nodesqueezenet0_conv20_fwd = Conv
      .ConvV1(
        "squeezenet0_conv20_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu19_fwd)),
        W = ((nodesqueezenet0_conv20_weight)),
        B = (Some((nodesqueezenet0_conv20_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu20_fwd =
      Relu.ReluV6("squeezenet0_relu20_fwd", X = ((nodesqueezenet0_conv20_fwd))).apply(0)
    val nodesqueezenet0_conv21_fwd = Conv
      .ConvV1(
        "squeezenet0_conv21_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu19_fwd)),
        W = ((nodesqueezenet0_conv21_weight)),
        B = (Some((nodesqueezenet0_conv21_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu21_fwd =
      Relu.ReluV6("squeezenet0_relu21_fwd", X = ((nodesqueezenet0_conv21_fwd))).apply(0)
    val nodesqueezenet0_concat6 = Concat
      .ConcatV4(
        "squeezenet0_concat6",
        axis = ((1)),
        inputs = (Seq(nodesqueezenet0_relu20_fwd, nodesqueezenet0_relu21_fwd))
      )
      .apply(0)
    val nodesqueezenet0_conv22_fwd = Conv
      .ConvV1(
        "squeezenet0_conv22_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_concat6)),
        W = ((nodesqueezenet0_conv22_weight)),
        B = (Some((nodesqueezenet0_conv22_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu22_fwd =
      Relu.ReluV6("squeezenet0_relu22_fwd", X = ((nodesqueezenet0_conv22_fwd))).apply(0)
    val nodesqueezenet0_conv23_fwd = Conv
      .ConvV1(
        "squeezenet0_conv23_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu22_fwd)),
        W = ((nodesqueezenet0_conv23_weight)),
        B = (Some((nodesqueezenet0_conv23_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu23_fwd =
      Relu.ReluV6("squeezenet0_relu23_fwd", X = ((nodesqueezenet0_conv23_fwd))).apply(0)
    val nodesqueezenet0_conv24_fwd = Conv
      .ConvV1(
        "squeezenet0_conv24_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(3, 3))),
        pads = Some((Array(1, 1, 1, 1))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_relu22_fwd)),
        W = ((nodesqueezenet0_conv24_weight)),
        B = (Some((nodesqueezenet0_conv24_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu24_fwd =
      Relu.ReluV6("squeezenet0_relu24_fwd", X = ((nodesqueezenet0_conv24_fwd))).apply(0)
    val nodesqueezenet0_concat7 = Concat
      .ConcatV4(
        "squeezenet0_concat7",
        axis = ((1)),
        inputs = (Seq(nodesqueezenet0_relu23_fwd, nodesqueezenet0_relu24_fwd))
      )
      .apply(0)
    val nodesqueezenet0_dropout0_fwd =
      Dropout.DropoutV7("squeezenet0_dropout0_fwd", data = ((nodesqueezenet0_concat7))).apply(0)
    val nodesqueezenet0_conv25_fwd = Conv
      .ConvV1(
        "squeezenet0_conv25_fwd",
        dilations = Some((Array(1, 1))),
        kernel_shape = Some((Array(1, 1))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(1, 1))),
        group = Some((1)),
        X = ((nodesqueezenet0_dropout0_fwd)),
        W = ((nodesqueezenet0_conv25_weight)),
        B = (Some((nodesqueezenet0_conv25_bias)))
      )
      .apply(0)
    val nodesqueezenet0_relu25_fwd =
      Relu.ReluV6("squeezenet0_relu25_fwd", X = ((nodesqueezenet0_conv25_fwd))).apply(0)
    val nodesqueezenet0_pool3_fwd = AveragePool
      .AveragePoolV7(
        "squeezenet0_pool3_fwd",
        kernel_shape = ((Array(13, 13))),
        pads = Some((Array(0, 0, 0, 0))),
        strides = Some((Array(13, 13))),
        X = ((nodesqueezenet0_relu25_fwd))
      )
      .apply(0)
    val nodesqueezenet0_flatten0_reshape0 = Reshape
      .ReshapeV5(
        "squeezenet0_flatten0_reshape0",
        data = ((nodesqueezenet0_pool3_fwd)),
        shapeInput = ((nodereshape_attr_tensor118))
      )
      .apply(0)
    return nodesqueezenet0_flatten0_reshape0
  }
  override def close(): Unit = {
    backend.close
    bytesDataSource.close
  }
}
