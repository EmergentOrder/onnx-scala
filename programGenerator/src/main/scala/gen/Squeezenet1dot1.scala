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
  def program(inputDatadata: Tensor[Float]): List[Tensor[Float]] =
    for {
      nodedata                     <- List(inputDatadata)
      nodesqueezenet0_conv0_weight <- List(dataSource.getParams[Float]("squeezenet0_conv0_weight"))
      nodesqueezenet0_conv0_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv0_bias"))
      nodesqueezenet0_conv1_weight <- List(dataSource.getParams[Float]("squeezenet0_conv1_weight"))
      nodesqueezenet0_conv1_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv1_bias"))
      nodesqueezenet0_conv2_weight <- List(dataSource.getParams[Float]("squeezenet0_conv2_weight"))
      nodesqueezenet0_conv2_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv2_bias"))
      nodesqueezenet0_conv3_weight <- List(dataSource.getParams[Float]("squeezenet0_conv3_weight"))
      nodesqueezenet0_conv3_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv3_bias"))
      nodesqueezenet0_conv4_weight <- List(dataSource.getParams[Float]("squeezenet0_conv4_weight"))
      nodesqueezenet0_conv4_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv4_bias"))
      nodesqueezenet0_conv5_weight <- List(dataSource.getParams[Float]("squeezenet0_conv5_weight"))
      nodesqueezenet0_conv5_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv5_bias"))
      nodesqueezenet0_conv6_weight <- List(dataSource.getParams[Float]("squeezenet0_conv6_weight"))
      nodesqueezenet0_conv6_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv6_bias"))
      nodesqueezenet0_conv7_weight <- List(dataSource.getParams[Float]("squeezenet0_conv7_weight"))
      nodesqueezenet0_conv7_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv7_bias"))
      nodesqueezenet0_conv8_weight <- List(dataSource.getParams[Float]("squeezenet0_conv8_weight"))
      nodesqueezenet0_conv8_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv8_bias"))
      nodesqueezenet0_conv9_weight <- List(dataSource.getParams[Float]("squeezenet0_conv9_weight"))
      nodesqueezenet0_conv9_bias   <- List(dataSource.getParams[Float]("squeezenet0_conv9_bias"))
      nodesqueezenet0_conv10_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv10_weight")
      )
      nodesqueezenet0_conv10_bias <- List(dataSource.getParams[Float]("squeezenet0_conv10_bias"))
      nodesqueezenet0_conv11_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv11_weight")
      )
      nodesqueezenet0_conv11_bias <- List(dataSource.getParams[Float]("squeezenet0_conv11_bias"))
      nodesqueezenet0_conv12_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv12_weight")
      )
      nodesqueezenet0_conv12_bias <- List(dataSource.getParams[Float]("squeezenet0_conv12_bias"))
      nodesqueezenet0_conv13_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv13_weight")
      )
      nodesqueezenet0_conv13_bias <- List(dataSource.getParams[Float]("squeezenet0_conv13_bias"))
      nodesqueezenet0_conv14_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv14_weight")
      )
      nodesqueezenet0_conv14_bias <- List(dataSource.getParams[Float]("squeezenet0_conv14_bias"))
      nodesqueezenet0_conv15_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv15_weight")
      )
      nodesqueezenet0_conv15_bias <- List(dataSource.getParams[Float]("squeezenet0_conv15_bias"))
      nodesqueezenet0_conv16_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv16_weight")
      )
      nodesqueezenet0_conv16_bias <- List(dataSource.getParams[Float]("squeezenet0_conv16_bias"))
      nodesqueezenet0_conv17_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv17_weight")
      )
      nodesqueezenet0_conv17_bias <- List(dataSource.getParams[Float]("squeezenet0_conv17_bias"))
      nodesqueezenet0_conv18_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv18_weight")
      )
      nodesqueezenet0_conv18_bias <- List(dataSource.getParams[Float]("squeezenet0_conv18_bias"))
      nodesqueezenet0_conv19_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv19_weight")
      )
      nodesqueezenet0_conv19_bias <- List(dataSource.getParams[Float]("squeezenet0_conv19_bias"))
      nodesqueezenet0_conv20_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv20_weight")
      )
      nodesqueezenet0_conv20_bias <- List(dataSource.getParams[Float]("squeezenet0_conv20_bias"))
      nodesqueezenet0_conv21_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv21_weight")
      )
      nodesqueezenet0_conv21_bias <- List(dataSource.getParams[Float]("squeezenet0_conv21_bias"))
      nodesqueezenet0_conv22_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv22_weight")
      )
      nodesqueezenet0_conv22_bias <- List(dataSource.getParams[Float]("squeezenet0_conv22_bias"))
      nodesqueezenet0_conv23_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv23_weight")
      )
      nodesqueezenet0_conv23_bias <- List(dataSource.getParams[Float]("squeezenet0_conv23_bias"))
      nodesqueezenet0_conv24_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv24_weight")
      )
      nodesqueezenet0_conv24_bias <- List(dataSource.getParams[Float]("squeezenet0_conv24_bias"))
      nodesqueezenet0_conv25_weight <- List(
        dataSource.getParams[Float]("squeezenet0_conv25_weight")
      )
      nodesqueezenet0_conv25_bias <- List(dataSource.getParams[Float]("squeezenet0_conv25_bias"))
      nodereshape_attr_tensor118  <- List(dataSource.getParams[Long]("reshape_attr_tensor118"))
      nodesqueezenet0_conv0_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu0_fwd <- List(
        Relu.ReluV6("squeezenet0_relu0_fwd", X = ((nodesqueezenet0_conv0_fwd))).apply(0)
      )
      nodesqueezenet0_pool0_fwd <- List(
        MaxPool
          .MaxPoolV1(
            "squeezenet0_pool0_fwd",
            kernel_shape = ((Array(3, 3))),
            pads = Some((Array(0, 0, 0, 0))),
            strides = Some((Array(2, 2))),
            X = ((nodesqueezenet0_relu0_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_conv1_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu1_fwd <- List(
        Relu.ReluV6("squeezenet0_relu1_fwd", X = ((nodesqueezenet0_conv1_fwd))).apply(0)
      )
      nodesqueezenet0_conv2_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu2_fwd <- List(
        Relu.ReluV6("squeezenet0_relu2_fwd", X = ((nodesqueezenet0_conv2_fwd))).apply(0)
      )
      nodesqueezenet0_conv3_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu3_fwd <- List(
        Relu.ReluV6("squeezenet0_relu3_fwd", X = ((nodesqueezenet0_conv3_fwd))).apply(0)
      )
      nodesqueezenet0_concat0 <- List(
        Concat
          .ConcatV4(
            "squeezenet0_concat0",
            axis = ((1)),
            inputs = (Seq(nodesqueezenet0_relu2_fwd, nodesqueezenet0_relu3_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_conv4_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu4_fwd <- List(
        Relu.ReluV6("squeezenet0_relu4_fwd", X = ((nodesqueezenet0_conv4_fwd))).apply(0)
      )
      nodesqueezenet0_conv5_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu5_fwd <- List(
        Relu.ReluV6("squeezenet0_relu5_fwd", X = ((nodesqueezenet0_conv5_fwd))).apply(0)
      )
      nodesqueezenet0_conv6_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu6_fwd <- List(
        Relu.ReluV6("squeezenet0_relu6_fwd", X = ((nodesqueezenet0_conv6_fwd))).apply(0)
      )
      nodesqueezenet0_concat1 <- List(
        Concat
          .ConcatV4(
            "squeezenet0_concat1",
            axis = ((1)),
            inputs = (Seq(nodesqueezenet0_relu5_fwd, nodesqueezenet0_relu6_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_pool1_fwd <- List(
        MaxPool
          .MaxPoolV1(
            "squeezenet0_pool1_fwd",
            kernel_shape = ((Array(3, 3))),
            pads = Some((Array(0, 0, 0, 0))),
            strides = Some((Array(2, 2))),
            X = ((nodesqueezenet0_concat1))
          )
          .apply(0)
      )
      nodesqueezenet0_conv7_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu7_fwd <- List(
        Relu.ReluV6("squeezenet0_relu7_fwd", X = ((nodesqueezenet0_conv7_fwd))).apply(0)
      )
      nodesqueezenet0_conv8_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu8_fwd <- List(
        Relu.ReluV6("squeezenet0_relu8_fwd", X = ((nodesqueezenet0_conv8_fwd))).apply(0)
      )
      nodesqueezenet0_conv9_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu9_fwd <- List(
        Relu.ReluV6("squeezenet0_relu9_fwd", X = ((nodesqueezenet0_conv9_fwd))).apply(0)
      )
      nodesqueezenet0_concat2 <- List(
        Concat
          .ConcatV4(
            "squeezenet0_concat2",
            axis = ((1)),
            inputs = (Seq(nodesqueezenet0_relu8_fwd, nodesqueezenet0_relu9_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_conv10_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu10_fwd <- List(
        Relu.ReluV6("squeezenet0_relu10_fwd", X = ((nodesqueezenet0_conv10_fwd))).apply(0)
      )
      nodesqueezenet0_conv11_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu11_fwd <- List(
        Relu.ReluV6("squeezenet0_relu11_fwd", X = ((nodesqueezenet0_conv11_fwd))).apply(0)
      )
      nodesqueezenet0_conv12_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu12_fwd <- List(
        Relu.ReluV6("squeezenet0_relu12_fwd", X = ((nodesqueezenet0_conv12_fwd))).apply(0)
      )
      nodesqueezenet0_concat3 <- List(
        Concat
          .ConcatV4(
            "squeezenet0_concat3",
            axis = ((1)),
            inputs = (Seq(nodesqueezenet0_relu11_fwd, nodesqueezenet0_relu12_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_pool2_fwd <- List(
        MaxPool
          .MaxPoolV1(
            "squeezenet0_pool2_fwd",
            kernel_shape = ((Array(3, 3))),
            pads = Some((Array(0, 0, 0, 0))),
            strides = Some((Array(2, 2))),
            X = ((nodesqueezenet0_concat3))
          )
          .apply(0)
      )
      nodesqueezenet0_conv13_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu13_fwd <- List(
        Relu.ReluV6("squeezenet0_relu13_fwd", X = ((nodesqueezenet0_conv13_fwd))).apply(0)
      )
      nodesqueezenet0_conv14_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu14_fwd <- List(
        Relu.ReluV6("squeezenet0_relu14_fwd", X = ((nodesqueezenet0_conv14_fwd))).apply(0)
      )
      nodesqueezenet0_conv15_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu15_fwd <- List(
        Relu.ReluV6("squeezenet0_relu15_fwd", X = ((nodesqueezenet0_conv15_fwd))).apply(0)
      )
      nodesqueezenet0_concat4 <- List(
        Concat
          .ConcatV4(
            "squeezenet0_concat4",
            axis = ((1)),
            inputs = (Seq(nodesqueezenet0_relu14_fwd, nodesqueezenet0_relu15_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_conv16_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu16_fwd <- List(
        Relu.ReluV6("squeezenet0_relu16_fwd", X = ((nodesqueezenet0_conv16_fwd))).apply(0)
      )
      nodesqueezenet0_conv17_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu17_fwd <- List(
        Relu.ReluV6("squeezenet0_relu17_fwd", X = ((nodesqueezenet0_conv17_fwd))).apply(0)
      )
      nodesqueezenet0_conv18_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu18_fwd <- List(
        Relu.ReluV6("squeezenet0_relu18_fwd", X = ((nodesqueezenet0_conv18_fwd))).apply(0)
      )
      nodesqueezenet0_concat5 <- List(
        Concat
          .ConcatV4(
            "squeezenet0_concat5",
            axis = ((1)),
            inputs = (Seq(nodesqueezenet0_relu17_fwd, nodesqueezenet0_relu18_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_conv19_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu19_fwd <- List(
        Relu.ReluV6("squeezenet0_relu19_fwd", X = ((nodesqueezenet0_conv19_fwd))).apply(0)
      )
      nodesqueezenet0_conv20_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu20_fwd <- List(
        Relu.ReluV6("squeezenet0_relu20_fwd", X = ((nodesqueezenet0_conv20_fwd))).apply(0)
      )
      nodesqueezenet0_conv21_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu21_fwd <- List(
        Relu.ReluV6("squeezenet0_relu21_fwd", X = ((nodesqueezenet0_conv21_fwd))).apply(0)
      )
      nodesqueezenet0_concat6 <- List(
        Concat
          .ConcatV4(
            "squeezenet0_concat6",
            axis = ((1)),
            inputs = (Seq(nodesqueezenet0_relu20_fwd, nodesqueezenet0_relu21_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_conv22_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu22_fwd <- List(
        Relu.ReluV6("squeezenet0_relu22_fwd", X = ((nodesqueezenet0_conv22_fwd))).apply(0)
      )
      nodesqueezenet0_conv23_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu23_fwd <- List(
        Relu.ReluV6("squeezenet0_relu23_fwd", X = ((nodesqueezenet0_conv23_fwd))).apply(0)
      )
      nodesqueezenet0_conv24_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu24_fwd <- List(
        Relu.ReluV6("squeezenet0_relu24_fwd", X = ((nodesqueezenet0_conv24_fwd))).apply(0)
      )
      nodesqueezenet0_concat7 <- List(
        Concat
          .ConcatV4(
            "squeezenet0_concat7",
            axis = ((1)),
            inputs = (Seq(nodesqueezenet0_relu23_fwd, nodesqueezenet0_relu24_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_dropout0_fwd <- List(
        Dropout.DropoutV7("squeezenet0_dropout0_fwd", data = ((nodesqueezenet0_concat7))).apply(0)
      )
      nodesqueezenet0_conv25_fwd <- List(
        Conv
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
      )
      nodesqueezenet0_relu25_fwd <- List(
        Relu.ReluV6("squeezenet0_relu25_fwd", X = ((nodesqueezenet0_conv25_fwd))).apply(0)
      )
      nodesqueezenet0_pool3_fwd <- List(
        AveragePool
          .AveragePoolV7(
            "squeezenet0_pool3_fwd",
            kernel_shape = ((Array(13, 13))),
            pads = Some((Array(0, 0, 0, 0))),
            strides = Some((Array(13, 13))),
            X = ((nodesqueezenet0_relu25_fwd))
          )
          .apply(0)
      )
      nodesqueezenet0_flatten0_reshape0 <- List(
        Reshape
          .ReshapeV5(
            "squeezenet0_flatten0_reshape0",
            data = ((nodesqueezenet0_pool3_fwd)),
            shapeInput = nodereshape_attr_tensor118
          )
          .apply(
            0
          ) //Some(nodereshape_attr_tensor118) One small patch here, due to a limitation in nGraph
        //shapeInput = ((nodereshape_attr_tensor118))).apply(0)
      )
    } yield (nodesqueezenet0_flatten0_reshape0)

  override def close(): Unit = {
    backend.close
    bytesDataSource.close
  }
}
