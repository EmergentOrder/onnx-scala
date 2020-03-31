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

//TODO: Squeezenet for ONNX-Scala dotty
class Squeezenet1dot1(byteArray: Array[Byte]) {
  val Conv: Conv               = new NGraphOperatorBackendFull()
  val Relu: Relu               = new NGraphOperatorBackendFull()
  val MaxPool: MaxPool         = new NGraphOperatorBackendFull()
  val Concat: Concat           = new NGraphOperatorBackendFull()
  val Dropout: Dropout         = new NGraphOperatorBackendFull()
  val AveragePool: AveragePool = new NGraphOperatorBackendFull()
  val Reshape: Reshape         = new NGraphOperatorBackendFull()
  val dataSource: DataSource   = new ONNXBytesDataSource(byteArray)
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
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu0_fwd <- List(
        Relu.Relu6("squeezenet0_relu0_fwd", X = Some(nodesqueezenet0_conv0_fwd))
      )
      nodesqueezenet0_pool0_fwd <- List(
        MaxPool.MaxPool1(
          "squeezenet0_pool0_fwd",
          kernel_shape = Some((Array(3, 3))),
          pads = Some((Array(0, 0, 0, 0))),
          strides = Some((Array(2, 2))),
          X = Some(nodesqueezenet0_relu0_fwd)
        )
      )
      nodesqueezenet0_conv1_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu1_fwd <- List(
        Relu.Relu6("squeezenet0_relu1_fwd", X = Some(nodesqueezenet0_conv1_fwd))
      )
      nodesqueezenet0_conv2_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu2_fwd <- List(
        Relu.Relu6("squeezenet0_relu2_fwd", X = Some(nodesqueezenet0_conv2_fwd))
      )
      nodesqueezenet0_conv3_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu3_fwd <- List(
        Relu.Relu6("squeezenet0_relu3_fwd", X = Some(nodesqueezenet0_conv3_fwd))
      )
      nodesqueezenet0_concat0 <- List(
        Concat.Concat4(
          "squeezenet0_concat0",
          axis = Some((1)),
          inputs = Seq(Some(nodesqueezenet0_relu2_fwd), Some(nodesqueezenet0_relu3_fwd))
        )
      )
      nodesqueezenet0_conv4_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu4_fwd <- List(
        Relu.Relu6("squeezenet0_relu4_fwd", X = Some(nodesqueezenet0_conv4_fwd))
      )
      nodesqueezenet0_conv5_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu5_fwd <- List(
        Relu.Relu6("squeezenet0_relu5_fwd", X = Some(nodesqueezenet0_conv5_fwd))
      )
      nodesqueezenet0_conv6_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu6_fwd <- List(
        Relu.Relu6("squeezenet0_relu6_fwd", X = Some(nodesqueezenet0_conv6_fwd))
      )
      nodesqueezenet0_concat1 <- List(
        Concat.Concat4(
          "squeezenet0_concat1",
          axis = Some((1)),
          inputs = Seq(Some(nodesqueezenet0_relu5_fwd), Some(nodesqueezenet0_relu6_fwd))
        )
      )
      nodesqueezenet0_pool1_fwd <- List(
        MaxPool.MaxPool1(
          "squeezenet0_pool1_fwd",
          kernel_shape = Some((Array(3, 3))),
          pads = Some((Array(0, 0, 0, 0))),
          strides = Some((Array(2, 2))),
          X = Some(nodesqueezenet0_concat1)
        )
      )
      nodesqueezenet0_conv7_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu7_fwd <- List(
        Relu.Relu6("squeezenet0_relu7_fwd", X = Some(nodesqueezenet0_conv7_fwd))
      )
      nodesqueezenet0_conv8_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu8_fwd <- List(
        Relu.Relu6("squeezenet0_relu8_fwd", X = Some(nodesqueezenet0_conv8_fwd))
      )
      nodesqueezenet0_conv9_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu9_fwd <- List(
        Relu.Relu6("squeezenet0_relu9_fwd", X = Some(nodesqueezenet0_conv9_fwd))
      )
      nodesqueezenet0_concat2 <- List(
        Concat.Concat4(
          "squeezenet0_concat2",
          axis = Some((1)),
          inputs = Seq(Some(nodesqueezenet0_relu8_fwd), Some(nodesqueezenet0_relu9_fwd))
        )
      )
      nodesqueezenet0_conv10_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu10_fwd <- List(
        Relu.Relu6("squeezenet0_relu10_fwd", X = Some(nodesqueezenet0_conv10_fwd))
      )
      nodesqueezenet0_conv11_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu11_fwd <- List(
        Relu.Relu6("squeezenet0_relu11_fwd", X = Some(nodesqueezenet0_conv11_fwd))
      )
      nodesqueezenet0_conv12_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu12_fwd <- List(
        Relu.Relu6("squeezenet0_relu12_fwd", X = Some(nodesqueezenet0_conv12_fwd))
      )
      nodesqueezenet0_concat3 <- List(
        Concat.Concat4(
          "squeezenet0_concat3",
          axis = Some((1)),
          inputs = Seq(Some(nodesqueezenet0_relu11_fwd), Some(nodesqueezenet0_relu12_fwd))
        )
      )
      nodesqueezenet0_pool2_fwd <- List(
        MaxPool.MaxPool1(
          "squeezenet0_pool2_fwd",
          kernel_shape = Some((Array(3, 3))),
          pads = Some((Array(0, 0, 0, 0))),
          strides = Some((Array(2, 2))),
          X = Some(nodesqueezenet0_concat3)
        )
      )
      nodesqueezenet0_conv13_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu13_fwd <- List(
        Relu.Relu6("squeezenet0_relu13_fwd", X = Some(nodesqueezenet0_conv13_fwd))
      )
      nodesqueezenet0_conv14_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu14_fwd <- List(
        Relu.Relu6("squeezenet0_relu14_fwd", X = Some(nodesqueezenet0_conv14_fwd))
      )
      nodesqueezenet0_conv15_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu15_fwd <- List(
        Relu.Relu6("squeezenet0_relu15_fwd", X = Some(nodesqueezenet0_conv15_fwd))
      )
      nodesqueezenet0_concat4 <- List(
        Concat.Concat4(
          "squeezenet0_concat4",
          axis = Some((1)),
          inputs = Seq(Some(nodesqueezenet0_relu14_fwd), Some(nodesqueezenet0_relu15_fwd))
        )
      )
      nodesqueezenet0_conv16_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu16_fwd <- List(
        Relu.Relu6("squeezenet0_relu16_fwd", X = Some(nodesqueezenet0_conv16_fwd))
      )
      nodesqueezenet0_conv17_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu17_fwd <- List(
        Relu.Relu6("squeezenet0_relu17_fwd", X = Some(nodesqueezenet0_conv17_fwd))
      )
      nodesqueezenet0_conv18_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu18_fwd <- List(
        Relu.Relu6("squeezenet0_relu18_fwd", X = Some(nodesqueezenet0_conv18_fwd))
      )
      nodesqueezenet0_concat5 <- List(
        Concat.Concat4(
          "squeezenet0_concat5",
          axis = Some((1)),
          inputs = Seq(Some(nodesqueezenet0_relu17_fwd), Some(nodesqueezenet0_relu18_fwd))
        )
      )
      nodesqueezenet0_conv19_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu19_fwd <- List(
        Relu.Relu6("squeezenet0_relu19_fwd", X = Some(nodesqueezenet0_conv19_fwd))
      )
      nodesqueezenet0_conv20_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu20_fwd <- List(
        Relu.Relu6("squeezenet0_relu20_fwd", X = Some(nodesqueezenet0_conv20_fwd))
      )
      nodesqueezenet0_conv21_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu21_fwd <- List(
        Relu.Relu6("squeezenet0_relu21_fwd", X = Some(nodesqueezenet0_conv21_fwd))
      )
      nodesqueezenet0_concat6 <- List(
        Concat.Concat4(
          "squeezenet0_concat6",
          axis = Some((1)),
          inputs = Seq(Some(nodesqueezenet0_relu20_fwd), Some(nodesqueezenet0_relu21_fwd))
        )
      )
      nodesqueezenet0_conv22_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu22_fwd <- List(
        Relu.Relu6("squeezenet0_relu22_fwd", X = Some(nodesqueezenet0_conv22_fwd))
      )
      nodesqueezenet0_conv23_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu23_fwd <- List(
        Relu.Relu6("squeezenet0_relu23_fwd", X = Some(nodesqueezenet0_conv23_fwd))
      )
      nodesqueezenet0_conv24_fwd <- List(
        Conv.Conv1(
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
      )
      nodesqueezenet0_relu24_fwd <- List(
        Relu.Relu6("squeezenet0_relu24_fwd", X = Some(nodesqueezenet0_conv24_fwd))
      )
      nodesqueezenet0_concat7 <- List(
        Concat.Concat4(
          "squeezenet0_concat7",
          axis = Some((1)),
          inputs = Seq(Some(nodesqueezenet0_relu23_fwd), Some(nodesqueezenet0_relu24_fwd))
        )
      )
      nodesqueezenet0_dropout0_fwd <- List(
        Dropout.Dropout7("squeezenet0_dropout0_fwd", data = Some(nodesqueezenet0_concat7))
      )
      nodesqueezenet0_conv25_fwd <- List(
        Conv.Conv1(
          "squeezenet0_conv25_fwd",
          dilations = Some((Array(1, 1))),
          kernel_shape = Some((Array(1, 1))),
          pads = Some((Array(0, 0, 0, 0))),
          strides = Some((Array(1, 1))),
          group = Some((1)),
          X = Some(nodesqueezenet0_dropout0_fwd),
          W = Some(nodesqueezenet0_conv25_weight),
          B = Some(nodesqueezenet0_conv25_bias)
        )
      )
      nodesqueezenet0_relu25_fwd <- List(
        Relu.Relu6("squeezenet0_relu25_fwd", X = Some(nodesqueezenet0_conv25_fwd))
      )
      nodesqueezenet0_pool3_fwd <- List(
        AveragePool.AveragePool7(
          "squeezenet0_pool3_fwd",
          kernel_shape = Some((Array(13, 13))),
          pads = Some((Array(0, 0, 0, 0))),
          strides = Some((Array(13, 13))),
          X = Some(nodesqueezenet0_relu25_fwd)
        )
      )
      nodesqueezenet0_flatten0_reshape0 <- List(
        Reshape.Reshape5(
          "squeezenet0_flatten0_reshape0",
          data = Some(nodesqueezenet0_pool3_fwd),
          shapeInput = None //Some(nodereshape_attr_tensor118) One small patch here, due to a limitation in nGraph
        )
      )
    } yield (nodesqueezenet0_flatten0_reshape0)
}
