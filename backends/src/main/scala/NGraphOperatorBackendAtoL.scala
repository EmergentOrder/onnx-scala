package org.emergentorder.onnx.backends
import org.emergentorder.onnx._

//Current ONNX opset 12 and ngraph test opset 11
//Commented out ops are not supported in NGraph currently
class NGraphOperatorBackendAtoL
    extends NGraphOperatorBackend
    with AbsV6
    with AcosV7
    with AcoshV9
//  with AdagradV1 //New in 1.7.0, training
    with AddV7
    with AndV7
    with ArgMaxV11
    with ArgMinV11
//    with ArrayFeatureExtractorV1 //ONNX ML, not tested for in scoreboard
    with AsinV7
    with AsinhV9
    with AtanV7
    with AtanhV9
    with AveragePoolV11
    with BatchNormalizationV9
//    with BinarizerV1 //ONNX ML, not tested for in scoreboard
//    with BitShiftV11 //fails in scoreboard //passes in ORT
    with CastV9
//    with CastMapV1 //ONNX ML, not tested for in scoreboard
//    with CategoryMapperV1 //ONNX ML, not tested for in scoreboard
    with CeilV6
//    with CeluV12 //new in 1.7.0
    with ClipV11
//    with CompressV11 //fails in scoreboard //passes in ORT
    with ConcatV11
//    with ConcatFromSequenceV11 //fails in scoreboard //passes in ORT
    with ConstantV11
//    with ConstantOfShapeV9 //fails in scoreboard //passes in ORT
    with ConvV11
    with ConvIntegerV10
    with ConvTransposeV11
    with CosV7
    with CoshV9
//    with CumSumV11 //fails in scoreboard //passes in nGraph dev branch & ORT
    with DepthToSpaceV1
    with DequantizeLinearV10
//    with DetV11 //fails in scoreboard //passes in ORT
//    with DictVectorizerV1 //ONNX ML, not tested for in scoreboard
    with DivV7
    with DropoutV7
//    with DynamicQuantizeLinearV11 //fails in scoreboard //passes in ORT
// with EinsumV12 //new in 1.7.0
    with EluV6
    with EqualV11
    with ErfV9
    with ExpV6
//    with ExpandV8 //fails in scoreboard //passes in ORT
    with EyeLikeV9
//  with FeatureVectorizerV1 //ONNX ML, not tested for in scoreboard
    with FlattenV11
    with FloorV6
//    with GRUV7 //fails in scoreboard //passes in ORT
    with GatherV11
//    with GatherElementsV11 //fails in scoreboard //passes in ORT
//    with GatherNDV11 //fails in scoreboard //passes in nGraph dev branch & ORT
    with GemmV11
    with GlobalAveragePoolV1
//    with GlobalLpPoolV2 //fails in scoreboard
    with GlobalMaxPoolV1
//    with GradientV1 //Training, new in 1.7.0
//    with GraphCallV1 //Training, new in 1.7.0
    with GreaterV9
//    with GreaterOrEqualV12 //new in 1.7.0
    with HardSigmoidV6
    with HardmaxV11
    with IdentityV1
//    with IfV11 //fails in scoreboard
//    with ImputerV1 //ONNX ML, not tested for in scoreboard
    with InstanceNormalizationV6
//    with InverseV12 //New in 1.7.0
//    with IsInfV10 //fails in scoreboard //passes in ORT
//    with IsNaNV9 //fails in scoreboard //passes in ORT
    with LRNV1
    with LSTMV7
//    with LabelEncoderV2 //ONNX ML, not tested for in scoreboard
    with LeakyReluV6
    with LessV9
//    with LinearClassifierV1 //ONNX ML, not tested for in scoreboard
//    with LinearRegressorV1 //ONNX ML, not tested for in scoreboard
    with LogV6
    with LogSoftmaxV11
//    with LoopV11 //fails in scoreboard //passes in ORT
//    with LpNormalizationV1 //fails in scoreboard
//    with LpPoolV11 //fails in scoreboard
