package org.emergentorder.onnx.backends
import org.emergentorder.onnx._

//Current ONNX opset 12 and ngraph test opset 11
//Commented out ops are not supported in NGraph currently
class ORTOperatorBackendAll
    extends ORTOperatorBackend
    with AbsV6
    with AcosV7
    with AcoshV9
    with AdagradV1 //New in 1.7.0, training
    with AddV7
    with AndV7
    with ArgMaxV11
    with ArgMinV11
//    with ArrayFeatureExtractorV1 //ONNX ML, not tested for in scoreboard
    with AsinV7
    with AsinhV9
    with AtanV7
    with AtanhV9
    with AveragePoolV7
    with AveragePoolV11
    with BatchNormalizationV9
//    with BinarizerV1 //ONNX ML, not tested for in scoreboard
    with BitShiftV11 //fails in scoreboard //passes in ORT
    with CastV9
//    with CastMapV1 //ONNX ML, not tested for in scoreboard
//    with CategoryMapperV1 //ONNX ML, not tested for in scoreboard
    with CeilV6
//    with CeluV12 //new in 1.7.0
    with ClipV11
    with CompressV11 //fails in scoreboard //passes in ORT
    with ConcatV4
    with ConcatV11
    with ConcatFromSequenceV11 //fails in scoreboard //passes in ORT
    with ConstantV9
    with ConstantV11
    with ConstantOfShapeV9 //fails in scoreboard //passes in ORT
    with ConvV1
    with ConvV11
    with ConvIntegerV10
    with ConvTransposeV11
    with CosV7
    with CoshV9
    with CumSumV11 //fails in scoreboard //passes in nGraph dev branch & ORT
    with DepthToSpaceV1
    with DequantizeLinearV10
    with DetV11 //fails in scoreboard //passes in ORT
//    with DictVectorizerV1 //ONNX ML, not tested for in scoreboard
    with DivV7
    with DropoutV7
    with DynamicQuantizeLinearV11 //fails in scoreboard //passes in ORT
// with EinsumV12 //new in 1.7.0
    with EluV6
    with EqualV11
    with ErfV9
    with ExpV6
    with ExpandV8 //fails in scoreboard //passes in ORT
    with EyeLikeV9
//  with FeatureVectorizerV1 //ONNX ML, not tested for in scoreboard
    with FlattenV11
    with FloorV6
    with GRUV7 //fails in scoreboard //passes in ORT
    with GatherV1
    with GatherV11
    with GatherElementsV11 //fails in scoreboard //passes in ORT
    with GatherNDV11       //fails in scoreboard //passes in nGraph dev branch & ORT
    with GemmV9
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
    with IsInfV10 //fails in scoreboard //passes in ORT
    with IsNaNV9  //fails in scoreboard //passes in ORT
    with LRNV1
    with LSTMV7
//    with LabelEncoderV2 //ONNX ML, not tested for in scoreboard
    with LeakyReluV6
    with LessV9
//    with LinearClassifierV1 //ONNX ML, not tested for in scoreboard
//    with LinearRegressorV1 //ONNX ML, not tested for in scoreboard
    with LogV6
    with LogSoftmaxV11
    with LoopV11 //fails in scoreboard //passes in ORT
//    with LpNormalizationV1 //fails in scoreboard
//    with LpPoolV11 //fails in scoreboard
    with MatMulV9
    with MatMulIntegerV10 //fails in scoreboard //passes in ORT
    with MaxV8
    with MaxPoolV1
    with MaxPoolV11
//    with MaxRoiPoolV1 //fails in scoreboard
    with MaxUnpoolV11 //fails in scoreboard //passes in ORT
    with MeanV8
//    with MeanSquaredDistanceV12 //new in 1.7.0
    with MeanVarianceNormalizationV9
    with MinV8
    with ModV10 //fails in scoreboard //passes in nGraph dev branch & ORT
//    with MomentumV1 //Training, new in 1.7.0
    with MulV7
//    with MultinomialV7 //fails in scoreboard
    with NegV6
// with NegativeLogLikelihoodLossV12 //new in 1.7.0
    with NonMaxSuppressionV11 //fails in scoreboard //passes in ORT
    with NonZeroV9            //fails in scoreboard //passes in ORT
//    with NormalizerV1 //ONNX ML, not tested in scoreboard
    with NotV1
    with OneHotV11 //fails in scoreboard //passes in ORT
//    with OneHotEncoderV1 //ONNX ML, not tested in scoreboard
    with OrV7
    with PReluV9
    with PadV11
    with PowV7
    with QLinearConvV10   //fails in scoreboard //passes in ORT
    with QLinearMatMulV10 //fails in scoreboard //passes in ORT
    with QuantizeLinearV10
    with RNNV7 //fails in scoreboard //passes in ORT
//    with RandomNormalV1 //fails in scoreboard
//    with RandomNormalLikeV1 //fails in scoreboard
//    with RandomUniformV1 //fails in scoreboard
//    with RandomUniformLikeV1 //fails in scoreboard
    with RangeV11 //fails in scoreboard //passes in ORT
    with ReciprocalV6
    with ReduceL1V11
    with ReduceL2V11
    with ReduceLogSumV11
    with ReduceLogSumExpV11
    with ReduceMaxV11
    with ReduceMeanV1
    with ReduceMeanV11
    with ReduceMinV11
    with ReduceProdV11
    with ReduceSumV11
    with ReduceSumSquareV11
    with ReluV6
    with ReshapeV5
    with ResizeV11 //fails in scoreboard //passes in ORT
    with ReverseSequenceV10
    with RoiAlignV10 //fails in scoreboard //passes in ORT
    with RoundV11    //fails in scoreboard //passes in nGraph dev branch & ORT
//    with SVMClassifierV1 //ONNXML, not tested for in scoreboard
//    with SVMRegressorV1 //ONNXML, not tested for in scoreboard
//    with ScalerV1 //ONNXML, not tested for in scoreboard
    with ScanV11            //fails in scoreboard //passes in ORT
    with ScatterV11         //fails in scoreboard //passes in ORT
    with ScatterElementsV11 //fails in scoreboard //passes in ORT
    with ScatterNDV11       //fails in scoreboard //passes in nGraph dev branch & ORT
    with SeluV6
    with SequenceAtV11        //fails in scoreboard //passes in ORT
    with SequenceConstructV11 //fails in scoreboard //passes in ORT
    with SequenceEmptyV11     //fails in scoreboard //passes in ORT
    with SequenceEraseV11     //fails in scoreboard //passes in ORT
    with SequenceInsertV11    //fails in scoreboard //passes in ORT
    with SequenceLengthV11    //fails in scoreboard //passes in ORT
    with ShapeV1
    with ShrinkV9
    with SigmoidV6
    with SignV9
    with SinV7
    with SinhV9
    with SizeV1
    with SliceV10
    with SliceV11
    with SoftmaxV1
    with SoftmaxV11
//    with SoftmaxCrossEntropyLossV12 //new in 1.7.0
    with SoftplusV1
    with SoftsignV1
//    with SpaceToDepthV1 //fails in scoreboard
    with SplitV2
    with SplitV11
    with SplitToSequenceV11 //fails in scoreboard //passes in ORT
    with SqrtV6
    with SqueezeV1
    with SqueezeV11
//    with StringNormalizerV10 //fails in scoreboard
    with SubV7
    with SumV8
    with TanV7
    with TanhV6
    with TfIdfVectorizerV9 //fails in scoreboard //passes in ORT
    with ThresholdedReluV10
    with TileV6  //fails in scoreboard //passes in ORT
    with TopKV11 //fails in scoreboard //passes in ORT
    with TransposeV1
//    with TreeEnsembleClassifierV1 //ONNX ML, not tested for in scoreboard
//    with TreeEnsembleRegressorV1 //ONNX ML, not tested for in scoreboard
    with UniqueV11 //fails in scoreboard //passes in ORT
    with UnsqueezeV1
    with UnsqueezeV11
    with UpsampleV10 //fails in scoreboard //passes in ORT
    with WhereV9
    with XorV7
//    with ZipMapV1 //ONNX ML, not tested for in scoreboard
