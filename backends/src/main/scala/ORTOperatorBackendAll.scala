package org.emergentorder.onnx.backends
import org.emergentorder.onnx._


//Going forward we only support ops (and their versions) which are supported in both ONNX Runtime & ONNX.js (CPU), opset 6+
//Plus a handful of others we need
//Current ONNX opset 12, no new ops in opset 13, just support for training
class ORTOperatorBackendAll
    extends ORTOperatorBackend
    with AbsV6
    with AcosV7
    with AcoshV9
//    with AdagradV1 //New in 1.7.0, training
    with AddV7
    with AndV7
    with ArgMaxV11
//    with ArgMinV11 //Strangely missing from ONNX.js
//    with ArrayFeatureExtractorV1 //ONNX ML, not tested for in scoreboard
    with AsinV7
    with AsinhV9
    with AtanV7
    with AtanhV9
    with AveragePoolV7
    with AveragePoolV10
//    with AveragePoolV11 //Missing from ONNX.js
    with BatchNormalizationV7
    with BatchNormalizationV9
//    with BinarizerV1 //ONNX ML, not tested for in scoreboard
//    with BitShiftV11
    with CastV6
    with CastV9
//    with CastMapV1 //ONNX ML, not tested for in scoreboard
//    with CategoryMapperV1 //ONNX ML, not tested for in scoreboard
//    with CeilV1
    with CeilV6
//    with CeluV12 //new in 1.7.0
    with ClipV6
//    with ClipV11 //not supported in ONNX.js
//    with CompressV11 
    with ConcatV4 //Retained for Squeezenet v1.1
    with ConcatV11
//    with ConcatFromSequenceV11
//    with ConstantV9
//    with ConstantV11
//    with ConstantOfShapeV9 
    with ConvV1 //Retained for Squeezenet v1.1
    with ConvV11
//   with ConvIntegerV10
//    with ConvTransposeV11
    with CosV7
    with CoshV9
//    with CumSumV11
//    with DepthToSpaceV1
//    with DequantizeLinearV10
//    with DetV11
//    with DictVectorizerV1 //ONNX ML, not tested for in scoreboard
    with DivV7
    with DropoutV7
    with DropoutV10
    with DropoutV12
//    with DynamicQuantizeLinearV11
// with EinsumV12 //new in 1.7.0
    with EluV6
    with EqualV11 //Missing from ONNX.js, but we need it
//    with ErfV9
    with ExpV6
    with ExpandV8
//    with EyeLikeV9
//  with FeatureVectorizerV1 //ONNX ML, not tested for in scoreboard
    with FlattenV11
    with FloorV6
//    with GRUV7
    with GatherV11
//    with GatherElementsV11
//    with GatherNDV11
    with GemmV7
    with GemmV9
    with GemmV11
    with GlobalAveragePoolV1
//    with GlobalLpPoolV2 //fails in scoreboard
    with GlobalMaxPoolV1
//    with GradientV1 //Training, new in 1.7.0
//    with GraphCallV1 //Training, new in 1.7.0
    with GreaterV9 //Missing in ONNX.js, but we need it
    with GreaterOrEqualV12 //Missing in ONNX.js, but we need it
//    with HardSigmoidV6
//    with HardmaxV11
//    with IdentityV1
//    with IfV11 //fails in scoreboard
//    with ImputerV1 //ONNX ML, not tested for in scoreboard
    with InstanceNormalizationV6
//    with InverseV12 //New in 1.7.0
//    with IsInfV10
    with IsNaNV9
    with LRNV1
//    with LSTMV7
//    with LabelEncoderV2 //ONNX ML, not tested for in scoreboard
    with LeakyReluV6
    with LessV9 //Missing in ONNX.js, but we need it
    with LessOrEqualV12 //Missing in ONNX.js, but we need it
//    with LinearClassifierV1 //ONNX ML, not tested for in scoreboard
//    with LinearRegressorV1 //ONNX ML, not tested for in scoreboard
    with LogV6
//    with LogSoftmaxV11
//    with LoopV11
//    with LpNormalizationV1 //fails in scoreboard
//    with LpPoolV11 //fails in scoreboard
    with MatMulV9
//    with MatMulIntegerV10
    with MaxV8 //Fails in ONNX.js, but we need it
    with MaxPoolV1 //Retained for Squeezenet v1.1
    with MaxPoolV8
//    with MaxPoolV11 //Missing in ONNX.js
//    with MaxRoiPoolV1 //fails in scoreboard
//    with MaxUnpoolV11
//    with MeanV8 //Missing in ONNX.js, we can use ReduceMean instead
//    with MeanSquaredDistanceV12
//    with MeanVarianceNormalizationV9
    with MinV8 //Missing in ONNX.js, but we need it
    with ModV10 //Missing in ONNX.js, but we need it 
//    with MomentumV1 //Training, new in 1.7.0
    with MulV7
//    with MultinomialV7 //fails in scoreboard
    with NegV6
// with NegativeLogLikelihoodLossV12 //new in 1.7.0
//    with NonMaxSuppressionV11
//    with NonZeroV9
//    with NormalizerV1 //ONNX ML, not tested in scoreboard
    with NotV1
//    with OneHotV11
//    with OneHotEncoderV1 //ONNX ML, not tested in scoreboard
    with OrV7
    with PReluV7
    with PReluV9
    with PadV11
    with PowV7
    with PowV12
//    with QLinearConvV10
//    with QLinearMatMulV10
//    with QuantizeLinearV10
//    with RNNV7
//    with RandomNormalV1 //fails in scoreboard
//    with RandomNormalLikeV1 //fails in scoreboard
//    with RandomUniformV1 //fails in scoreboard
//    with RandomUniformLikeV1 //fails in scoreboard
    with RangeV11 //Missing in ONNX.js, but we need it
    with ReciprocalV6
//    with ReduceL1V11
//    with ReduceL2V11
    with ReduceLogSumV11
//    with ReduceLogSumExpV11
    with ReduceMaxV11
    with ReduceMaxV12
//    with ReduceMeanV1
    with ReduceMeanV11
    with ReduceMinV11
    with ReduceMinV12
    with ReduceProdV11
    with ReduceSumV11
    with ReduceSumSquareV11
    with ReluV6
    with ReshapeV5
//    with ResizeV11
//    with ReverseSequenceV10
//    with RoiAlignV10
    with RoundV11 //Missing in ONNX.js, but we need it
//    with SVMClassifierV1 //ONNXML, not tested for in scoreboard
//    with SVMRegressorV1 //ONNXML, not tested for in scoreboard
//    with ScalerV1 //ONNXML, not tested for in scoreboard
//    with ScanV11
//    with ScatterV11 //Deprecated in 1.8
//    with ScatterElementsV11
//    with ScatterNDV11
//    with SeluV6
//    with SequenceAtV11
//    with SequenceConstructV11
//    with SequenceEmptyV11
//    with SequenceEraseV11
//    with SequenceInsertV11
//    with SequenceLengthV11
    with ShapeV1
//    with ShrinkV9
//    with SigmoidV1
    with SigmoidV6
    with SignV9
    with SinV7
    with SinhV9
//    with SizeV1
    with SliceV10
    with SliceV11
//    with SoftmaxV1
    with SoftmaxV11
//    with SoftmaxCrossEntropyLossV12 //new in 1.7.0
//    with SoftplusV1
//    with SoftsignV1
//    with SpaceToDepthV1 //fails in scoreboard
//    with SplitV2
//    with SplitV11 //Nice to have
//    with SplitToSequenceV11
    with SqrtV6
    with SqueezeV11
//    with StringNormalizerV10
    with SubV7
    with SumV6
    with SumV8
    with TanV7
    with TanhV6
//    with TfIdfVectorizerV9
//    with ThresholdedReluV10
    with TileV6
//    with TopKV11 //Nice to have
    with TransposeV1
//    with TreeEnsembleClassifierV1 //ONNX ML, not tested for in scoreboard
//    with TreeEnsembleRegressorV1 //ONNX ML, not tested for in scoreboard
//    with UniqueV11
//    with UnsqueezeV1
    with UnsqueezeV11
//    with UpsampleV10 //Deprecated in 1.8
//    with WhereV9
    with XorV7
//    with ZipMapV1 //ONNX ML, not tested for in scoreboard
