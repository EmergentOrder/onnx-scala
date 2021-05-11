package org.emergentorder.onnx.backends
import org.emergentorder.onnx._
import org.emergentorder.onnxruntime._
import org.emergentorder.onnxruntimecontrib._

//Going forward we only support ops (and their versions) which are supported in both ONNX Runtime & ONNX.js (CPU), opset 6+
//Plus a handful of others we need
//Current ONNX opset 12, no new ops in opset 13, just support for training
class ORTOperatorBackendAll
    extends ORTOperatorBackend
    with AbsV13
    with AcosV7
    with AcoshV9
//    with AdagradV1 //New in 1.7.0, training
    with AddV13
    with AndV7
    with ArgMaxV13
    with ArgMinV13 //Strangely missing from ONNX.js
//    with ArrayFeatureExtractorV1 //ONNX ML, not tested for in scoreboard
    with AsinV7
    with AsinhV9
    with AtanV7
    with AtanhV9
    with AveragePoolV11
//    with AveragePoolV11 //Missing from ONNX.js
    with BatchNormalizationV9
//    with BinarizerV1 //ONNX ML, not tested for in scoreboard
//    with BitShiftV11 //not supported in ONNX.js
//    with CastV13
//    with CastMapV1 //ONNX ML, not tested for in scoreboard
//    with CategoryMapperV1 //ONNX ML, not tested for in scoreboard
//    with CeilV1
    with CeilV13
    with CeluV12 //new in 1.7.0
    with ClipV13
//    with ClipV11 //not supported in ONNX.js
//    with CompressV11 
    with ConcatV13
//    with ConcatFromSequenceV11
    with ConstantV13 //Not supported in ONNX.js
//    with ConstantOfShapeV9 
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
    with DivV13
    with DropoutV13
//    with DynamicQuantizeLinearV11
// with EinsumV12 //new in 1.7.0
    with EluV6
    with EqualV13 //Missing from ONNX.js, but we need it
//    with ErfV9
    with ExpV13
    with ExpandV13
//    with EyeLikeV9
//  with FeatureVectorizerV1 //ONNX ML, not tested for in scoreboard
    with FlattenV13
    with FloorV13
//    with GRUV7 //fails in scoreboard
    with GatherV13
//    with GatherElementsV11
//    with GatherNDV11
    with GemmV13
    with GlobalAveragePoolV1
//    with GlobalLpPoolV2 //fails in scoreboard
    with GlobalMaxPoolV1
//    with GradientV1 //Training, new in 1.7.0
//    with GraphCallV1 //Training, new in 1.7.0
    with GreaterV13 //Missing in ONNX.js, but we need it
    with GreaterOrEqualV12 //Missing in ONNX.js, but we need it
//    with HardSigmoidV6
//    with HardmaxV11
//    with IdentityV1
//    with IfV11
//    with ImputerV1 //ONNX ML, not tested for in scoreboard
    with InstanceNormalizationV6
    with InverseV1 //New in 1.7.0
//    with IsInfV10
    with IsNaNV13
    with LRNV13
//    with LSTMV7 //fails in scoreboard
//    with LabelEncoderV2 //ONNX ML, not tested for in scoreboard
    with LeakyReluV6
    with LessV13 //Missing in ONNX.js, but we need it
    with LessOrEqualV12 //Missing in ONNX.js, but we need it
//    with LinearClassifierV1 //ONNX ML, not tested for in scoreboard
//    with LinearRegressorV1 //ONNX ML, not tested for in scoreboard
    with LogV13
//    with LogSoftmaxV11
//    with LoopV11
//    with LpNormalizationV1 //fails in scoreboard
//    with LpPoolV11 //fails in scoreboard
    with MatMulV13
//    with MatMulIntegerV10
    with MaxV13 //Fails in ONNX.js, but we need it
    with MaxPoolV12
//    with MaxPoolV11 //Missing in ONNX.js
//    with MaxRoiPoolV1 //fails in scoreboard
//    with MaxUnpoolV11
    with MeanV13 //Missing in ONNX.js, we can use ReduceMean instead
//    with MeanSquaredDistanceV12
//    with MeanVarianceNormalizationV9
    with MinV13 //Missing in ONNX.js, but we need it
    with ModV13 //Missing in ONNX.js, but we need it 
//    with MomentumV1 //Training, new in 1.7.0
    with MulV13
//    with MultinomialV7 //fails in scoreboard
    with NegV13
// with NegativeLogLikelihoodLossV12 //new in 1.7.0
//    with NonMaxSuppressionV11
//    with NonZeroV9
//    with NormalizerV1 //ONNX ML, not tested in scoreboard
    with NotV1
//    with OneHotV11
//    with OneHotEncoderV1 //ONNX ML, not tested in scoreboard
    with OrV7
    with PReluV9
    with PadV13
    with PowV13
//    with QLinearConvV10
//    with QLinearMatMulV10
//    with QuantizeLinearV10
//    with RNNV7 //fails in scoreboard
//    with RandomNormalV1 //fails in scoreboard
//    with RandomNormalLikeV1 //fails in scoreboard
//    with RandomUniformV1 //fails in scoreboard
//    with RandomUniformLikeV1 //fails in scoreboard
    with RangeV11 //Missing in ONNX.js, but we need it
    with ReciprocalV13
//    with ReduceL1V11
//    with ReduceL2V11
    with ReduceLogSumV13
//    with ReduceLogSumExpV11
    with ReduceMaxV13
//    with ReduceMeanV1
    with ReduceMeanV13
    with ReduceMinV13
    with ReduceProdV13 
    with ReduceSumV13
    with ReduceSumSquareV13
    with ReluV13
    with ReshapeV13
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
    with SeluV6
//    with SequenceAtV11
//    with SequenceConstructV11
//    with SequenceEmptyV11
//    with SequenceEraseV11
//    with SequenceInsertV11
//    with SequenceLengthV11
    with ShapeV13
//    with ShrinkV9
//    with SigmoidV1
    with SigmoidV13
    with SignV13
    with SinV7
    with SinhV9
//    with SizeV1
    with SliceV13
//    with SoftmaxV1
    with SoftmaxV13
//    with SoftmaxCrossEntropyLossV12 //new in 1.7.0
//    with SoftplusV1
//    with SoftsignV1
//    with SpaceToDepthV1 //fails in scoreboard
//    with SplitV2
//    with SplitV11 //Nice to have
//    with SplitToSequenceV11
    with SqrtV13
    with SqueezeV13
//    with StringNormalizerV10
    with SubV13
    with SumV13
    with TanV7
    with TanhV13
//    with TfIdfVectorizerV9
//    with ThresholdedReluV10
    with TileV13
//    with TopKV11 //Nice to have
    with TransposeV13
//    with TreeEnsembleClassifierV1 //ONNX ML, not tested for in scoreboard
//    with TreeEnsembleRegressorV1 //ONNX ML, not tested for in scoreboard
//    with UniqueV11
    with UnsqueezeV13
//    with UpsampleV10 //Deprecated in 1.8
//    with WhereV9
    with XorV7
//    with ZipMapV1 //ONNX ML, not tested for in scoreboard
