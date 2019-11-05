package org.emergentorder.onnx.backends
import org.emergentorder.onnx._

class NGraphBackendFull()
    extends NGraphBackendFullAtoL()
    with MatMul
    with MatMulInteger
    with Max
    with MaxPool
//    with MaxRoiPool
//    with MaxUnpool
    with Mean
    with MeanVarianceNormalization
    with Min
//    with Mod
    with Mul
    with Multinomial
    with Neg
//    with NonMaxSuppression
//    with NonZero
//    with Normalizer
    with Not
    with OneHot
//    with OneHotEncoder
    with Or
    with PRelu
    with Pad
    with Pow
    with QLinearConv
    with QLinearMatMul
    with QuantizeLinear
//    with RNN
//    with RandomNormal
//    with RandomNormalLike
//    with RandomUniform
//    with RandomUniformLike
//    with Range
    with Reciprocal
    with ReduceL1
    with ReduceL2
    with ReduceLogSum
    with ReduceLogSumExp
    with ReduceMax
    with ReduceMean
    with ReduceMin
    with ReduceProd
    with ReduceSum
    with ReduceSumSquare
    with Relu
    with Reshape
//    with Resize
    with ReverseSequence
//    with RoiAlign
//    with Round
//    with SVMClassifier
//    with SVMRegressor
//    with Scaler
//    with Scan
//    with Scatter
//    with ScatterElements
//    with ScatterND
    with Selu
//    with SequenceAt
//    with SequenceConstruct
//    with SequenceEmpty
//    with SequenceErase
//    with SequenceInsert
//    with SequenceLength
    with Shape
    with Shrink
    with Sigmoid
    with Sign
    with Sin
    with Sinh
    with Size
    with Slice
    with Softmax
    with Softplus
    with Softsign
    with SpaceToDepth
    with Split
//    with SplitToSequence
    with Sqrt
    with Squeeze
//    with StringNormalizer
    with Sub
    with Sum
    with Tan
    with Tanh
//    with TfIdfVectorizer
    with ThresholdedRelu
//    with Tile
    with TopK
    with Transpose
//    with TreeEnsembleClassifier
//    with TreeEnsembleRegressor
//    with Unique
    with Unsqueeze
//    with Upsample
    with Where
    with Xor
//    with ZipMap

