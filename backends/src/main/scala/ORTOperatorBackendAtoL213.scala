package org.emergentorder.onnx.backends
import org.emergentorder.onnx._

//TODO: add extra ORT ops
//Commented out ops are not supported in NGraph currently
class ORTOperatorBackendAtoL
    extends ORTOperatorBackend
    with Abs
    with Acos
    with Acosh
    with Add
    with And
    with ArgMax
    with ArgMin
//    with ArrayFeatureExtractor
    with Asin
    with Asinh
    with Atan
// Missing op: Atanh
    with AveragePool
    with BatchNormalization
//    with Binarizer
    with BitShift
    with Cast
//    with CastMap
//    with CategoryMapper
    with Ceil
    with Clip
    with Compress
    with Concat
    with ConcatFromSequence
    with Constant
    with ConstantOfShape
    with Conv
    with ConvInteger
    with ConvTranspose
    with Cos
    with Cosh
    with CumSum
    with DepthToSpace
    with DequantizeLinear
    with Det
//    with DictVectorizer
    with Div
    with Dropout
    with DynamicQuantizeLinear
    with Elu
    with Equal
    with Erf
    with Exp
    with Expand
    with EyeLike
    with Flatten
    with Floor
    with GRU
    with Gather
    with GatherElements
    with GatherND
    with Gemm
    with GlobalAveragePool
//    with GlobalLpPool
    with GlobalMaxPool
    with Greater
    with GreaterOrEqual
    with HardSigmoid
    with Hardmax
    with Identity
//    with If
//    with Imputer
    with InstanceNormalization
    with IsInf
    with IsNaN
    with LRN
    with LSTM
//    with LabelEncoder
    with LeakyRelu
    with Less
    with LessOrEqual
//    with LinearClassifier
//    with LinearRegressor
    with Log
    with LogSoftmax
    with Loop
    with LpNormalization
    with LpPool
