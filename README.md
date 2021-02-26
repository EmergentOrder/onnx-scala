<p align="center"><img src="Logotype-500px.png" /></p>

--------------------------------------------------------------------------------

[![Build status](https://travis-ci.com/EmergentOrder/onnx-scala.svg?branch=master)](http://travis-ci.com/EmergentOrder/onnx-scala)
[![Latest version](https://index.scala-lang.org/emergentorder/onnx-scala/onnx-scala/latest.svg?color=orange)](https://index.scala-lang.org/emergentorder/onnx-scala/onnx-scala)
## Getting Started
Add this to the build.sbt in your project:

```scala
libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala-backends" % "0.8.0"
```

A short, recent talk I gave about the project: [ONNX-Scala: Typeful, Functional Deep Learning / Dotty Meets an Open AI Standard](https://youtu.be/8HuZTeHi7lg?t=1156)

### Full ONNX model inference - quick start
First, download the [model file](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx) for [SqueezeNet](https://en.wikipedia.org/wiki/SqueezeNet).
You can use `get_models.sh`

Note that all code snippets are written in Scala 3 (Dotty).

First we create an "image" tensor composed entirely of pixel value [42](https://upload.wikimedia.org/wikipedia/commons/0/0e/Answer_to_Life_42.svg):

```scala
import java.nio.file.{Files, Paths}
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

val squeezenetBytes = Files.readAllBytes(Paths.get("squeezenet1.1.onnx"))
val squeezenet = new ORTModelBackend(squeezenetBytes)

val data = Array.fill(1*3*224*224){42f}

//In NCHW tensor image format
val shape =                    1     #:     3      #:    224    #: 224     #: SNil
val tensorShapeDenotation = "Batch" ##: "Channel" ##: "Height" ##: "Width" ##: TSNil

val tensorDenotation: String & Singleton = "Image"

val imageTens = Tensor(data,tensorDenotation,tensorShapeDenotation,shape)

//or as a shorthand if you aren't concerned with enforcing denotations
val imageTensDefaultDenotations = Tensor(data,shape)
```

Note that ONNX tensor content is in row-major order.

Next we run SqueezeNet image classification inference on it:

```scala
val out = squeezenet.fullModel[Float, 
                               "ImageNetClassification",
                               "Batch" ##: "Class" ##: TSNil,
                               1 #: 1000 #: SNil](Tuple(imageTens))
// val out:
//  Tensor[Float,("ImageNetClassification", 
//                "Batch" ##: "Class" ##: TSNil,
//                1 #: 1000 #: SNil)] = (Array(0.8230729,
// ...

//The output shape
out.shape
// val res0: Array[Int] = Array(1, 1000)


//The highest probability (predicted) class
out.data.indices.maxBy(out.data)
// val res1: Int = 418
```

Referring to the [ImageNet 1000 class labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), we see that the predicted class is "ballpoint pen".

Based on a simple benchmark of 100000 iterations of SqueezeNet inference (run on my laptop), the run time is roughly on par with (within 10% of) ONNX Runtime (via Python).

The resulting output values also match ONNX Runtime/Python.

When using this API, we load the provided ONNX model file and pass it as-is to the underlying ONNX backend.
This is the most performant execution mode, and is recommended for off-the-shelf models / performance-critical scenarios.

This full-model API is untyped in the inputs, so it can fail at runtime. This inevitable because we load models from disk at runtime.
Feel free to wrap your calls into it in a facade with typed inputs.

### Operator-level (Fine-grained) API - quick start

You can call individual operators:

```scala
val onnxBackend = new ORTOperatorBackendAll()

val longTens = Tensor(Array.fill(1*3*224*224){-42l},tensorDenotation,tensorShapeDenotation,shape)
// longTens:
//  org.emergentorder.onnx.Tensors.Tensor[Float, 
//                                         ("Image", 
//                                          "Batch" ##: "Channel" ##: "Height" ##: "Width" ##:
//    org.emergentorder.compiletime.TSNil
//  , 1 #: 1000 #: io.kjaer.compiletime.SNil)] = (
//   Array(
//     -42L,
//     -42L,
// ...

onnxBackend.AbsV6("abs", longTens)
// res2:
//  org.emergentorder.onnx.Tensors.Tensor[Float, 
//                                          ("Image", 
//                                           "Batch" ##: "Channel" ##: "Height" ##: "Width" ##:
//    org.emergentorder.compiletime.TSNil
//  , 1 #: 1000 #: io.kjaer.compiletime.SNil)] = ( 
//   Array(
//     42L,
//     42L,
// ...
```

Sqrt will fail to compile because it's not defined for Long:
```scala
onnxBackend.SqrtV6("sqrt", longTens)
// ...
//Required: org.emergentorder.onnx.Tensors.Tensor[T, (
//...
//where:    T            is a type variable with constraint <: org.emergentorder.onnx.Float16 | Float | Double

```
Note that in real use backends should be closed to prevent native memory leaks.

## Project Details

Automatic differentiation to enable training is under consideration (ONNX currently provides facilities for training as a tech preview only).

The ONNX-Scala core (fine-grained) API is cross-built against Scala JVM (for Scala 2.13 and Dotty/3.0) , Scala.js / JavaScript (for Scala 2.13 and Dotty/3.0).

Currently at ONNX 1.8.0 (Backward compatible to at least 1.2.0), ONNX Runtime 1.6.0.
 
### A) Fine-grained API
A complete\*, versioned, numerically generic, type-safe / typeful API to ONNX(Open Neural Network eXchange, an open format to represent deep learning and classical machine learning models), derived from the Protobuf definitions and the operator schemas (defined in C++) via the JavaCPP Preset for ONNX. We also generate implementations for each operator in terms of core methods to be implemented by the backend.

This API is expressed via traits, with version-named methods. For example, Abs, the absolute value operator (defined here for operator set 6):

\* Up to roughly the intersection of supported ops in ONNX Runtime and ONNX.js

```scala
import scala.{specialized => sp}
import spire.math._
import spire.implicits._
import org.emergentorder.onnx._

  trait AbsV6 extends Operator {
    def AbsV6[
        @sp T <: UByte | UShort | UInt | 
                 ULong | Byte | Short | Int | 
                 Long | Float16 | Float | Double: Numeric,
      Tt <: TensorTypeDenotation, 
      Td <: TensorShapeDenotation, 
      S <: Shape]
      (name: String, X: Tensor[T, Tuple3[Tt, Td, S]])
      (using tt: ValueOf[Tt], 
             td: TensorShapeDenotationOf[Td], 
             s: ShapeOf[S]): Tensor[T, Tuple3[Tt, Td, S]] = {
      val map: Map[String, Any] = Map()
      val allInputs             = Tuple1(X)
      (callOp(name, "Abs", allInputs, map))
    }
  }
```

Using this API, each ONNX operation is executed on the underyling backend individually.
As a result, you can write your own models from scratch in Scala using ONNX-Scala operations, injecting parameters from outside sources as need be.
This allows for dynamic graph structure, in which the execution itself defines the graph, similar to PyTorch and Tensorflow Eager.
The trade-off made for this flexibility is that the underlying ONNX backend can no longer optimize the full graph, and the JNI boundary-crossing and ONNX graph structure at each operation results in additional overhead.

#### Type-safe Tensors
Featuring type-level tensor and axis labels/denotations, which along with literal types for dimension sizes allow for tensor/axes/shape/data-typed tensors.
Type constraints, as per the ONNX spec, are implemented at the operation level on inputs and outputs, using union types, match types and compiletime singleton ops (thanks to @MaximeKjaer for getting the latter into dotty).
Using ONNX docs for [dimension](https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md) and [type](https://github.com/onnx/onnx/blob/master/docs/TypeDenotation.md) denotation, as well as the [operators doc](https://github.com/onnx/onnx/blob/v1.7.0/docs/Operators.md) as a reference,
and inspired by [Nexus](https://github.com/ctongfei/nexus), [Neurocat](https://github.com/mandubian/neurocat) and [Named Tensors](https://pytorch.org/docs/stable/named_tensor.html).

### B) Backend
Currently there is one backend support, based on [ONNX Runtime](https://github.com/microsoft/onnxruntime), via their official Java API.
An alternate backend to enable Scala.js support, based on [ONNX.js](https://github.com/microsoft/onnxjs) is coming soon (blocked on new Scala.js bundler / ScalaPB releases for dotty support). 

Supported ONNX input and output tensor data types:
* Byte
* Short
* Int
* Long
* Float
* Double
* Boolean

Supported ONNX ops:

* ONNX Runtime: 145/154 total.
* ONNX JS: 72/154 total.
* ONNX-Scala: 82/154 total.

See the [ONNX backend scoreboard](http://onnx.ai/backend-scoreboard/index.html) 

#### Example execution

TODO: T5 example

## Build / Publish

You'll need sbt.

To build and publish locally:

```
sbt publishLocal
```

or

```
sbt +publishLocal
```

to build against Scala 2.13 and Dotty/3.0, where possible.

### Built With

#### Core

* [ONNX](https://github.com/onnx/onnx) via [ScalaPB](https://github.com/scalapb/ScalaPB) - Open Neural Network Exchange / The missing bridge between Java and native C++ libraries (For access to Protobuf definitions, used in the fine-grained API to create ONNX models in memory to send to the backend)

* [Spire](https://github.com/typelevel/spire) - Powerful new number types and numeric abstractions for Scala.  (For support for unsigned ints, complex numbers and the Numeric type class in the core API)

* [Dotty](https://github.com/lampepfl/dotty) - The Scala 3 compiler, also known as Dotty. (For union types (used here to express ONNX type constraints), match types, compiletime singleton ops, ...)

#### Backends

* [ONNX Runtime via ORT Java API](https://github.com/microsoft/onnxruntime/tree/master/java) - ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator

### Inspiration

#### Scala

* [Neurocat](https://github.com/mandubian/neurocat) -  From neural networks to the Category of composable supervised learning algorithms in Scala with compile-time matrix checking based on singleton-types

* [Nexus](https://github.com/ctongfei/nexus) - Experimental typesafe tensors & deep learning in Scala

* [Lantern](https://github.com/feiwang3311/Lantern) - Machine learning framework prototype in Scala. The design of Lantern is built on two important and well-studied programming language concepts, delimited continuations (for automatic differentiation) and multi-stage programming (staging for short).

* [DeepLearning.scala](https://github.com/ThoughtWorksInc/DeepLearning.scala) - A simple library for creating complex neural networks

* [tf-dotty](https://github.com/MaximeKjaer/tf-dotty) - Shape-safe TensorFlow in Dotty 
