<p align="center"><img src="Logotype-500px.png" /></p>

--------------------------------------------------------------------------------

[![Build status](https://travis-ci.com/EmergentOrder/onnx-scala.svg?branch=master)](http://travis-ci.com/EmergentOrder/onnx-scala)
[![Latest version](https://index.scala-lang.org/emergentorder/onnx-scala/onnx-scala/latest.svg?color=orange)](https://index.scala-lang.org/emergentorder/onnx-scala/onnx-scala)
## Getting Started
Add this to the build.sbt in your project:

```scala
libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala-backends" % "0.2.0"
```

As of v0.1.0, artifacts are published to Sonatype OSS / Maven Central. For the latest, build and publish locally from master.


### Full ONNX model inference quick start
First, download the [model file](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx) for [SqueezeNet](https://en.wikipedia.org/wiki/SqueezeNet).

Using the console, from this project root:
```
sbt
project backendsJVM
console 
```

or from your project:
```
sbt console
```

Run SqueezeNet image classification inference on an "image" composed entirely of pixel value [42](https://upload.wikimedia.org/wikipedia/commons/0/0e/Answer_to_Life_42.svg):

```scala mdoc:silent
import java.nio.file.{Files, Paths}
import org.emergentorder.onnx.{Tensor, TensorFactory}
import org.emergentorder.onnx.backends.NGraphOperatorBackendFull
import org.emergentorder.onnx.backends.ORTOperatorBackendAll
import org.emergentorder.onnx.backends.ORTModelBackend

val squeezenetBytes = Files.readAllBytes(Paths.get("squeezenet1.1.onnx"))

val squeezenet = new ORTModelBackend(squeezenetBytes)

val imageTens = TensorFactory.getTensor(Array.fill(1*3*224*224){42f},Array(1,3,224,224))
```

Note that ONNX Tensor content is in row-major order.

```scala mdoc
val out: Tensor[Float] = squeezenet.fullModel(imageTens, None, None, None, None, None, None, None, None)

out._2

out._1.indices.maxBy(out._1)
```

Referring to the [ImageNet 1000 class labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), we see that the predicted class is "ballpoint pen".

Based on a simple benchmark of 10000 iterations of SqueezeNet inference (run on my laptop), it is on par with (within 3% of) ONNX Runtime (via Python).

The resulting output values also match ONNX Runtime.

### Operator-level (Fine-grained) API and generated programs

You can call individual operators:
```scala mdoc
val onnx = new ORTOperatorBackendAll()

//TODO: restore
//val longTens = TensorFactory.getTensor(Array.fill(1*3*224*224){-42l},Array(1,3,224,224))

//onnx.Abs6("abs", Some(longTens))
```

Sqrt will fail to compile because it's not defined for Long:
```scala mdoc:fail
onnx.Sqrt6("sqrt", Some(longTens))
```

And similarly you can call generated programs composed of these operators (details on how to generate from onnx file follow):
```scala mdoc
import org.emergentorder.onnx.Squeezenet1dot1
val generatedSqueezenet = new Squeezenet1dot1(squeezenetBytes)
val result = generatedSqueezenet.program(imageTens)

result(0)._2

result(0)._1.indices.maxBy(out._1)
```

And you can freely combine the two:
```scala mdoc
onnx.Softmax1("softmax", None, Some(result(0))) 
```

Note the type-safety (the full model version shown above fails at runtime):

```scala mdoc:fail
generatedSqueezenet.program(longTens)
```

Take note however, the generated version runs ~6x slower over 1000 iterations.
Also note that in real use backends should be closed to prevent native memory leaks.

## Project Overview
 
### A) API
A complete, versioned, numerically generic, type-safe / typeful API to ONNX(Open Neural Network eXchange, an open format to represent deep learning and classical machine learning models), derived from the Protobuf definitions and the operator schemas (defined in C++) via the JavaCPP Preset for ONNX. We also generate implementations for each operator in terms of core methods to be implemented by the backend.

This API is expressed via traits, with version-named methods. For example, Abs, the absolute value operator (defined here for operator set 6):

```scala mdoc
import scala.{specialized => sp}
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Numeric
import spire.implicits._
import scala.reflect.ClassTag
import org.emergentorder.onnx._

trait Abs extends Operator {
  def Abs6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T]) //Implementation omitted
}
```

A few more examples of the type constraints in action:
```scala mdoc:fail
val stringTens = TensorFactory.getTensor(Array.fill(3*224*224){"test"},Array(3,224,224))
onnx.Abs6("abs", Some(stringTens))
```

```scala mdoc:fail
val aBigInt = new BigInt(new java.math.BigInteger("5"))
val bigIntTens = TensorFactory.getTensor(Array.fill(3*224*224){aBigInt},Array(3,224,224))
onnx.Abs6("abs", Some(bigIntTens))
```

### B) Program Generator
Capable of translating ONNX model Protobuf (.onnx) files into Scala programs written in terms of this API.  
For example, an ["absolute value network"](https://raw.githubusercontent.com/onnx/onnx/master/onnx/backend/test/data/node/test_abs/model.onnx):

Depending on the size of the ONNX model, you may need to add 

```
export SBT_OPTS="-XX:+CMSClassUnloadingEnabled -Xmx16G -Xss8M -XX:MaxMetaspaceSize=1024M"
```

to your `~/.bashrc` file or equivalent, or you will encounter errors.

Move the model file to `programGenerator/.jvm/src/main/resources/absnet.onnx`). More models can be found in the [ONNX Model Zoo](https://github.com/onnx/models).

```
sbt "project programGeneratorJVM" "run absnet.onnx"
```

The resulting generated program appears as `programGenerator/src/gen/scala/Absnet.scala`:

```scala mdoc
import org.emergentorder.onnx.backends._

class Absnet(byteArray: Array[Byte]) {
  val Abs: org.emergentorder.onnx.Abs = new ORTOperatorBackendAll()
  def program(inputDatax: Tensor[Float]): List[Tensor[Float]]  = 
    for {
      nodex <- List(inputDatax)
      nodey <- List(Abs.Abs6("y" ,X = Some(nodex)))
    } yield (nodey)
}
```

and you can run `sbt compile` to confirm that the generated code compiles.

### C) Backend
Currently. at the operator level, a single partial backend implementation of ONNX, accessible from the JVM, is available.

This backend is based on [nGraph](https://github.com/NervanaSystems/ngraph), via nGraph JavaCPP Preset.

Supported ONNX input and output tensor data types:
* Byte
* Short
* Int
* Long
* Float
* Double

Supported ONNX ops:

* All those [supported](https://github.com/NervanaSystems/ngraph/tree/v0.26.0/src/ngraph/frontend/onnx_import/op) by nGraph, currently 100 of 153 total. The rest are in the API, but will error if called.

ONNX Runtime, which supports all ONNX ops, is the next targeted backend.

You can also pass entire models to nGraph (see Execution Modes below).

All together, these should enable model inspection and modification, extra compile-time assurances, mixing/matching of backend operator implementations and integration into JVM-based production systems, for a start.

#### Example execution

The most extensive working example at the moment is `zio/src/main/scala/NCFZIO.scala`, an implementation of Neural Collaborative Filtering, although you currently need to provide your own model file to load params from at `zio/.jvm/src/main/resources/NCF.onnx`, as well as item and user id maps at `zio/.jvm/src/main/resources/itemIds.csv` and `zio/.jvm/src/main/resources/userIds.csv`.

This example provides full model execution via the `fullNCF` method. 

To run it, use:

```
sbt "project zioJVM" "run"`
```

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

to build against all of Scala 2.11, 2.12, 2.13 and Dotty/3.0, where possible.

## Program Execution

There are 2 execution modes:

#### Full model / Black-box execution

In this mode we load the provided ONNX model file and pass it as-is to the underlying ONNX backend.
The model is represented as a Scala function with appropriately typed inputs and outputs.
Although we also generate the ONNX-Scala version of the program for type-checking and readability reasons, in the live code path the model internals are invisible to Scala.
This is the most performant execution mode, and is recommended in cases where static, pre-defined graphs are sufficient.

#### Fine-grained mode

In this mode, each ONNX operation is executed on the underyling backend individually.
As a result, you can write your own models from scratch in Scala using ONNX-Scala operations, injecting parameters from outside sources as need be.
You can also generate an ONNX-Scala program from an ONNX model file and then customize it as you see fit.

Type-checking is done at the operation level on inputs and outputs for data types, with support for type-checking over axis labels and tensor shapes coming.
This allows for dynamic graph structure, in which the execution itself defines the graph, similar to PyTorch and Tensorflow Eager.
The trade-off made for this flexibility is that the underlying ONNX backend can no longer optimize the full graph, and the boundary-crossing at each operation results in additional overhead.

## Project Details

Automatic differentiation to enable training is under consideration (ONNX does not provide facilities for training).

Balancing the interests of minimal imposition of dependencies with purely functional programming, ONNX-Scala comes in two flavors: Vanilla and ZIO-infused.

The ONNX-Scala core API is cross-built against Scala JVM (for Scala 2.11, 2.12, 2.13 and Dotty/3.0) , Scala.js / JavaScript (for Scala 2.11, 2.12 and 2.13) and Scala Native (for Scala 2.11).
The Scala Native build will fail unless you apply this [PR](https://github.com/scala-native/scala-native/pull/1641).

Currently at ONNX 1.6.0.

## Type-safe Tensors (Experimental, Scala 2.13 only)
Featuring type-checked axis labels (Dim), which along with literal types (new in Scala 2.13) for dimension sizes allow for axes-typed/shape-typed (Axes) tensors (TypesafeTensor).
Using ONNX docs for [dimension](https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md) and [type](https://github.com/onnx/onnx/blob/master/docs/TypeDenotation.md) denotation as a reference,
and inspired by [Nexus](https://github.com/ctongfei/nexus), [Neurocat](https://github.com/mandubian/neurocat) and [Named Tensors](https://pytorch.org/docs/stable/named_tensor.html).

```scala mdoc:silent
import org.emergentorder.onnx._

trait DataBatch extends Dim
trait DataChannel extends Dim
trait DataFeature extends Dim

val imageAxes = new Tuple3OfDim(3, new DataChannel{}, 224, new DataFeature{},224, new DataFeature{})
type ImageAxes = imageAxes.type
type ImageTensor = TypesafeTensor[Float, ImageAxes]

val typesafeTens: ImageTensor = TensorFactory.getTypesafeTensor(Array.fill(3*224*224){42f},imageAxes) 
onnx.Sqrt6[Float, ImageAxes]("sqrt", Some(typesafeTens))
onnx.Sqrt6("sqrt", Some(typesafeTens))

val textAxes = (new Vec(100, new DataFeature{}))
type TextAxes = textAxes.type
type TextTensor = TypesafeTensor[Float, TextAxes]

```
```scala mdoc:crash
//Fails at runtime, as designed
val wrongSizeDataTens: ImageTensor = TensorFactory.getTypesafeTensor(Array.fill(3*224*225){42f},imageAxes)
```
```scala mdoc:fail
//The rest fail to compile, as designed

val wordShouldBeImageTens: TextTensor = TensorFactory.getTypesafeTensor(Array.fill(3*224*224){42f},imageAxes)
```
```scala mdoc:fail
onnx.Sqrt6[Float, TextAxes]("sqrt", Some(typesafeTens))
```
```scala mdoc:fail
val wrongSizedImageAxes = (new Tuple3OfDim(15, new DataChannel{}, 224, new DataFeature{}, 224, new DataFeature{}))
type WrongSizedImageAxes = wrongSizedImageAxes.type
onnx.Sqrt6[Float, WrongSizedImageAxes]("sqrt", Some(typesafeTens))
```
```scala mdoc:fail
val wrongDimTypeAxes = (new Tuple3OfDim(3, new DataBatch{}, 224, new DataFeature{}, 224, new DataFeature{}))
type WrongDimTypeAxes = wrongDimTypeAxes.type
onnx.Sqrt6[Float, WrongDimTypeAxes]("sqrt", Some(typesafeTens))
```

### Built With

#### Core

* [ONNX via JavaCPP Preset for ONNX 1.6.0](https://github.com/bytedeco/javacpp-presets/tree/master/onnx) - Open Neural Network Exchange / The missing bridge between Java and native C++ libraries (For access to Protobuf definitions and operator schemas)

* [Spire](https://github.com/non/spire) - Typelevel project enabling generic numeric programming (For support for unsigned ints, complex numbers, the Numeric type class and type specialization to avoid boxing overhead)

#### Optional - Dotty Variant

* [Dotty](https://github.com/lampepfl/dotty) - A next-generation compiler that will become Scala 3 (For native union types, formerly used here to express ONNX type constraints, but currently using cross-version source compatibile union types instead)

#### Optional - ZIO Variant

* [ZIO](https://github.com/zio/zio) - A type-safe, composable library for asynchronous and concurrent programming in Scala

#### Program Generator

* [Scalameta](https://github.com/scalameta/scalameta) - Library to read, analyze, transform and generate Scala programs (For a runtime parse pass of generated programs)

#### Backend

* [nGraph via JavaCPP Preset for nGraph 0.26.0](https://github.com/bytedeco/javacpp-presets/tree/master/ngraph) - nGraph is an open source C++ library, compiler and runtime for Deep Learning frameworks / The missing bridge between Java and native C++ libraries


### Inspiration

#### Scala

* [Neurocat](https://github.com/mandubian/neurocat) -  From neural networks to the Category of composable supervised learning algorithms in Scala with compile-time matrix checking based on singleton-types

* [Nexus](https://github.com/ctongfei/nexus) - Experimental typesafe tensors & deep learning in Scala

* [Lantern](https://github.com/feiwang3311/Lantern) - Machine learning framework prototype in Scala. The design of Lantern is built on two important and well-studied programming language concepts, delimited continuations (for automatic differentiation) and multi-stage programming (staging for short).

* [DeepLearning.scala](https://github.com/ThoughtWorksInc/DeepLearning.scala) - A simple library for creating complex neural networks

* [Deeplearning4j / Scalnet / ND4S ](https://github.com/deeplearning4j/deeplearning4j/tree/master/scalnet) - ScalNet is a wrapper around Deeplearning4J emulating a Keras like API for deep learning.

#### Haskell

* [Backprop](https://github.com/mstksg/backprop) - Heterogeneous automatic differentiation ("backpropagation") in Haskell

* [Grenade](https://github.com/HuwCampbell/grenade) - Grenade is a composable, dependently typed, practical, and fast recurrent neural network library for concise and precise specifications of complex networks in Haskell.

