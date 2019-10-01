<p align="center"><img src="Logotype-500px.png" /></p>

--------------------------------------------------------------------------------

This project provides:

A) a complete, versioned, numerically generic, type-safe / typeful API to ONNX(Open Neural Network eXchange, an open format to represent deep learning and classical machine learning models), derived from the Protobuf definitions and the operator schemas (defined in C++) via the JavaCPP Preset for ONNX.

This API is expressed via traits, with version-named methods. For example, Abs, the absolute value operator (defined in operator sets 1 and 6):

```scala
  trait Abs extends Operator {

    def Abs1[@sp T: Numeric: ClassTag](
        name: String,
        consumed_inputs: Option[(Array[Int])] = None,
        X: Option[Tensor[T]]
    )(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T])

    def Abs6[@sp T: Numeric: ClassTag](name: String, X: Option[Tensor[T]])(
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): (Tensor[T])

  }
```

B) a program generator, capable of translating ONNX model Protobuf (.onnx) files into Scala programs written in terms of this API. For example, an "absolute value network":

```scala
trait AbsNet {
  val dataSource: DataSource
  val Abs: Abs
  def program[
      T: Numeric: ClassTag
  ]
  (inputData: Tensor[T])
  (
        implicit evT: Contains[
          T,
          Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
            Short
          ]#or[Int]#or[Long]#or[UNil]#create
        ]
    ): List[Tensor[T]] = {
    for {
      nodedata <- List(inputData)
      nodeabs <- List(
        Abs.Abs6[T](
          "abs",
          X = Some(nodedata)
        )
      )
    } yield (nodeabs)
  }
}
```

C) Currently a single partial backend implementation of ONNX, accessible from the JVM, is available. More backends may be added in due time.

This backend is based on [nGraph](https://github.com/NervanaSystems/ngraph), via nGraph JavaCPP Preset.

Supported ONNX ops (more coming):

* Abs
* Add
* ArgMax
* ArgMin
* AveragePool
* Concat
* Constant
* Conv
* Dropout
* Equal
* Gather
* Gemm
* GlobalAveragePool
* Log 
* Max
* MaxPool
* Min
* Mul
* Relu
* Reshape
* Sigmoid
* Softmax

All together, these should enable model inspection and modification, extra compile-time assurances, mixing/matching of backend operator implementations and integration into JVM-based production systems, for a start.

## Getting Started

You'll need sbt.

```
sbt publishLocal
```

or 

```
sbt +publishLocal
```

to build against all of Scala 2.11, 2.12, 2.13 and Dotty/3.0, where possible.

Then you can add this to your project's build.sbt 

```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala" % "1.6.0-0.1.0-SNAPSHOT"
```

or 

```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala-zio" % "1.6.0-0.1.0-SNAPSHOT"
``` 

and build away with the traits provided. Backend implementation (and other) PRs welcome!

## Program Generator

To generate an ONNX-Scala program from an ONNX Protobuf file (often `*.onnx`):

Depending on the size of the ONNX model, you may need to 

```
export SBT_OPTS="-XX:+CMSClassUnloadingEnabled -Xmx28G -Xss8M -XX:MaxMetaspaceSize=1024M"
```

either each time in the terminal, or in your `~/.bashrc` file or equivalent, or you will encounter errors.

Now, get `squeezenet.onnx` [here](https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz) (rename `model.onnx` from inside the tar.gz and move it to: `programGenerator/.jvm/src/main/resources/`). This model and more can be found in the [ONNX Model Zoo](https://github.com/onnx/models).

Then:

```
sbt "project programGeneratorJVM" "run squeezenet.onnx"
```

The resulting generated program appears as `programGenerator/src/gen/scala/Squeezenet.scala` and you can run `sbt compile` to confirm that the generated code compiles.

## Program Execution

There are 3 execution modes:

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

#### Tracing mode
In this mode, we construct the ONNX model to be executed on the fly, in memory and then execute it all at once on the backend.
We thus recover some of the performance we sacrificed in fine-grained mode, at the cost of losing support for dynamic graph structure (because we don't get outputs for individual ops).
This mode will allow for export of your from-scratch or generated-but-customized ONNX models.
This mode is a work-in-progress.

### Example execution
The most extensive working example at the moment is `zio/src/main/scala/NCFZIO.scala`, an implementation of Neural Collaborative Filtering, although you currently need to provide your own model file to load params from at `zio/.jvm/src/main/resources/NCF.onnx`. 

This example provides full model execution via the `fullNCF` method, while an example of fine-grained execution can be found at `zio/src/main/scala/NCFZIOFine.scala` in the `fineNCF` method.

To run it, use: 
```
sbt "project zioJVM" "run"`
```

## Project Details 

Automatic differentiation to enable training is under consideration (ONNX does not provide facilities for training).

Balancing the interests of minimal imposition of dependencies with purely functional programming, ONNX-Scala comes in two flavors: Vanilla and ZIO-infused.

ONNX-Scala is cross-built against Scala JVM (for Scala 2.11, 2.12 and 2.13) , Scala.js / JavaScript (for Scala 2.11, 2.12 and 2.13) and Scala Native (for Scala 2.11).

To take advantage of union types to express type constraints, a Dotty (Scala 3) build is available. The Dotty build does not support Scala.js or Scala Native.

The ZIO variant is not yet available for Dotty.

Currently at ONNX 1.6.0.


### Built With

#### Core

* [ONNX via JavaCPP Preset for ONNX 1.6.0](https://github.com/bytedeco/javacpp-presets/tree/master/onnx) - Open Neural Network Exchange / The missing bridge between Java and native C++ libraries (For access to Protobuf definitions and operator schemas)

* [Spire](https://github.com/non/spire) - Typelevel project enabling generic numeric programming (For support for unsigned ints, complex numbers, the Numeric type class and type specialization to avoid boxing overhead)

#### Optional - Dotty Variant

* [Dotty](https://github.com/lampepfl/dotty) - A next-generation compiler that will become Scala 3 (For native union types, used here to express ONNX type constraints)

#### Optional - ZIO Variant

* [ZIO](https://github.com/zio/zio) - A type-safe, composable library for asynchronous and concurrent programming in Scala 

#### Program Generator

* [Scalameta](https://github.com/scalameta/scalameta) - Library to read, analyze, transform and generate Scala programs (For a runtime parse pass of generated programs)

#### Backend

* [nGraph via JavaCPP Preset for nGraph 0.22.0](https://github.com/bytedeco/javacpp-presets/tree/master/ngraph) - nGraph is an open source C++ library, compiler and runtime for Deep Learning frameworks / The missing bridge between Java and native C++ libraries (For access to Protobuf definitions and operator schemas)


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
