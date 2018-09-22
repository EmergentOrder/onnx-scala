# ONNX-Scala

This project currently provides a complete, versioned, numerically generic, type-safe / typeful API to ONNX(Open Neural Network eXchange, an open format to represent deep learning and classical machine learning models), derived from the Protobuf definitions and the operator schemas (defined in C++) via the JavaCPP Preset for ONNX.

ONNX-Scala will also be home to A) a program generator, capable of translating .onnx files into Scala programs written in terms of this API and B) at least one, eventually many backend implementations of ONNX accessible from the JVM.

All together, these should enable model inspection and modification, extra compile-time assurances, mixing/matching of backend operator implementations and integration into JVM-based production systems, for a start.

Balancing the interests of minimal imposition of dependencies with stack-safe, purely functional programming, ONNX-Scala comes in two flavors: Vanilla and Freestyle-infused.

ONNX-Scala is cross-built against Scala JVM (for Scala 2.11, 2.12 and 2.13.0-M5) , Scala.js / JavaScript (for Scala 2.11 and 2.12) and Scala Native (for Scala 2.11).

To take advantage of union types to express type constraints, a Dotty (Scala 3) build is available. The Dotty build does not support Scala.js or Scala Native.

Due to Freestyle's dependency on the EOLed scalameta paradise compiler plugin, the free variant is not available for Scala 2.13 or Dotty.

Currently at ONNX 1.3.0.


## Getting Started

You'll need sbt.

```
sbt +publishLocal
```

Then you can add this to your project's build.sbt 

```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala" % "1.3.0-0.1.0-SNAPSHOT"
```

or 

```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala-free" % "1.3.0-0.1.0-SNAPSHOT"
``` 

and build away with the traits provided. Backend implementation (and other) PRs welcome!


### Built With

* [Spire](https://github.com/non/spire) - Typelevel project enabling generic numeric programming (For support for unsigned ints, complex numbers, the Numeric type class and type specialization to avoid boxing overhead)

* [Dotty](https://github.com/lampepfl/dotty) - A next-generation compiler that will become Scala 3 (For native union types, used here to express ONNX type constraints)

* [Singleton-ops](https://github.com/fthomas/singleton-ops) - Operations for primitive and String singleton types (For compile-time dimension checking)

* [Cats-effect](https://github.com/typelevel/cats-effect) - standard IO type together with Sync, Async and Effect type classes (Or your effect type here)

* [Freestyle](https://github.com/frees-io/freestyle) - pure functional framework for Free and Tagless Final apps & libs (For stack safety and parallelism without sacrificing composition) 

* [JavaCPP Preset for ONNX 1.3.0](https://github.com/bytedeco/javacpp-presets/tree/master/onnx) - The missing bridge between Java and native C++ libraries (For access to Protobuf definitions and operator schemas)

* [Scalameta](https://github.com/scalameta/scalameta) - Library to read, analyze, transform and generate Scala programs (For a runtime parse pass of generated programs)


### Inspiration

* [Neurocat](https://github.com/mandubian/neurocat) -  From neural networks to the Category of composable supervised learning algorithms in Scala with compile-time matrix checking based on singleton-types

* [Nexus](https://github.com/ctongfei/nexus) - Experimental typesafe tensors & deep learning in Scala

* [Backprop](https://github.com/mstksg/backprop) - Heterogeneous automatic differentiation ("backpropagation") in Haskell

* [Grenade](https://github.com/HuwCampbell/grenade) - Grenade is a composable, dependently typed, practical, and fast recurrent neural network library for concise and precise specifications of complex networks in Haskell.

* [DeepLearning.scala](https://github.com/ThoughtWorksInc/DeepLearning.scala) - A simple library for creating complex neural networks

* [Deeplearning4j / Scalnet / ND4S ](https://github.com/deeplearning4j/deeplearning4j/tree/master/scalnet) - ScalNet is a wrapper around Deeplearning4J emulating a Keras like API for deep learning. 
