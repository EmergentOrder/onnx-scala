# ONNX-Scala

This project currently provides a complete, versioned, numerically generic, type-safe / typeful API to ONNX(Open Neural Network eXchange, an open format to represent deep learning and classical machine learning models), derived from the Protobuf definitions and the operator schemas (defined in C++) via the JavaCPP Preset for ONNX.

ONNX-Scala will also be home to A) a program generator, capable of translating .onnx files into Scala programs written in terms of this API and B) at least one, eventually many backend implementations of ONNX accessible from the JVM.

All together, these should enable model inspection and modification, extra compile-time assurances, mixing/matching of backend operator implementations and integration into JVM-based production systems, for a start.

Balancing the interests of minimal imposition of dependencies with stack-safe, purely functional programming, ONNX-Scala comes in two flavors: Vanilla and cats-free-infused.

ONNX-Scala is cross-built against Scala JVM (for both Scala 2.12 and 2.13.0-M4) and Scala.js / JavaScript (for Scala 2.12).

To take advantage of union types to express type constraints, a Dotty (Scala 3) build is available. The Dotty build does not support Scala.js.

Freestyle has been replaced with cats-free due to A) not supporting rewriting of unboxed union types workaround for Scala 2.x and B) its dependency on the EOLed scalameta paradise compiler plugin.

Currently at ONNX 1.2.2.


## Getting Started

You'll need sbt.

```
sbt +publishLocal
```

Then you can add this to your project's build.sbt 

```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala" % "1.2.2-0.1.0-SNAPSHOT"
```

or 

```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala-free" % "1.2.2-0.1.0-SNAPSHOT"
``` 

and build away with the traits provided. Backend implementation (and other) PRs welcome!

### Built With

Spire - Typelevel project enabling generic numeric programming (For support for unsigned ints, complex numbers, the Numeric type class and type specialization to avoid boxing overhead)
Dotty - A next-generation compiler that will become Scala 3 (For native union types, used here to express ONNX type constraints)
Singleton-ops - Operations for primitive and String singleton types (For compile-time dimension checking)
Cats-effect - standard IO type together with Sync, Async and Effect type classes (Or your effect type here)
Cats-free - Free structures such as the free monad, and supporting type classes (for the free variant)
