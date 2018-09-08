# ONNX-Scala

This project currently provides a complete, versioned, numerically generic, type-safe / typeful API to ONNX(Open Neural Network eXchange, an open format to represent deep machine learning models), derived from the Protobuf definitions and the operator schemas (defined in C++) via the JavaCPP Preset for ONNX.

ONNX-Scala will also be home to A) a program generator, capable of translating .onnx files into Scala programs written in terms of this API and B) at least one, eventually many backend implementations of ONNX accessible from the JVM.

All together, these should enable model inspection and modification, extra compile-time assurances, mixing/matching of backend operator implementations and integration into JVM-based production systems, for a start.

Balancing the interests of minimal imposition of dependencies with stack-safe, purely functional programming, ONNX-Scala comes in two flavors: Vanilla and Freestyle-infused.

ONNX-Scala is cross-built against Scala.js / JavaScript.

To take advantage of union types to express type constraints, a Dotty (Scala 3) build is available. The Dotty build does not support Freestyle or JS variants.

Freestyle is likely to be replaced here soon with cats-free due to A) not supporting rewriting of unboxed union types workaround for Scala 2.x and B) its dependency on the EOLed scalameta paradise compiler plugin.

Currently at ONNX 1.2.2.


## Getting Started

You'll need sbt.

```
sbt compile package publishLocal
```

Then you can add this to your project's build.sbt 

```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala" % "1.2.2-0.1.0-SNAPSHOT"
```

or 

```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala-freestyle" % "1.2.2-0.1.0-SNAPSHOT"
``` 

or


```scala
libraryDependencies += "org.emergentorder.onnx" %% "onnx-scala-dotty" % "1.2.2-0.1.0-SNAPSHOT"
```

and build away with the traits provided. Backend implementation (and other) PRs welcome!

### Built With

Spire - Typelevel project enabling generic numeric programming
