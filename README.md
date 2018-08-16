# ONNX-Scala

This project currently provides a complete, versioned, numerically generic, type-safe / typeful API to ONNX(Open Neural Network eXchange, an open format to represent deep machine learning models), derived from the Protobuf definitions and the operator schemas (defined in C++) via the JavaCPP Preset for ONNX.

ONNX-Scala will also be home to A) a program generator, capable of translating .onnx files into Scala programs written in terms of this API and B) at least one, eventually many backend implementations of ONNX accessible from the JVM.

All together, these should enable model inspection and modification, extra compile-time assurances, mixing/matching of backend operator implementations and integration into JVM-based production systems, for a start.

Balancing the interests of minimal imposition of dependencies with stack-safe, purely functional programming, ONNX-Scala comes in two flavors: Vanilla and Freestyle-infused.

Currently at ONNX 1.2.2.


## Getting Started

You'll need sbt.

```
sbt compile
```

Then, build away with the traits provided. Backend implementation (and other) PRs welcome!

### Built With

Spire - Typelevel project enabling generic numeric programming
