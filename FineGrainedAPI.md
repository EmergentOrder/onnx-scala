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
Note that in real use backends should be closed to prevent native memory leaks

### Fine-grained API

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
