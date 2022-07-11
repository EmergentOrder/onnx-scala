//> using scala "3.1.2"
//> using lib "org.emergent-order::onnx-scala-backends:0.16.0"

import java.nio.file.{Files, Paths}
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.backends._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._

// prepare tensor

// Input tensor has shape (1x1x28x28), with type of float32. One image at a time. This model doesn't support mini-batch.
val shape = 1 #: 1 #: 28 #: 28 #: SNil
val tensorShapeDenotation =
   "Batch" ##: "Channel" ##: "Width" ##: "Height" ##: TSNil
val tensorDenotation: String & Singleton = "Image"

// array of length 784 = 1 * 1 * 28 * 28
val data: Array[Float] = ???

val imageTens = Tensor(data, tensorDenotation, tensorShapeDenotation, shape)

// load model
val modelData: Array[Byte] = Files.readAllBytes(Paths.get("./mnist/model.onnx"))
val model                  = new ORTModelBackend(modelData)

// run
val out = model.fullModel[
  Float,
  "Batch",
  "Class" ##: "Digit" ##: TSNil,
  // The likelihood of each number before softmax, with shape of (1x10).
  1 #: 10 #: SNil
](Tuple(imageTens))
println(out.data.zipWithIndex.maxBy(_._1)._2)
