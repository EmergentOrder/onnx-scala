package org.emergentorder.onnx.backends

import java.nio.*
import scala.jdk.CollectionConverters.*
import scala.language.implicitConversions
import scala.util.Using
import ai.onnxruntime.*
import ai.onnxruntime.TensorInfo.OnnxTensorType
//import ai.onnxruntime.extensions.OrtxPackage
import org.emergentorder.onnx.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.Tensors.Tensor.*
import org.emergentorder.compiletime.*
import org.emergentorder.io.kjaer.compiletime.*
import onnx.onnx.*

import cats.implicits.*
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import ORTTensorUtils.*

trait ORTOperatorBackend extends OpToONNXBytesConverter with AutoCloseable {

   val env = OrtEnvironment.getEnvironment()

   val coreCount = java.lang.Runtime.getRuntime().availableProcessors()
   def getSession(bytes: Array[Byte]) = {
      // Can now set symbolic dimension values, but only at session creation time
      val session_options = new OrtSession.SessionOptions()
//      session_options.addCPU(false)
//      session_options.setMemoryPatternOptimization(true)
//      session_options.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
      session_options.setIntraOpNumThreads(coreCount)
//    session_options.addCUDA()
//    session_options.addDnnl(true)
//      session_options.addXnnpack(java.util.Collections.emptyMap())
      env.createSession(bytes, session_options)
   }

   def runModel[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       sess: OrtSession,
       input_tensor_values: Array[OnnxTensor],
       inputNames: List[String],
       outputNames: List[String]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      val inputs = (inputNames zip input_tensor_values).toMap.asJava
      // TODO: More outputs / handle via ONNXSequence / ONNXMap

      val shapeFromType: S              = s.value
      val tensorTypeDenotationFromType  = tt.value
      val tensorShapeDenotationFromType = td.value

      val tensArr: IO[Array[T]] = cats.effect.Resource
         .make(IO.blocking { sess.run(inputs) })(outTens => IO { outTens.close })
         .use(outTens => {
            val firstOut = outTens.get(0).asInstanceOf[OnnxTensor]
            val shape    = firstOut.getInfo.getShape.map(_.toInt)

            require(shape sameElements shapeFromType.toSeq)
            IO.blocking { getArrayFromOnnxTensor(firstOut) }
         })

      // TODO: Denotations
      val result: Tensor[T, Tuple3[Tt, Td, S]] = tensArr
         .flatMap(x =>
            Tensor(
              x,
              tensorTypeDenotationFromType,
              tensorShapeDenotationFromType,
              shapeFromType
            )
         )
      // result.flatMap(IO.println("Invoking run").as(_))
      result
   }

   // Idea: prepopulate models for ops with no params
   def callByteArrayOp[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       inputs: Tuple,
       input_node_names: List[String],
       opName: String,
       attrs: Map[String, Any]
   )(using
       s: ShapeOf[S],
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      // TODO: more outputs
      val output_node_names = List(inputs.size.toString)

      // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
      // TODO: don't mix up Options and Tensors here
      @annotation.nowarn
      val inputTensors: IO[Array[OnnxTensor]] = {

         inputs.toArray
            .flatMap { elem =>
               elem match {
                  case opt: Option[Tensor[T, Tuple3[Tt, Td, S]]] =>
                     opt match {
                        case Some(x) =>
                           Some(x.map { y =>
                              getOnnxTensor(y._1, y._2._3.toSeq.toArray, env)
                           })
                        case None => None
                     }
                  case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                     Some(tens.map { x =>
                        getOnnxTensor(x._1, x._2._3.toSeq.toArray, env)
                     })
               }
            }
            .toList
            .sequence
            .map(_.toArray)
      }

      def res(
          opModelBytes: Array[Byte],
          inputTensorss: IO[Array[OnnxTensor]]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         cats.effect.Resource
            .make(inputTensorss)(inTens => IO { inTens.map(_.close) })
            .use(inTens =>
               cats.effect.Resource
                  .make(IO.blocking(getSession(opModelBytes)))(sess => IO { sess.close })
                  .use(sess =>
                     runModel(
                       sess,
                       inTens,
                       input_node_names,
                       output_node_names
                     )
                  )
            )
      }

      val resFinal = for {
         tens <- inputTensors.memoize
         t    <- tens
      } yield res(
        opToModelProto(
          opName,
          (t.map(_.getInfo.onnxType.value match {
             // ORT has two different enums for this for the Java and C APIs
             // Neither matches the ONNX spec
             case 2  => 3
             case 4  => 5
             case 10 => 1
             case 8  => 7
             case 13 => 9
             case n  => n
          })

             zip {
                t.map(_.getInfo.getShape.map(_.toInt) match {
                   // ORT shape inference diverges from the ONNX spec in requiring a scalar here instead of a tensor with shape,
                   // causing a crash without this fix
                   case Array(1)      => if (opName.equals("Dropout")) Array[Int]() else Array(1)
                   case y: Array[Int] => y
                })
             }),
          attrs
        ).toByteArray,
        tens
      )

      // res.flatMap(IO.println("Post run").as(_))
      resFinal.flatten
   }

   def callOp[T <: Supported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](
       name: String,
       opName: String,
       inputs: Tuple,
       //    outName: String,
       attrs: Map[String, Any]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      val inputNodeNames = (0 until inputs.size).toList.map(_.toString)

      val result: Tensor[T, Tuple3[Tt, Td, S]] =
         callByteArrayOp(
           inputs,
           inputNodeNames,
           opName,
           attrs
         )
      // Using unsafeRunSync here to restore eager evaluation
      // and avoid redundant op invocations in case user code refers to Tensors more than once
      result.memoize.unsafeRunSync()
      // .flatMap(IO.println("Real call opName => " + opName).as(_))
   }

   def modelToPersist(mod: ModelProto, outName: String) = {
      val outNode      = mod.getGraph.node(0).clearOutput.withOutput(Seq(outName))
      val outInfoProto = mod.getGraph.output(0).clearName.withName(outName)
      val graphToPersist =
         mod.getGraph.clearNode.withNode(Seq(outNode)).clearOutput.withOutput(Seq(outInfoProto))
      mod.clearGraph.withGraph(graphToPersist)
   }

   override def close(): Unit = {}
}
