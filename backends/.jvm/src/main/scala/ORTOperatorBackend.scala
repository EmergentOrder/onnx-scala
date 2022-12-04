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
import io.kjaer.compiletime.*
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
         .map(x =>
            Tensor(
              x,
              tensorTypeDenotationFromType,
              tensorShapeDenotationFromType,
              shapeFromType
            )
         )
         .unsafeRunSync()
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
       opModel: Array[Byte],
       inputs: Tuple,
       input_node_names: IO[List[String]]
   )(using
       s: ShapeOf[S],
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      /*
     val input_node_names = inputs.toArray.zipWithIndex.map { (e, i) =>
         val incr: String = if inputs.toArray.distinct.size == inputs.size then "" else i.toString
         val tensE = e.asInstanceOf[Tensor[T, Tuple3[Tt, Td, S]]]
         tensE.map{x =>
           val t = ((x.toString + incr).hashCode).toString
           println("ANESMMMS " + t + " " + i)
           t
         }
      }.toList.sequence
       */

      // TODO: more outputs
      val output_node_names = List(input_node_names.toString)

      // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
      // TODO: don't mix up Options and Tensors here
      @annotation.nowarn
      def inputTensors: IO[Array[OnnxTensor]] = {

         inputs.toArray
            .flatMap { elem =>
               elem match {
                  case opt: Option[Tensor[T, Tuple3[Tt, Td, S]]] =>
                     opt match {
                        case Some(x) =>
                           Some(x.data.flatMap { y =>
                              x.shape.map { z =>
                                 getOnnxTensor(y, z, env)
                              }
                           })
                        case None => None
                     }
                  case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                     Some(tens.data.flatMap { x =>
                        tens.shape.map { y =>
                           getOnnxTensor(x, y, env)
                        }
                     })
               }
            }
            .toList
            .sequence
            .map(_.toArray)
      }

      def res: Tensor[T, Tuple3[Tt, Td, S]] = {
//        val resource = cats.effect.Resource.make(IO{getSession(opModel)})(sess => IO{sess.close})
         // resource.use( sess =>
         cats.effect.Resource
            .make(inputTensors)(inTens => IO { inTens.map(_.close) })
            .use(inTens =>
               input_node_names.flatMap { y =>
                  cats.effect.Resource
                     .make(IO.blocking(getSession(opModel)))(sess => IO { sess.close })
                     .use(sess =>
                        IO {
                           runModel(
                             sess,
                             inTens,
                             y,
                             output_node_names
                           )
                        }
                     )
               }
            )
      }.unsafeRunSync()
      // res.flatMap(IO.println("Post run").as(_))
      res
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
      // TODO: prevent passing input to opToONNXBytes

      val modelProto = opToModelProto(opName, inputs, attrs)

//      val mp = opToModelProto(opName, inputs, attrs)

      val result: IO[Tensor[T, Tuple3[Tt, Td, S]]] =
         for {
            mp <- modelProto // modelProto.flatMap(IO.println("OpName => " + opName).as(_))
         } yield callByteArrayOp(
           mp.toByteArray,
           inputs,
           IO.pure {
              mp.graph.map(_.input.map(_.name.getOrElse(""))).getOrElse(List[String]()).toList
           }
         )
      val r =
         result.unsafeRunSync() // If don't use unsafe here, we get redundant callOp invocations. If we memoize w/ unsafe, we leak memory.
      r // This approach makes callOp sync/eager again.
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
