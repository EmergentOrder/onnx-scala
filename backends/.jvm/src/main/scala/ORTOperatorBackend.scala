package org.emergentorder.onnx.backends

import java.nio.*
import scala.jdk.CollectionConverters.*
import scala.language.implicitConversions
import scala.util.Using
//import ai.onnxruntime.extensions.OrtxPackage
import org.emergentorder.onnx.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.Tensors.Tensor.*
import org.emergentorder.compiletime.*
import io.kjaer.compiletime.*
import onnx.onnx.*
import compiletime.asMatchable

import com.jyuzawa.onnxruntime.Environment;
import com.jyuzawa.onnxruntime.NamedCollection;
import com.jyuzawa.onnxruntime.OnnxRuntime;
import com.jyuzawa.onnxruntime.OnnxValue;
import com.jyuzawa.onnxruntime.Session;
import com.jyuzawa.onnxruntime.Transaction;
import com.jyuzawa.onnxruntime.OnnxTensor
import cats.implicits.*
import cats.effect.IO
import cats.effect.unsafe.implicits.global
import ORTTensorUtils.*

trait ORTOperatorBackend extends OpToONNXBytesConverter with AutoCloseable {

   val env = OnnxRuntime.get().getApi().newEnvironment().build()
   val coreCount = java.lang.Runtime.getRuntime().availableProcessors()
   def getSession(bytes: Array[Byte]) = {
      // Can now set symbolic dimension values, but only at session creation time
//      val session_options = new OrtSession.SessionOptions()
//      session_options.addCPU(false)
//      session_options.setMemoryPatternOptimization(true)
//      session_options.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
//      session_options.setIntraOpNumThreads(coreCount)
//    session_options.addCUDA()
//    session_options.addDnnl(true)
//      session_options.addXnnpack(java.util.Collections.emptyMap())
//      env.createSession(bytes, session_options)

     //Missing options here
      env.newSession().setByteArray(bytes).build()
   }

   def runModel[
       T <: Supported,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
       sess: Session,
       inputs: Tuple, 
       inputNames: List[String],
       outputNames: List[String]
   )(using
       tt: ValueOf[Tt],
       td: TensorShapeDenotationOf[Td],
       s: ShapeOf[S]
   ): Tensor[T, Tuple3[Tt, Td, S]] = {
      // TODO: More outputs / handle via ONNXSequence / ONNXMap

      val shapeFromType: S              = s.value
      val tensorTypeDenotationFromType  = tt.value
      val tensorShapeDenotationFromType = td.value

      val tensArr: IO[Array[T]] = cats.effect.Resource
         .make(IO.blocking { sess.newTransaction().build()})(txn => IO { txn.close })
         .use((txn: Transaction) => {

         (inputs.toArray zip inputNames)
            .flatMap { elem =>
               val inTens = txn.addInput(elem._2).asTensor
               elem._1.asInstanceOf[Option[Tensor[T, Tuple3[Tt, Td, S]]] | Tensor[T, Tuple3[Tt, Td, S]]] match {
                  case opt: Option[Tensor[T, Tuple3[Tt, Td, S]]] =>
                     opt match {
                        case Some(x) =>
                           Some(x.map { y =>
                              putArrayIntoOnnxTensor(y._1, y._2._3.toSeq.toArray, inTens)
                           })
                        case None => None
                     }
                  case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                     Some(tens.map { x =>
                        putArrayIntoOnnxTensor(x._1, x._2._3.toSeq.toArray, inTens)
                     })
               }
            }
            txn.addOutput(0)
            val result = txn.run()
            val firstOut = result.get(0).asTensor().asInstanceOf[OnnxTensor]
            val shape    = firstOut.getInfo.getShape.asScala.map(_.toInt)

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

      //Redundant
      // Spurious warning here, see: https://github.com/lampepfl/dotty/issues/10318
      // TODO: don't mix up Options and Tensors here
      @annotation.nowarn
      val inputTypeAndShape: IO[Array[Tuple2[Int, Array[Int]]]] = {

         inputs.toArray
            .flatMap { elem =>
               elem match {
                  case opt: Option[Tensor[T, Tuple3[Tt, Td, S]]] =>
                     opt match {
                        case Some(x) =>
                           Some(x.map { y =>
                              val dataType = y._1(0).asMatchable match {
                                case i: Int => 6
                                case f: Float => 10
                                case l: Long => 8
                                case d: Double => 11
                                case s: Short => 4
                                case b: Boolean => 13
                                case bb: Byte => 2
                                case _ => 0
                              }
                              (dataType, y._2._3.toSeq.toArray)
                           })
                        case None => None
                     }
                  case tens: Tensor[T, Tuple3[Tt, Td, S]] =>
                     Some(tens.map { x =>
                              val dataType = x._1(0).asMatchable match {
                                case i: Int => 6
                                case f: Float => 10
                                case l: Long => 8
                                case d: Double => 11
                                case s: Short => 4
                                case b: Boolean => 13
                                case bb: Byte => 2
                                case _ => 0
                              }
                              (dataType, x._2._3.toSeq.toArray)
                     })
               }
            }
            .toList
            .sequence
            .map(_.toArray)
      }

      def res(
          opModelBytes: Array[Byte], 
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
//         cats.effect.Resource
//            .make(inputTensorss)(inTens => IO { })
//            .use(inTens =>
               cats.effect.Resource
                  .make(IO.blocking(getSession(opModelBytes)))(sess => IO { sess.close })
                  .use(sess =>
                     runModel(
                       sess,
                       inputs,
                       input_node_names,
                       output_node_names
                     )
//                  )
            )
      }
      val resFinal = for {
         tens <- inputTypeAndShape.memoize
         t    <- tens
      } yield res(
        opToModelProto(
          opName,
          (t.map( x => x._1 match {
            //ORT has two different enums for this for the Java and C APIs 
            //Neither matches the ONNX spec
            case 2 => 3
            case 4 => 5
            case 10 => 1
            case 8 => 7
            case 13 => 9
            case n => n
          }
          )
            ) 

            zip 
            { t.map(x => x._2 match {
              //ORT shape inference diverges from the ONNX spec in requiring a scalar here instead of a tensor with shape,
              //causing a crash without this fix
              case Array(1) => if(opName.equals("Dropout")) Array[Int]() else Array(1)
              case y: Array[Int] => y
            }
            )
          },
          attrs
        ).toByteArray
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
   //TODO: now that this is otherwise working, try memoizing here
      result.flatMap(IO.println("Real call opName => " + opName).as(_))
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
