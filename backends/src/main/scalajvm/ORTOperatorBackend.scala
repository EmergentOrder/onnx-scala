package org.emergentorder.onnx.backends

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import cats.effect.IO
//import cats.effect.unsafe.implicits.global
import cats.implicits.*
import onnx.onnx.*
import org.emergentorder.compiletime.*
import org.emergentorder.io.kjaer.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.*

import scala.jdk.OptionConverters.RichOptional
import scala.jdk.CollectionConverters.*
import scala.language.implicitConversions
//import java.io.File
//import java.nio.ByteBuffer
import scala.annotation.nowarn
//import scala.Conversion.into
//import compiletime.asMatchable

import ORTTensorUtils.*

trait ORTOperatorBackend extends OpToONNXBytesConverter with AutoCloseable {

   val env: OrtEnvironment = OrtEnvironment.getEnvironment()

   val coreCount: Int = java.lang.Runtime.getRuntime().availableProcessors()
   def getSession(bytes: Array[Byte]): OrtSession = {
      // Can now set symbolic dimension values, but only at session creation time
      val session_options = new OrtSession.SessionOptions()
//      session_options.addCPU(false)
//      session_options.setMemoryPatternOptimization(true)
//      session_options.setCPUArenaAllocator(true)
//      session_options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
      session_options.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
      session_options.setIntraOpNumThreads(coreCount)
//    session_options.addCUDA()
//    session_options.addDnnl(true)
//      session_options.addConfigEntry("kOrtSessionOptionsConfigAllowIntraOpSpinning", "0")
//      session_options.addXnnpack(java.util.Collections.singletonMap("intra_op_num_threads", coreCount.toString))

      // For model compilation
      /*
      val modelBuffer = ByteBuffer.wrap(bytes)
      val compileOptions = OrtModelCompilationOptions.createFromSessionOptions(env, session_options)

      compileOptions.setInputModelFromBuffer(modelBuffer)
      compileOptions.setOutputModelPath("compiled_model.onnx")
      val f = new File("compiled_model.onnx")
      compileOptions.compileModel()

      env.createSession(f.toString, session_options)
       */
      env.createSession(bytes, session_options)
   }

   def runModel[
       T <: Supported: scala.reflect.Typeable,
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

      val outSize = shapeFromType.toSeq.reduce(_ * _)

      def f[X: scala.reflect.Typeable](in: Any): Option[X] =
         in match
            case x: X => Some(x)
            case _    => None

      def realSomet: Option[T] = Seq(
        f[T]('1'.toByte),
        f[T](1.toShort),
        f[T](1),
        f[T](1L),
        f[T](1f),
        f[T](1d),
        f[T](true),
        f[T]("1")
      ).filter(_.isDefined)(0)

      // TODO: more cases here, and refactor
      @nowarn
      val outputs: java.util.Map[String, OnnxTensor] = realSomet match {
         case Some(f: Float) =>
            val outputABuff: java.nio.FloatBuffer = java.nio.ByteBuffer
               .allocateDirect(outSize * 4)
               .order(java.nio.ByteOrder.nativeOrder())
               .asFloatBuffer();
            val outputA =
               OnnxTensor.createTensor(env, outputABuff, shapeFromType.toSeq.toArray.map(_.toLong))
            val pinnedOutputs: java.util.Map[String, OnnxTensor] =
               (new scala.collection.immutable.HashMap[String, OnnxTensor]() + (outputNames(
                 0
               ) -> outputA)).toMap.asJava
            // pinnedOutputs.put("output-0", outputA);

            // val fb = outputA.getFloatBuffer
            val outBuf: java.nio.FloatBuffer = outputA.getBufferRef().toScala match {
               case Some(x) =>
                  x match {
                     case fb: java.nio.FloatBuffer => fb
                     case _                        => throw new Exception("missing")
                  }
               case None => throw new Exception("missing")
            }

            if (!outBuf.isDirect)
               throw new Exception("Output A buff is not direct!!!")

            pinnedOutputs

         case Some(b: Byte) =>
            val outputABuff: java.nio.ByteBuffer =
               java.nio.ByteBuffer.allocateDirect(outSize).order(java.nio.ByteOrder.nativeOrder())
            val outputA =
               OnnxTensor.createTensor(env, outputABuff, shapeFromType.toSeq.toArray.map(_.toLong))
            val pinnedOutputs: java.util.Map[String, OnnxTensor] =
               (new scala.collection.immutable.HashMap[String, OnnxTensor]() + (outputNames(
                 0
               ) -> outputA)).toMap.asJava
            // pinnedOutputs.put("output-0", outputA);

            // val fb = outputA.getFloatBuffer
            val outBuf: java.nio.ByteBuffer = outputA.getBufferRef().toScala match {
               case Some(x) =>
                  x match {
                     case fb: java.nio.ByteBuffer => fb
                     case _                       => throw new Exception("missing")
                  }
               case None => throw new Exception("missing")
            }

            if (!outBuf.isDirect)
               throw new Exception("Output A buff is not direct!!!")

            pinnedOutputs

         case Some(b: Short) =>
            val outputABuff: java.nio.ShortBuffer = java.nio.ByteBuffer
               .allocateDirect(outSize * 2)
               .order(java.nio.ByteOrder.nativeOrder())
               .asShortBuffer
            val outputA =
               OnnxTensor.createTensor(env, outputABuff, shapeFromType.toSeq.toArray.map(_.toLong))
            val pinnedOutputs: java.util.Map[String, OnnxTensor] =
               (new scala.collection.immutable.HashMap[String, OnnxTensor]() + (outputNames(
                 0
               ) -> outputA)).toMap.asJava
            // pinnedOutputs.put("output-0", outputA);

            // val fb = outputA.getFloatBuffer
            val outBuf: java.nio.ShortBuffer = outputA.getBufferRef().toScala match {
               case Some(x) =>
                  x match {
                     case fb: java.nio.ShortBuffer => fb
                     case _                        => throw new Exception("missing")
                  }
               case None => throw new Exception("missing")
            }

            if (!outBuf.isDirect)
               throw new Exception("Output A buff is not direct!!!")

            pinnedOutputs

         case Some(f: Long) =>
            val outputABuff: java.nio.LongBuffer = java.nio.ByteBuffer
               .allocateDirect(outSize * 8)
               .order(java.nio.ByteOrder.nativeOrder())
               .asLongBuffer();
            val outputA =
               OnnxTensor.createTensor(env, outputABuff, shapeFromType.toSeq.toArray.map(_.toLong))
            val pinnedOutputs: java.util.Map[String, OnnxTensor] =
               (new scala.collection.immutable.HashMap[String, OnnxTensor]() + (outputNames(
                 0
               ) -> outputA)).toMap.asJava
            // pinnedOutputs.put("output-0", outputA);

            // val fb = outputA.getFloatBuffer
            val outBuf: java.nio.LongBuffer = outputA.getBufferRef().toScala match {
               case Some(x) =>
                  x match {
                     case fb: java.nio.LongBuffer => fb
                     case _                       => throw new Exception("missing")
                  }
               case None => throw new Exception("missing")
            }

            if (!outBuf.isDirect)
               throw new Exception("Output A buff is not direct!!!")

            pinnedOutputs

         case Some(f: Int) =>
            val outputABuff: java.nio.IntBuffer = java.nio.ByteBuffer
               .allocateDirect(outSize * 4)
               .order(java.nio.ByteOrder.nativeOrder())
               .asIntBuffer();
            val outputA =
               OnnxTensor.createTensor(env, outputABuff, shapeFromType.toSeq.toArray.map(_.toLong))
            val pinnedOutputs: java.util.Map[String, OnnxTensor] =
               (new scala.collection.immutable.HashMap[String, OnnxTensor]() + (outputNames(
                 0
               ) -> outputA)).toMap.asJava
            // pinnedOutputs.put("output-0", outputA);

            // val fb = outputA.getFloatBuffer
            val outBuf: java.nio.IntBuffer = outputA.getBufferRef().toScala match {
               case Some(x) =>
                  x match {
                     case fb: java.nio.IntBuffer => fb
                     case _                      => throw new Exception("missing")
                  }
               case None => throw new Exception("missing")
            }

            if (!outBuf.isDirect)
               throw new Exception("Output A buff is not direct!!!")

            pinnedOutputs

         case Some(f: Double) =>
            val outputABuff: java.nio.DoubleBuffer = java.nio.ByteBuffer
               .allocateDirect(outSize * 8)
               .order(java.nio.ByteOrder.nativeOrder())
               .asDoubleBuffer();
            val outputA =
               OnnxTensor.createTensor(env, outputABuff, shapeFromType.toSeq.toArray.map(_.toLong))
            val pinnedOutputs: java.util.Map[String, OnnxTensor] =
               (new scala.collection.immutable.HashMap[String, OnnxTensor]() + (outputNames(
                 0
               ) -> outputA)).toMap.asJava
            // pinnedOutputs.put("output-0", outputA);

            // val fb = outputA.getFloatBuffer
            val outBuf: java.nio.DoubleBuffer = outputA.getBufferRef().toScala match {
               case Some(x) =>
                  x match {
                     case fb: java.nio.DoubleBuffer => fb
                     case _                         => throw new Exception("missing")
                  }
               case None => throw new Exception("missing")
            }

            if (!outBuf.isDirect)
               throw new Exception("Output A buff is not direct!!!")

            pinnedOutputs

         // TODO:
         /*
        case Some(f: String) =>
          val outputABuff: java.nio.CharBuffer = java.nio.ByteBuffer.allocateDirect(outSize * f.length).order(java.nio.ByteOrder.nativeOrder()).asCharBuffer();
          val outputA = OnnxTensor.createTensor(env, outputABuff, shapeFromType.toSeq.toArray.map(_.toLong))
          val pinnedOutputs: java.util.Map[String, OnnxTensor] = (new scala.collection.immutable.HashMap[String, OnnxTensor]() + (outputNames(0) -> outputA)).toMap.asJava
          //pinnedOutputs.put("output-0", outputA);

          //val fb = outputA.getFloatBuffer
          val outBuf: java.nio.CharBuffer = outputA.getBufferRef().toScala match {
                   case Some(x) => x match {
                     case fb: java.nio.CharBuffer => fb
                     case _ => throw new Exception("missing")
                   }
                   case None => throw new Exception("missing")
          }

          if (!outBuf.isDirect)
            throw new Exception("Output A buff is not direct!!!")

          pinnedOutputs
          */
         case Some(b: Boolean) =>
            val outputABuff: java.nio.ByteBuffer =
               java.nio.ByteBuffer.allocateDirect(outSize).order(java.nio.ByteOrder.nativeOrder())
            val outputA = OnnxTensor.createTensor(
              env,
              outputABuff,
              shapeFromType.toSeq.toArray.map(_.toLong),
              ai.onnxruntime.OnnxJavaType.BOOL
            )
            val pinnedOutputs: java.util.Map[String, OnnxTensor] =
               (new scala.collection.immutable.HashMap[String, OnnxTensor]() + (outputNames(
                 0
               ) -> outputA)).toMap.asJava
            // pinnedOutputs.put("output-0", outputA);

            // val fb = outputA.getFloatBuffer
            val outBuf: java.nio.ByteBuffer = outputA.getBufferRef().toScala match {
               case Some(x) =>
                  x match {
                     case fb: java.nio.ByteBuffer => fb
                     case _                       => throw new Exception("missing")
                  }
               case None => throw new Exception("missing")
            }

            if (!outBuf.isDirect)
               throw new Exception("Output A buff is not direct!!!")

            pinnedOutputs

         case None => throw RuntimeException("tensor type T not found")
         case _    => throw RuntimeException("tensor type T not found")
         // case x:Any => {println(x);  java.util.Collections.emptyMap[String, OnnxTensor]()}
      }

      val tensArr: IO[Array[T]] = cats.effect.Resource
         .make(IO.blocking { sess.run(inputs, outputs) })(outTens => IO.blocking { outTens.close })
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
       T <: Supported: scala.reflect.Typeable,
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
      @nowarn
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
            .toSeq
            .sequence
            .map(_.toArray)
      }

      def res(
          opModelBytes: Array[Byte],
          inputTensorss: IO[Array[OnnxTensor]]
      ): Tensor[T, Tuple3[Tt, Td, S]] = {
         cats.effect.Resource
            .make(inputTensorss)(inTens => IO.blocking { inTens.map(_.close) })
            .use(inTens =>
               cats.effect.Resource
                  .make(IO.blocking(getSession(opModelBytes)))(sess => IO.blocking { sess.close })
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

      val resFinal = for
         tens <- inputTensors.memoize
         t    <- tens
      yield res(
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
                   case Array(1)      => if opName.equals("Dropout") then Array[Int]() else Array(1)
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

   def callOp[
       T <: Supported: scala.reflect.Typeable,
       Tt <: TensorTypeDenotation,
       Td <: TensorShapeDenotation,
       S <: Shape
   ](
//       name: String,
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
      val resultToReturn = result // result.memoize.unsafeRunSync()
      resultToReturn
      // .flatMap(IO.println("Real call opName => " + opName).as(_))
   }

   def modelToPersist(mod: ModelProto, outName: String): ModelProto = {
      val outNode        = mod.getGraph.node(0).clearOutput.withOutput(Array(outName))
      val outInfoProto   = mod.getGraph.output(0).withName(outName)
      val graphToPersist =
         mod.getGraph.clearNode.withNode(Array(outNode)).clearOutput.withOutput(Array(outInfoProto))
      mod.clearGraph.withGraph(graphToPersist)
   }

   override def close(): Unit = {
      env.close
   }
}
