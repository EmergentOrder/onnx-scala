package org.emergentorder.onnxZIO

import scala.reflect.io.Streamable
import scala.io.Source
import scala.util.Random
import scala.{specialized => sp}
import scala.collection.mutable.{Map => MMap};
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.Numeric
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex

import zio.Task
import zio.UIO
import zio.Managed
import zio.DefaultRuntime

import org.bytedeco.javacpp._
import org.emergentorder.onnx._
import org.emergentorder.onnxZIO._
import org.emergentorder.union._
import org.emergentorder.onnx.backends._

class ONNXNGraphHandlers(onnxBytes: Array[Byte]) extends AutoCloseable {
  val scope         = new PointerScope()
  val ngraphBackend = new NGraphModelBackend(onnxBytes)

  def fullModel[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag,
      T9: ClassTag,
      T10: ClassTag,
      T11: ClassTag,
      T12: ClassTag,
      T13: ClassTag,
      T14: ClassTag,
      T15: ClassTag,
      T16: ClassTag,
      T17: ClassTag
  ](
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8]
  ): (Task[T9]) = {
    Task {
      ngraphBackend
        .fullModel[T, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17](
          inputs
        )
    }
  }

  override def close(): Unit = {
    ngraphBackend.close
    scope.close
  }

}

object ZIONGraphMain extends App {

  val scope          = new PointerScope()
  val dummyArraySize = 100000 //Output size is determined by model

  val byteStream = getClass.getResourceAsStream("/" + "NCF.onnx")

  val byteArray = Streamable.bytes(
    byteStream
  ) // JAVA 9+ only : .readAllBytes()

  byteStream.close
  def getIdMap(idMapFilename: String) = {

    val mapStream    = getClass.getResourceAsStream("/" + idMapFilename)
    val idsMapSource = Source.fromInputStream(mapStream)

    val result = idsMapSource.getLines.toList
      .drop(1)
      .map { line =>
        val cols = line.split(",").map(_.trim)
        cols(1).toLong -> cols(2).toLong
      }
      .toMap
    mapStream.close
    result
  }

  val itemIdMapFilename = "itemIds.csv"
  val userIdMapFilename = "userIds.csv"

  val userIdsMap = getIdMap(userIdMapFilename)
  val itemIdsMap = getIdMap(itemIdMapFilename)

  val userIds = userIdsMap.keys.toArray
  val itemIds = itemIdsMap.keys.toArray
  val rand    = new Random(42)
  val rand1   = new Random(53)

  def getRandomId(arr: Array[Long], aRand: Random) = arr(aRand.nextInt(arr.length))

  val input = Task {
    val tens = TensorFactory.getTensor(
      (0 until dummyArraySize).toArray.map(x => getRandomId(userIds, rand)),
      Array(dummyArraySize)
    )
    tens
  }
  val input2 = Task {
    val tens = TensorFactory.getTensor(
      (0 until dummyArraySize).toArray.map(x => getRandomId(itemIds, rand1)),
      Array(dummyArraySize)
    )
    tens
  }

//  val scope = new PointerScope()
//  val finalizer = UIO.effectTotal(scope.close)

  val ncfZIO = new NCFZIO(byteArray, userIdsMap, itemIdsMap)

  val runtime = new DefaultRuntime {}

//  val scope = new PointerScope()
  val before  = System.nanoTime
  val output2 = runtime.unsafeRun(ncfZIO.fullNCF(input, input2))

  val after = System.nanoTime
  println("Elapsed: " + (after - before))

  ncfZIO.close

  //Pointer.close
//  System.gc()
//  Pointer.deallocateReferences()
  println(Pointer.totalBytes)
  println(Pointer.physicalBytes)
  println(Pointer.maxPhysicalBytes)
  println("Output size: " + output2._1.size)
  println("Output 0: " + output2._1(0))
  println("Output 7999: " + output2._1(7999))
//   println("Output 2: " + output2._1(2))
//   println("Output 3: " + output2._1(3))
//   println("Output 4: " + output2._1(4))

}
