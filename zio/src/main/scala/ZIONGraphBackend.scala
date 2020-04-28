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
import zio.Runtime

import org.bytedeco.javacpp._
import org.emergentorder.onnx._
import org.emergentorder.onnxZIO._
import org.emergentorder.union._
import org.emergentorder.onnx.backends._

class ONNXORTHandlers(onnxBytes: Array[Byte]) extends AutoCloseable {
  val scope         = new PointerScope()
  val ngraphBackend = new ORTModelBackend(onnxBytes)

  def fullModel[T: ClassTag](
      inputs: Option[NonEmptyTuple]
  ): (Task[Tuple1[Tensor[T]]]) = {
    Task {
      ngraphBackend
        .fullModel[Tensor[T]](
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

  val runtime = Runtime.default

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
  println("Output 0: " + output2._1(0)(0))
  println("Output 7999: " + output2._1(0)(7999))
//   println("Output 2: " + output2._1(2))
//   println("Output 3: " + output2._1(3))
//   println("Output 4: " + output2._1(4))

}
