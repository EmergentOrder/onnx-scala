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

class ONNXNGraphHandlers(onnxBytes: Array[Byte])
    extends SigmoidZIO
    with GemmZIO
    with GatherZIO
    with MulZIO
    with ReluZIO
    with ConcatZIO
    with DropoutZIO
//    with DataSourceZIO
    with AutoCloseable {
  val scope         = new PointerScope()
  val ngraphBackend = new NGraphBackend(onnxBytes)

  def fullModel[@sp T: ClassTag, T1: ClassTag, T2: ClassTag, T3: ClassTag](
      A: Option[Tensor[T]],
      B: Option[Tensor[T1]],
      C: Option[Tensor[T2]]
  ): (Tensor[T3]) = {
    ngraphBackend.fullModel[T, T1, T2, T3](A, B, C)
  }

  /*
  def getParamsZIO[T: Numeric: ClassTag](name: String)(
      implicit ev: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): Task[Tensor[T]] = {
    Task {
      ngraphBackend.getParams(name)
    }

  }
   */
  def getAttributesZIO[T: Numeric: ClassTag](name: String)(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[
          Short
        ]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): Task[Tensor[T]] = ???

  override def Relu1ZIO[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])],
      X: Option[Tensor[T]]
  )(
      implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
  ): Task[(Tensor[T])] = Task {
    ngraphBackend.Relu1(name, consumed_inputs, X)
  }

  override def Relu6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      X: Option[org.emergentorder.onnx.Tensor[T]]
  )
  //(
  (
      implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
  ): Task[(Tensor[T])] = {
    Task {
      ngraphBackend.Relu6[T](name, X)
    }
  }

  def Dropout1ZIO[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])] = None,
      is_test: Option[(Int)] = None,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
  ): Task[(Tensor[T], Tensor[T])] = ???

  def Dropout6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      is_test: Option[(Int)] = None,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
  ): Task[(Tensor[T], Tensor[T])] = ???

  def Dropout10ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create],
      evT1: Contains[T1, Union[Boolean]#or[UNil]#create]
  ): Task[(Tensor[T], Tensor[T1])] = ???

  override def Dropout7ZIO[@sp T: Numeric: ClassTag](
      name: String,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
  ): Task[(Tensor[T], Tensor[T])] =
    Task {
      ngraphBackend.Dropout7[T](name, None, data)
    }

  override def Gather1ZIO[
      @sp T: Numeric: ClassTag,
      @sp Tind: Numeric: ClassTag
  ](
      name: String,
      axis: Option[(Int)] = None,
      data: Option[Tensor[T]],
      indices: Option[Tensor[Tind]]
  )(
      implicit evT: Contains[T, Union[UByte]#or[UShort]#or[UInt]#or[ULong]#or[Byte]#or[Short]#or[
        Int
      ]#or[Long]#or[Float16]#or[Float]#or[Double]#or[String]#or[Boolean]#or[Complex[Float]]#or[
        Complex[Double]
      ]#or[UNil]#create],
      evTind: Contains[Tind, Union[Int]#or[Long]#or[UNil]#create]
  ): Task[(Tensor[T])] =
    Task {
      ngraphBackend.Gather1[T, Tind](name, axis, data, indices)
    }

  def Mul1ZIO[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      consumed_inputs: Option[(Array[Int])] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): Task[(Tensor[T])] = ???

  def Mul6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): Task[(Tensor[T])] = ???

  override def Mul7ZIO[@sp T: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): Task[(Tensor[T])] =
    Task {
      ngraphBackend.Mul7[T](name, A, B)
    }

  def Gemm1ZIO[@sp T: Numeric: ClassTag](
      name: String,
      alpha: Option[(Float)] = None,
      beta: Option[(Float)] = None,
      broadcast: Option[(Int)] = None,
      transA: Option[(Int)] = None,
      transB: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]],
      C: Option[Tensor[T]]
  )(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): Task[(Tensor[T])] = ???

  def Gemm6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      alpha: Option[(Float)] = None,
      beta: Option[(Float)] = None,
      broadcast: Option[(Int)] = None,
      transA: Option[(Int)] = None,
      transB: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]],
      C: Option[Tensor[T]]
  )(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): Task[(Tensor[T])] = ???

  def Gemm7ZIO[@sp T: Numeric: ClassTag](
      name: String,
      alpha: Option[(Float)] = None,
      beta: Option[(Float)] = None,
      transA: Option[(Int)] = None,
      transB: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]],
      C: Option[Tensor[T]]
  )(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): Task[(Tensor[T])] = ???

  override def Gemm9ZIO[@sp T: Numeric: ClassTag](
      name: String,
      alpha: Option[(Float)] = None,
      beta: Option[(Float)] = None,
      transA: Option[(Int)] = None,
      transB: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]],
      C: Option[Tensor[T]]
  )(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UInt]#or[ULong]#or[Int]#or[Long]#or[UNil]#create
      ]
  ): Task[(Tensor[T])] =
    Task {
      ngraphBackend.Gemm9[T](name, alpha, beta, transA, transB, A, B, C)
    }

  def Sigmoid1ZIO[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])] = None,
      X: Option[Tensor[T]]
  )(
      implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
  ): Task[(Tensor[T])] = ???

  override def Sigmoid6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      X: Option[Tensor[T]]
  )(
      implicit evT: Contains[T, Union[Float16]#or[Float]#or[Double]#or[UNil]#create]
  ): Task[(Tensor[T])] =
    Task {
      ngraphBackend.Sigmoid6[T](name, X)
    }

  override def Concat4ZIO[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)],
      inputs: Seq[Option[Tensor[T]]]
  )(
      implicit evT: Contains[
        T,
        Union[Float16]#or[Float]#or[Double]#or[UByte]#or[UShort]#or[UInt]#or[
          ULong
        ]#or[Byte]#or[Short]#or[Int]#or[Long]#or[String]#or[Boolean]#or[Complex[Float]]#or[Complex[
          Double
        ]]#or[UNil]#create
      ]
  ): Task[(Tensor[T])] = {

    Task {
      ngraphBackend.Concat4[T](name, axis, inputs)
    }
  }

  override def close(): Unit = {
    ngraphBackend.close
    scope.close
  }

}

object ZIONGraphMain extends App {

  val scope          = new PointerScope()
  val dummyArraySize = 2000000

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

  val userIds                       = userIdsMap.keys.toArray
  val itemIds                       = itemIdsMap.keys.toArray
  def getRandomId(arr: Array[Long]) = arr(Random.nextInt(arr.length))

  val input = Task {
    val tens = TensorFactory.getTensor(
      (0 until dummyArraySize).toArray.map(x => getRandomId(userIds)),
      Array(dummyArraySize)
    )
    tens
  }
  val input2 = Task {
    val tens = TensorFactory.getTensor(
      (0 until dummyArraySize).toArray.map(x => getRandomId(itemIds)),
      Array(dummyArraySize)
    )
    tens
  }

//  val scope = new PointerScope()
//  val finalizer = UIO.effectTotal(scope.close)
  val ncfZIO = new NCFZIO(byteArray, userIdsMap, itemIdsMap)

//  def program = ncfZIO.fullNCF(input, input2)
  //val scope = Task(new PointerScope())
  //def close(scope: PointerScope) = UIO.effectTotal(scope.close)
  //val scopedProgram = scope.bracket(close(_)) { scope =>
  //  program
  // }
  val runtime = new DefaultRuntime {}

//  val scope = new PointerScope()
  val before  = System.nanoTime
  val output2 = runtime.unsafeRun(ncfZIO.fullNCF(input, input2))

  val after = System.nanoTime
  println("Elapsed: " + (after - before))
  //scope.close
  ncfZIO.close

  //Pointer.close
//  System.gc()
//  Pointer.deallocateReferences()
  println(Pointer.totalBytes)
  println(Pointer.physicalBytes)
  // scope.close
  // scope.close
  println(Pointer.maxPhysicalBytes)
  println("Output size: " + output2._1.size)
  println("Output 0: " + output2._1(0))
  println("Output 1: " + output2._1(7999)) //TODO: Investigate output flipping here, possibly due to race
//   println("Output 2: " + output2._1(2))
//   println("Output 3: " + output2._1(3))
//   println("Output 4: " + output2._1(4))

}
