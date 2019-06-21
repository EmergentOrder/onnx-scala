package org.emergentorder.onnxZIO

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
import zio.DefaultRuntime

import org.emergentorder.onnx._
import org.emergentorder.onnxZIO._
import org.emergentorder.union.UnionType._

import org.emergentorder.onnx.backends._

class ONNXNGraphHandlers(onnxHelper: ONNXHelper)
    extends SigmoidZIO
    with GemmZIO
    with GatherZIO
    with MulZIO
    with ReluZIO
    with ConcatZIO
    with DropoutZIO
    with DataSourceZIO {

  //TODO: Each time the effect loads, it loads the model: Inject Onnxhelper?
  val ngraphBackend = new NGraphBackend(onnxHelper)

  val dummyArraySize = 70000

  def getParamsZIO[T: Numeric: ClassTag](name: String)(
      implicit ev: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): Task[Tensor[T]] = {
    Task {
      ngraphBackend.getParams(name)
    }

  }

  def getAttributesZIO[T: Numeric: ClassTag](name: String)(
      implicit ev: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): Task[Tensor[T]] = ???

  override def Relu1ZIO[@sp T: Numeric: ClassTag](
      name: String,
      consumed_inputs: Option[(Array[Int])],
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): Task[(Tensor[T])] = Task {
    ngraphBackend.Relu1(name, consumed_inputs, X)
  }

  override def Relu6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      X: Option[org.emergentorder.onnx.Tensor[T]]
  )
  //(
  (
      implicit ev: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
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
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): Task[(Tensor[T], Tensor[T])] = ???

  def Dropout6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      is_test: Option[(Int)] = None,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): Task[(Tensor[T], Tensor[T])] = ???

  def Dropout10ZIO[@sp T: Numeric: ClassTag, @sp T1: Numeric: ClassTag](
      name: String,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],
      evT1: (UNil TypeOr Boolean)#check[T1]
  ): Task[(Tensor[T], Tensor[T1])] = ???

  override def Dropout7ZIO[@sp T: Numeric: ClassTag](
      name: String,
      ratio: Option[(Float)] = None,
      data: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
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
      implicit evT: (UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T],
      evTind: (UNil TypeOr Int TypeOr Long)#check[Tind]
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
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): Task[(Tensor[T])] = ???

  def Mul6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)] = None,
      broadcast: Option[(Int)] = None,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
      ]
  ): Task[(Tensor[T])] = ???

  override def Mul7ZIO[@sp T: Numeric: ClassTag](
      name: String,
      A: Option[Tensor[T]],
      B: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[
        T
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
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
        T
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
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
        T
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
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
        T
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
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[
        T
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
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): Task[(Tensor[T])] = ???

  override def Sigmoid6ZIO[@sp T: Numeric: ClassTag](
      name: String,
      X: Option[Tensor[T]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T]
  ): Task[(Tensor[T])] =
    Task {
      ngraphBackend.Sigmoid6[T](name, X)
    }

  override def Concat4ZIO[@sp T: Numeric: ClassTag](
      name: String,
      axis: Option[(Int)],
      inputs: Seq[Option[Tensor[T]]]
  )(
      implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[
        Float
      ] TypeOr Complex[Double])#check[T]
  ): Task[(Tensor[T])] = {

    Task {
      ngraphBackend.Concat4[T](name, axis, inputs)
    }
  }

}

object ZIONGraphMain extends App {

  val input = Task {
    //(Seq.fill(dummyArraySize)(Random.nextInt(10000)).toArray, Array(dummyArraySize)).asInstanceOf[Tensor[T]]
    (Array(0l, 10000l), Array(2))
  }
  val input2 = Task {
    //(Seq.fill(dummyArraySize)(Random.nextInt(10000)).toArray, Array(dummyArraySize)).asInstanceOf[Tensor[T]]
    (Array(0l, 10000l), Array(2))
  }

  def program = NCFZIO.program(input, input2)

  val runtime = new DefaultRuntime {}

  val before = System.nanoTime
  val output2 = runtime.unsafeRun(program)
  val after = System.nanoTime
  println("Elapsed: " + (after - before))

  println("Output size: " + output2._1.size)
  println("Output 0: " + output2._1(0))
  println("Output 1: " + output2._1(1)) //TODO: Investigate output flipping here, possibly due to race
//   println("Output 2: " + output2._1(2))
//   println("Output 3: " + output2._1(3))
//   println("Output 4: " + output2._1(4))

}
