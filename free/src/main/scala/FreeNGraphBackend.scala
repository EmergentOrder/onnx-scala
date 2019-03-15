package org.emergentorder.onnxFree

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

import freestyle.free._
import freestyle.free.implicits._

import cats.free.Free
import cats.free.FreeApplicative
import cats.effect.IO


import org.emergentorder.onnx._
import org.emergentorder.onnxFree._
import org.emergentorder.onnx.UnionType._

import org.emergentorder.onnx.backends._

object ONNXNGraphHandlers extends App {

  implicit val datasourceHandler = new org.emergentorder.onnxFree.DataSourceFree.Handler[IO]{
    override def inputDataFree[T : Numeric : ClassTag]
//    (implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
    : IO[Tensor[T]] = {
      IO{
        (Array(-1.0f, 0.5f, 1.0f, 0.75f, -1000f), Array(1,5)).asInstanceOf[Tensor[T]]
      }
    }
  }
  
  implicit val reluHandler = new org.emergentorder.onnxFree.ReluFree.Handler[IO] {
/*
  override def Relu1Free[@sp T: Numeric: ClassTag](name: String,
                                        consumed_inputs: Option[(Array[Int])],
                                        X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : IO[(Tensor[T])] = IO{
        (new NGraphBackend).Relu1(name, consumed_inputs, X)
      }
  */

  override def Relu6Free[@sp T : Numeric: ClassTag](name: String, X: Option[org.emergentorder.onnx.Tensor[T]])
  //(
  //      (implicit ev: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : IO[(Tensor[T])] = {
        IO{
          (new NGraphBackend).Relu6[T](name, X)
        }
      }
}

  implicit val dropoutHandler = new org.emergentorder.onnxFree.DropoutFree.Handler[IO] {


      override def Dropout7Free[@sp T : Numeric:ClassTag](name: String,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
//      (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : IO[(Tensor[T], Tensor[T])] = 
        IO{
          (new NGraphBackend).Dropout7[T](name, None, data)
        }

  }



  def program[T: Numeric : ClassTag]
//  (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
  : FreeS[Single_reluFree.Op,(Tensor[T])] = Single_reluFree.instance.program[T]

//   def output = program[Float].interpret[IO].unsafeRunSync
   val output = program[Float].interpret[IO].unsafeRunSync
   println("Output size: " + output._1.size)
   println("Output 0: " + output._1(0))
   println("Output 1: " + output._1(1))
   println("Output 2: " + output._1(2))
   println("Output 3: " + output._1(3))
   println("Output 4: " + output._1(4))
}
