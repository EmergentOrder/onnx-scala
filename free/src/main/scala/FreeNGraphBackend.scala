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

   val fileName = "squeezenet.onnx" //Needed for params at runtime //TODO: inject
   val onnxHelper = new ONNXHelper(fileName)
   val ngraphBackend = new NGraphBackend(onnxHelper)

  //TODO: Fix concurrency problem caused by protobuf sharedDtor

  implicit val datasourceHandler = new org.emergentorder.onnxFree.DataSourceFree.Handler[IO]{
    override def inputDataFree[T: Numeric: ClassTag]
//    (implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
    : IO[Tensor[T]] = {
      IO{
        //TODO
        (Array(0f, 220f, -2200f, 5000f, 10000f), Array(5)).asInstanceOf[Tensor[T]]
      }
    }

    def getParamsFree[ T : Numeric : ClassTag](name: String)
    //(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
    : IO[Tensor[T]] = {
       IO{ 
           ngraphBackend.getParamsFree(name)
      } 

    }

  }

   implicit val gatherHandler = new org.emergentorder.onnxFree.GatherFree.Handler[IO] {


     override def Gather1Free[@sp T : Numeric:ClassTag,@sp Tind : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,data: Option[Tensor[T]], indices: Option[Tensor[Tind]])
//(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])    
      : IO[(Tensor[T])] = 
        IO{
          ngraphBackend.Gather1[T, Tind](name, axis, data, indices)
        }

  }


  implicit val mulHandler = new org.emergentorder.onnxFree.MulFree.Handler[IO] {
    override def Mul7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
    : IO[(Tensor[T])] = 
      IO{
        ngraphBackend.Mul7[T](name, A, B)
      }
  }

  implicit val gemmHandler = new org.emergentorder.onnxFree.GemmFree.Handler[IO] {
     override def Gemm9Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
//(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    
  : IO[(Tensor[T])] =
    IO{
        ngraphBackend.Gemm9[T](name, alpha, beta, transA, transB, A, B, C)
    }
  }    


  implicit val sigmoidHandler = new org.emergentorder.onnxFree.SigmoidFree.Handler[IO] {
    override   def Sigmoid6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
//(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    
  : IO[(Tensor[T])] =
    IO{
      ngraphBackend.Sigmoid6[T](name, X)
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
          ngraphBackend.Relu6[T](name, X)
        }
      }
}

 implicit val concatHandler = new org.emergentorder.onnxFree.ConcatFree.Handler[IO] {
     override def Concat4Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)],inputs: Seq[Option[Tensor[T]]])
//(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
  : IO[(Tensor[T])] = {

    IO{
    ngraphBackend.Concat4[T](name, axis, inputs)
    }
  }

 }

  implicit val dropoutHandler = new org.emergentorder.onnxFree.DropoutFree.Handler[IO] {


      override def Dropout7Free[@sp T : Numeric:ClassTag](name: String,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
//      (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : IO[(Tensor[T], Tensor[T])] = 
        IO{
          ngraphBackend.Dropout7[T](name, None, data)
        }

  }



//  def program[T: Numeric : ClassTag]
//  (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
//  : FreeS[Single_reluFree.Op,(Tensor[T])] = Single_reluFree.instance.program[T]

  /*
//   def output = program[Float].interpret[IO].unsafeRunSync
   val output = program[Float].interpret[IO].unsafeRunSync
   println("Output size: " + output._1.size)
   println("Output 0: " + output._1(0))
   println("Output 1: " + output._1(1))
   println("Output 2: " + output._1(2))
   println("Output 3: " + output._1(3))
   println("Output 4: " + output._1(4))
*/
   
  def program2 : FreeS[Single_reluFree.Op, (Tensor[Float])]  = Single_reluFree.instance.program

//   def output = program[Float].interpret[IO].unsafeRunSync
   val output2 = program2.interpret[IO].unsafeRunSync
   println("Output size: " + output2._1.size)
   println("Output 0: " + output2._1(0))
   println("Output 1: " + output2._1(1))
   println("Output 2: " + output2._1(2))
   println("Output 3: " + output2._1(3))
   println("Output 4: " + output2._1(4))


}
