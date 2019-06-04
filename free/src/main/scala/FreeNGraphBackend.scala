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

import scalaz.zio.Task
import scalaz.zio.DefaultRuntime

import org.emergentorder.onnx._
import org.emergentorder.onnxFree._
import org.emergentorder.onnx.UnionType._

import org.emergentorder.onnx.backends._

object ONNXNGraphHandlers extends App {

   val fileName = "NCF.onnx" //Needed for params at runtime //TODO: inject
   val onnxHelper = new ONNXHelper(fileName)
   val ngraphBackend = new NGraphBackend(onnxHelper)

  

  //TODO: Fix concurrency problem caused by protobuf sharedDtor

  class DatasourceHandler extends DataSourceFree {
    override def inputDataFree[T: Numeric: ClassTag]
    (implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
    : Task[Tensor[T]] = {
      Task{
        //TODO
        (Array(0f, 220f, -2200f, 5000f, 10000f), Array(5)).asInstanceOf[Tensor[T]]
      }
    }

    def getParamsFree[ T : Numeric : ClassTag](name: String)
    (implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
    : Task[Tensor[T]] = {
       Task{ 
           ngraphBackend.getParamsFree(name)
      } 

    }

     def getAttributesFree[T : Numeric:ClassTag](name: String)(implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T]): Task[Tensor[T]] = ???
  }
  
  class ReluHandler extends ReluFree {

  override def Relu1Free[@sp T: Numeric: ClassTag](name: String,
                                        consumed_inputs: Option[(Array[Int])],
                                        X: Option[Tensor[T]])(
        implicit evT: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : Task[(Tensor[T])] = Task{
        ngraphBackend.Relu1(name, consumed_inputs, X)
      }
  

  override def Relu6Free[@sp T : Numeric: ClassTag](name: String, X: Option[org.emergentorder.onnx.Tensor[T]])
  //(
        (implicit ev: (UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : Task[(Tensor[T])] = {
        Task{
          ngraphBackend.Relu6[T](name, X)
        }
      }
}




  class DropoutHandler extends DropoutFree {

     def Dropout1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T])] = ???


  def Dropout6Free[@sp T : Numeric:ClassTag](name: String,is_test : Option[(Int)] = None,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T], Tensor[T])] = ???


  def Dropout10Free[@sp T : Numeric:ClassTag,@sp T1 : Numeric:ClassTag](name: String,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T],evT1:(UNil TypeOr Boolean)#check[T1])    : Task[(Tensor[T], Tensor[T1])] = ???


      override def Dropout7Free[@sp T : Numeric:ClassTag](name: String,ratio : Option[(Float)] = None,data: Option[Tensor[T]])
      (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
      : Task[(Tensor[T], Tensor[T])] = 
        Task{
          ngraphBackend.Dropout7[T](name, None, data)
        }

  }

    class GatherHandler extends GatherFree {


     override def Gather1Free[@sp T : Numeric:ClassTag,@sp Tind : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,data: Option[Tensor[T]], indices: Option[Tensor[Tind]])
(implicit evT:(UNil TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T],evTind:(UNil TypeOr Int TypeOr Long)#check[Tind])
      : Task[(Tensor[T])] =
        Task{
          ngraphBackend.Gather1[T, Tind](name, axis, data, indices)
        }

  }


  class MulHandler extends MulFree {
     def Mul1Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,consumed_inputs : Option[(Array[Int])] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])] = ???


  def Mul6Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)] = None,broadcast : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])] = ???

    override def Mul7Free[@sp T : Numeric:ClassTag](name: String,A: Option[Tensor[T]], B: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
    : Task[(Tensor[T])] =
      Task{
        ngraphBackend.Mul7[T](name, A, B)
      }
  }

  class GemmHandler extends GemmFree {

      def Gemm1Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])] = ???


  def Gemm6Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,broadcast : Option[(Int)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])] = ???


  def Gemm7Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])    : Task[(Tensor[T])] = ???


     override def Gemm9Free[@sp T : Numeric:ClassTag](name: String,alpha : Option[(Float)] = None,beta : Option[(Float)] = None,transA : Option[(Int)] = None,transB : Option[(Int)] = None,A: Option[Tensor[T]], B: Option[Tensor[T]], C: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Float16 TypeOr Float TypeOr Double TypeOr UInt TypeOr ULong TypeOr Int TypeOr Long)#check[T])
  : Task[(Tensor[T])] =
    Task{
        ngraphBackend.Gemm9[T](name, alpha, beta, transA, transB, A, B, C)
    }
  }


  class SigmoidHandler extends SigmoidFree {
      def Sigmoid1Free[@sp T : Numeric:ClassTag](name: String,consumed_inputs : Option[(Array[Int])] = None,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])    : Task[(Tensor[T])] = ???

    override   def Sigmoid6Free[@sp T : Numeric:ClassTag](name: String,X: Option[Tensor[T]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
  : Task[(Tensor[T])] =
    Task{
      ngraphBackend.Sigmoid6[T](name, X)
    }

  }

  class ConcatHandler extends ConcatFree {
     override def Concat4Free[@sp T : Numeric:ClassTag](name: String,axis : Option[(Int)],inputs: Seq[Option[Tensor[T]]])
(implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr Float16 TypeOr Float TypeOr Double TypeOr String TypeOr Boolean TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
  : Task[(Tensor[T])] = {

    Task{
    ngraphBackend.Concat4[T](name, axis, inputs)
    }
  }

 }


/*
//  def program[T: Numeric : ClassTag]
//  (implicit evT:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
//  : FreeS[Single_reluFree.Op,(Tensor[T])] = Single_reluFree.instance.program[T]

  
//   def output = program[Float].interpret[Task].unsafeRunSync
   val output = program[Float].interpret[Task].unsafeRunSync
   println("Output size: " + output._1.size)
   println("Output 0: " + output._1(0))
   println("Output 1: " + output._1(1))
   println("Output 2: " + output._1(2))
   println("Output 3: " + output._1(3))
   println("Output 4: " + output._1(4))
*/

  def program = Single_reluFree.program

  val runtime = new DefaultRuntime {}

  val output2 = runtime.unsafeRun(program)

//  def program2 : FreeS[NCFFree.Op, (Tensor[Float])]  = NCFFree.instance.program

//   def output = program[Float].interpret[Task].unsafeRunSync
//   val output2 = program2.interpret[Task].unsafeRunSync
   println("Output size: " + output2._1.size)
   println("Output 0: " + output2._1(0))
   println("Output 1: " + output2._1(1))
   println("Output 2: " + output2._1(2))
   println("Output 3: " + output2._1(3))
   println("Output 4: " + output2._1(4))


}
