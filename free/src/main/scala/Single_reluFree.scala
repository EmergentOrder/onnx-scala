package org.emergentorder.onnxFree

import freestyle.free._
import freestyle.free.implicits._
import cats.free.{ Free, FreeApplicative } 
import cats.implicits._ 
import cats.effect.IO
import org.emergentorder.onnx._
import org.emergentorder.onnx.UnionType._
import scala.reflect.ClassTag
import spire.implicits._
import spire.math.UByte
import spire.math.UShort
import spire.math.UInt
import spire.math.ULong
import spire.math.Complex
import spire.algebra.Field
import spire.math.Numeric
import scala.language.higherKinds

@module trait Single_reluFree {
  val reluFree: ReluFree
  val dataSource: DataSourceFree
  val dropoutFree: DropoutFree
  def program[T :Numeric:ClassTag]
//  (implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double TypeOr Byte TypeOr Short TypeOr Int TypeOr Long TypeOr UByte TypeOr UShort TypeOr UInt TypeOr ULong TypeOr Complex[Float] TypeOr Complex[Double])#check[T])
//  (implicit ev:(UNil TypeOr Float16 TypeOr Float TypeOr Double)#check[T])
  : FS.Seq[Tensor[T]]  = 
    for {
      nodex <- dataSource.inputDataFree[T]
      nodey <- reluFree.Relu6Free[T]("y" ,X = Some(nodex))
      nodez <- dropoutFree.Dropout7Free[T]("z" , None, data = Some(nodey))
    } yield (nodez._1)
}
