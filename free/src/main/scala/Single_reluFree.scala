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
  def program
  : FS.Seq[Tensor[Float]]  = 
    for {
      nodex <- dataSource.inputDataFree[Float]
      nodey <- reluFree.Relu6Free("y" ,X = Some(nodex))
      nodez <- dropoutFree.Dropout7Free("z" , None, data = Some(nodey))
    } yield (nodez._1)
}
