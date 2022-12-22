package org.emergentorder.onnx.backends

import sys.process.*
import java.net.URL
import java.io.File
import java.nio.file.{Files, Paths}
import scala.language.postfixOps
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.Tensors.Tensor.*
import org.emergentorder.onnx.backends.*
import org.emergentorder.compiletime.*
import org.emergentorder.io.kjaer.compiletime.*

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.*
import cats.effect.testing.scalatest.AsyncIOSpec


class ONNXScalaBertTokenizerSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers {
/*
   new URL(
     "https://github.com/microsoft/onnxruntime-extensions/raw/main/test/data/test_bert_tokenizer.onnx"
   ) #> new File("test_bert_tokenizer.onnx") !!

   "BERT Tokenizer ONNX-Scala model should tokenize text" in {
      val bytes = Files.readAllBytes(Paths.get("test_bert_tokenizer.onnx"))
      val bertTokenizer      = new ORTModelBackend(bytes)
      val data            = Array.fill(1) { "This is a test" }
      // In NCHW tensor image format
      val shape                 = 1 #: SNil
      val tensorShapeDenotation = "Batch" ##: TSNil

      val tensorDenotation: String & Singleton = "Text"

      val textTens = Tensor(data, tensorDenotation, tensorShapeDenotation, shape)

      // or as a shorthand if you aren't concerned with enforcing denotations
      val textensDefaultDenotations = Tensor(data, shape)
      val out = bertTokenizer.fullModel[
        Long,
        "ImageNetClassification",
        "Batch" ##: "Class" ##: TSNil,
        6 #: SNil
      ](Tuple(textTens))

      // The output shape
      // and token values
      val singleIO = cats.effect.IO.both(out.shape, out.data)
      singleIO.asserting(x => ((x._1(0), 
                                x._2(0),
                                x._2(1),
                                x._2(2),
                                x._2(3),
                                x._2(4),
                                x._2(5))
                              shouldBe
                               (6,
                                101l,
                                1188l,
                                1110l,
                                170l,
                                2774l,
                                102l)))

      //Expected bert tokenizer output ("input_ids") : 101, 1188, 1110, 170, 2774, 102 , longs 
   }
*/
}

