package org.emergentorder.onnx.backends

object Main {
   def main(args: Array[String]): Unit = {
//    val lib = new MyLibrary
//    println(lib.sq(2))

      val t = new ORTOperatorBackend {}

      t.test()
      println(s"Using Scala.js version ${System.getProperty("java.vm.version")}")
   }
}
