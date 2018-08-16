lazy val root = (project in file(".")).
settings(
  inThisBuild(List(
    organization := "org.emergentorder",
    scalaOrganization := "org.scala-lang",
    scalaVersion := "2.12.6",
    crossScalaVersions := Seq("2.11.12","2.12.6", "2.13.0-M4"),
    version      := "1.2.2-0.1.0-SNAPSHOT"
  )),
  name := "onnx-scala",
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation", "-Ywarn-unused-import", "-Ywarn-unused:locals,privates"),
    libraryDependencies ++= Seq( 
      "org.typelevel" %% "spire" % "0.16.0",
      "org.scalatest" %% "scalatest" % "3.0.5-M1" % Test,
       ),
    wartremoverErrors ++= Warts.allBut(Wart.DefaultArguments),
    wartremoverExcluded += baseDirectory.value / "src" / "main" / "scala" / "Float16.scala"
)
