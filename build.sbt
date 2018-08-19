import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

lazy val commonSettings = Seq(

  organization := "org.emergentorder.onnx",
  scalaOrganization := "org.scala-lang",
  scalaVersion := "2.12.6",
  crossScalaVersions := Seq("2.11.12","2.12.6", "2.13.0-M4"),
  version      := "1.2.2-0.1.0-SNAPSHOT",
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation", "-Ywarn-unused-import", "-Ywarn-unused:locals,privates"),
  libraryDependencies ++= Seq( 
    "org.typelevel" %%% "spire" % "0.16.0",
    "org.scalatest" %%% "scalatest" % "3.0.5-M1" % Test 
  ),
//  wartremoverErrors ++= Warts.allBut(Wart.DefaultArguments, Wart.Nothing, Wart.ToString),
//  wartremoverExcluded += baseDirectory.value / "core" / "src" / "main" / "scala" / "Float16.scala"
)


  

lazy val core = (crossProject(JSPlatform, JVMPlatform).crossType(CrossType.Pure) in file("core"))
.settings( commonSettings,
  name := "onnx-scala"
)

lazy val freestyle = (crossProject(JSPlatform, JVMPlatform).crossType(CrossType.Pure) in file("freestyle")).dependsOn(core)
.settings(
  commonSettings,
  publishArtifact in (Compile, packageDoc) := false,
  name := "onnx-scala-freestyle",
  addCompilerPlugin("org.scalameta" % "paradise" % "3.0.0-M11" cross CrossVersion.full),
  libraryDependencies ++= Seq(
      "io.frees" %%% "frees-core" % "0.8.2"
       )
)

