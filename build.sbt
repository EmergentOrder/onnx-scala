import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

val dottyVersion = "3.0.0-RC1"
val scala213Version = "2.13.5"
val spireVersion = "0.17.0"
val scalaTestVersion = "3.2.6"

scalaVersion := dottyVersion

lazy val commonSettings = Seq(
//  scalaJSUseMainModuleInitializer := true, //Test only
  organization := "org.emergentorder.onnx",
  version := "0.11.0",
  scalaVersion := dottyVersion, 
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  updateOptions := updateOptions.value.withLatestSnapshots(false),
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation"),
  autoCompilerPlugins := true,
//  sources in (Compile, doc) := Seq(), //Bug w/ Dotty & JS on doc 
) ++ sonatypeSettings

lazy val common = (crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Pure) in file("common"))
  .settings(commonSettings, name := "onnx-scala-common",
    crossScalaVersions := Seq(
      dottyVersion,
      scala213Version
    ),
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 13)) => "TensorShapeDenotation.scala" | "TensorShapeDenotationOf.scala" | "Shape.scala" | "ShapeOf.scala" | "Indices.scala" | "IndicesOf.scala" | "dependent.scala"
      case _ => "" 
      }
    ),
)

lazy val proto = (crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Pure) in file("proto"))
  .settings(commonSettings, name := "onnx-scala-proto",
    crossScalaVersions := Seq(
      dottyVersion,
      scala213Version
    ),
    libraryDependencies -= ("com.thesamet.scalapb" %%% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion),
    libraryDependencies += ("com.thesamet.scalapb" %%% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion).withDottyCompat(scalaVersion.value),
  PB.targets in Compile := Seq(
      scalapb.gen() -> (sourceManaged in Compile).value
    ),
    // The trick is in this line:
    PB.protoSources in Compile := Seq(file("proto/src/main/protobuf")),
  )

lazy val backends = (crossProject(JVMPlatform, JSPlatform)
  .crossType(CrossType.Pure) in file("backends"))
  .dependsOn(core)
//conditionally enabling/disable based on version, still not working
//  .enablePlugins(ScalaJSBundlerPlugin)//, ScalablyTypedConverterPlugin)
  .settings(
    commonSettings,
    name := "onnx-scala-backends",
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 13)) => "NCF.scala" | 
                            "ORTOperatorBackend.scala" | 
                            "ORTOperatorBackendAll.scala" | "ORTOperatorBackendAtoL.scala" |
                            "ORTModelBackend.scala" | 
                            "Main.scala" | "ONNXJSOperatorBackend.scala"  
      case _ => "ORTModelBackend213.scala" | "NCF213.scala" |
                "ORTOperatorBackend213.scala" | "ORTOperatorBackendAll213.scala" | 
                "ORTOperatorBackendAtoL213.scala" |
                            "Main.scala" | "ONNXJSOperatorBackend.scala" 
      }
    ),
    scalacOptions ++= { if (isDotty.value) Seq("-source:3.0-migration") else Nil },
    libraryDependencies ++= Seq(
        "com.microsoft.onnxruntime" % "onnxruntime" % "1.7.0"
    ),
    crossScalaVersions := Seq(dottyVersion, scala213Version)
  )
.jvmSettings(
libraryDependencies += ("org.scalatest" %% "scalatest" % scalaTestVersion) % Test,
).jsSettings(
      scalaJSUseMainModuleInitializer := true) //, //Testing
//Seems to be a bundling issue, copying things manually seems to work
//     npmDependencies in Compile += "onnxjs" -> "0.1.8")

lazy val core = (crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Pure) in file("core"))
  .dependsOn(common)
  .dependsOn(proto)
  .settings(
    commonSettings,
    name := "onnx-scala",
    scalacOptions ++= { if (isDotty.value) Seq("-source:3.0-migration") else Nil },
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 13)) => "ONNX.scala" | "OpToONNXBytesConverter.scala" | "Tensors.scala" | "ONNXBytesDataSource.scala"
      case _ => "ONNX213.scala" | "OpToONNXBytesConverter213.scala" | "Tensors213.scala" | "ONNXBytesDataSource213.scala"
      }
    ),
    crossScalaVersions := Seq(
      dottyVersion,
      scala213Version
    ),
    libraryDependencies ++= (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, n)) =>
        Seq(
          "org.typelevel" %%% "spire" % spireVersion,
        )
      case _ =>
        Seq(
          ("org.typelevel" %%% "spire" % spireVersion).withDottyCompat(dottyVersion),
        )
    })
)

/*
lazy val docs = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure) in file("core-docs"))       // new documentation project
  .settings(
    commonSettings,
    mdocVariables := Map(
      "VERSION" -> version.value
   )
  )
  .dependsOn(backends)
  .enablePlugins(MdocPlugin)
  .jvmSettings(
    crossScalaVersions := Seq(scala213Version)
  )
*/

skip in publish := true
sonatypeProfileName := "com.github.EmergentOrder" 

lazy val sonatypeSettings = Seq(
organization := "com.github.EmergentOrder",
homepage := Some(url("https://github.com/EmergentOrder/onnx-scala")),
scmInfo := Some(ScmInfo(url("https://github.com/EmergentOrder/onnx-scala"),
                            "git@github.com:EmergentOrder/onnx-scala.git")),
developers := List(Developer("EmergentOrder",
                             "Alexander Merritt",
                             "lecaran@gmail.com",
                             url("https://github.com/EmergentOrder"))),
licenses += ("AGPL-3.0", url("https://www.gnu.org/licenses/agpl-3.0.html")),
publishMavenStyle := true,
publishConfiguration := publishConfiguration.value.withOverwrite(true),
publishLocalConfiguration := publishLocalConfiguration.value.withOverwrite(true),
publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)
)
