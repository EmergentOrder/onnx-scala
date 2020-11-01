import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

val dottyVersion = "0.27.0-RC1" //"3.0.0-M1"
val scala213Version = "2.13.3" // "2.13.4"
val spireVersion = "0.17.0"
val scalametaVersion = "4.3.24"
val onnxJavaCPPPresetVersion = "1.7.0-1.5.5-SNAPSHOT"
  
scalaVersion := scala213Version 

//PB.targets in Compile := Seq(
//  scalapb.gen() -> (sourceManaged in Compile).value 
//)

lazy val commonSettings = Seq(
//  scalaJSUseMainModuleInitializer := true, //Test only
  organization := "org.emergentorder.onnx",
  version := "0.8.0",
  scalaVersion := scala213Version,
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  updateOptions := updateOptions.value.withLatestSnapshots(false),
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation", "-language:strictEquality"),
  autoCompilerPlugins := true,
) ++ sonatypeSettings

lazy val onnx = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure) in file("onnx"))
  .settings(commonSettings, name := "onnx-scala-onnx",
    crossScalaVersions := Seq(
      dottyVersion,
      scala213Version
    )
  )

lazy val common = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure) in file("common"))
  .settings(commonSettings, name := "onnx-scala-common",
    crossScalaVersions := Seq(
      dottyVersion,
      scala213Version
    )
  )


lazy val programGenerator = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure) in file("programGenerator"))
  .dependsOn(backends)
  .settings(
    commonSettings,
    name := "onnx-scala-program-generator",
    mainClass in (Compile, run) := Some(
      "org.emergentorder.onnx.ONNXProgramGenerator"
    ),
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 13)) => "Absnet.scala" | "Squeezenet1dot1.scala" | "ONNXProgramGenerator.scala"
      case _ => "ONNXProgramGenerator213.scala" | "Squeezenet1dot1213.scala"
      }
    ),
    scalacOptions ++= { if (isDotty.value) Seq("-source 3.0-migration") else Nil },
    libraryDependencies ++= (CrossVersion
    .partialVersion(scalaVersion.value) match {
     case Some((2,_)) =>
        Seq(
          "org.scalameta" %% "scalameta" % scalametaVersion
        )
     case _ =>
        Seq(
         ("org.scalameta" %% "scalameta" % scalametaVersion).withDottyCompat(dottyVersion)
        )
      }
    ),
    crossScalaVersions := Seq(
      dottyVersion,
      scala213Version
    )
  )

lazy val backends = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure) in file("backends"))
  .dependsOn(core)
  .settings(
    commonSettings,
    name := "onnx-scala-backends",
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 13)) => "NCF.scala" | 
                            "ORTOperatorBackend.scala" | 
                            "ORTOperatorBackendAll.scala" | "ORTOperatorBackendAtoL.scala" |
                            "ORTModelBackend.scala"
      case _ => "ORTModelBackend213.scala" | "NCF213.scala" |
                "ORTOperatorBackend213.scala" | "ORTOperatorBackendAll213.scala" | 
                "ORTOperatorBackendAtoL213.scala"
      }
    ),
    scalacOptions ++= { if (isDotty.value) Seq("-source 3.0-migration") else Nil },
    libraryDependencies ++= Seq(
 //        "org.bytedeco" % "dnnl-platform" % "1.6.4-1.5.5-SNAPSHOT",
        "com.microsoft.onnxruntime" % "onnxruntime" % "1.5.2"
//      "org.bytedeco" % "onnxruntime-platform" % "1.5.2-1.5.5-SNAPSHOT"
    ),
    crossScalaVersions := Seq(dottyVersion, scala213Version)
  )

lazy val core = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure) in file("core"))
  .dependsOn(common)
  .dependsOn(onnx)
  .settings(
    commonSettings,
    name := "onnx-scala",
    scalacOptions ++= { if (isDotty.value) Seq("-source 3.0-migration") else Nil },
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
          "org.typelevel" %% "spire" % spireVersion,
        )
      case _ =>
        Seq(
          ("org.typelevel" %% "spire" % spireVersion).withDottyCompat(dottyVersion),
        )
    }),
    libraryDependencies ++= Seq(
        "com.microsoft.onnxruntime" % "onnxruntime" % "1.5.2",
        "org.bytedeco" % "onnx-platform" % onnxJavaCPPPresetVersion,
//      "org.osgi" % "org.osgi.annotation.versioning" % "1.1.0"
    )
  )

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
