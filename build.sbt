import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
val dottyVersion = "0.23.0-RC1"
val scala211Version = "2.11.12"
val scala212Version = "2.12.11"
val scala213Version = "2.13.1"
val spireVersion = "0.17.0-M1"
val zioVersion = "1.0.0-RC18-2"
val scalametaVersion = "4.3.8"
val onnxJavaCPPPresetVersion = "1.6.0-1.5.3"

lazy val commonSettings = Seq(
  scalaJSUseMainModuleInitializer := true, //Test only
  organization := "org.emergentorder.onnx",
  version := "0.2.0",
  scalaVersion := scala213Version,
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  updateOptions := updateOptions.value.withLatestSnapshots(false),
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation", "-Yrangepos"),
  autoCompilerPlugins := true,
) ++ sonatypeSettings

lazy val common = (crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Pure) in file("common"))
  .settings(commonSettings, name := "onnx-scala-common",
  )
  .jvmSettings(
    crossScalaVersions := Seq(
      dottyVersion,
      scala212Version,
      scala213Version,
      scala211Version
    ),
 //   sources in (Compile, doc) := Seq(),
//    publishArtifact in (Compile, packageDoc) := false
  )
  .jsSettings(
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
  )
//  .nativeSettings(
//    scalaVersion := scala211Version
//  )

lazy val commonJS = common.js
  .disablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .disablePlugins(dotty.tools.sbtplugin.DottyIDEPlugin)

lazy val programGenerator = (crossProject(JVMPlatform)//,JSPlatform)
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
      case Some((2, 11)) => "Absnet.scala" | "Squeezenet1dot1.scala" | "Squeezenet1dot1213.scala" | "ONNXProgramGenerator213.scala" | "ONNXProgramGenerator.scala"
      case Some((2, 12)) => "Absnet.scala" | "Squeezenet1dot1.scala" | "Squeezenet1dot1213.scala" | "ONNXProgramGenerator213.scala" | "ONNXProgramGenerator.scala"
      case Some((2, 13)) => "Absnet.scala" | "Squeezenet1dot1.scala" | "ONNXProgramGenerator.scala"
      case _ => "ONNXProgramGenerator213.scala" | "Squeezenet1dot1213.scala"
      }
    )
  )
  .jvmSettings(
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2Compat") else Nil }, 
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
//    sources in (Compile, doc) := Seq(),
//    publishArtifact in (Compile, packageDoc) := false
  )
  .jvmSettings(
    crossScalaVersions := Seq(
      dottyVersion,
      scala212Version,
      scala211Version,
      scala213Version
    )
  )
//  .jsSettings(
//    crossScalaVersions := Seq(
//      scala212Version,
//      scala211Version,
//      scala213Version
//    )
//  )
lazy val backends = (crossProject(JVMPlatform) //JSPlatform)
  .crossType(CrossType.Pure) in file("backends"))
  .dependsOn(core)
  .settings(
    commonSettings,
    name := "onnx-scala-backends",
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 11)) => "NGraphBackendUtils.scala" | "NGraphModelBackend.scala" | 
                            "NGraphOperatorBackend.scala" | "NCF.scala" | 
                            "NGraphOperatorBackendFull213.scala" | "NGraphOperatorBackendAtoL213.scala" |
                            "NGraphOperatorBackendAll.scala" | "ORTOperatorBackendAll.scala" | 
                            "ORTOperatorBackend.scala" | "ORTOperatorBackend213.scala" | 
                            "ORTOperatorBackendAll213.scala" | "ORTModelBackend.scala" | 
                            "ORTModelBackend213.scala" | "ORTOperatorBackendAtoL213.scala"
      case Some((2, 12)) => "NGraphBackendUtils.scala" | "NGraphModelBackend.scala" | 
                            "NGraphOperatorBackend.scala" | "NCF.scala" | 
                            "NGraphOperatorBackendFull213.scala" | "NGraphOperatorBackendAtoL213.scala" |
                            "NGraphOperatorBackendAll.scala" | "ORTOperatorBackendAll.scala" | 
                            "ORTOperatorBackend.scala" | "ORTOperatorBackend213.scala" | 
                            "ORTOperatorBackendAll213.scala" | "ORTModelBackend.scala" | 
                            "ORTModelBackend213.scala" | "ORTOperatorBackendAtoL213.scala"
      case Some((2, 13)) => "NGraphBackendUtils.scala" | "NGraphModelBackend.scala" | 
                            "NGraphOperatorBackend.scala" | "NCF.scala" | 
                            "NGraphOperatorBackendAll.scala" | "ORTOperatorBackend.scala" | 
                            "ORTOperatorBackendAll.scala" | "ORTModelBackend.scala" | 
                            "ORTModelBackend212.scala"
      case _ => "NGraphBackendUtils212.scala" | "NGraphModelBackend212.scala" | 
                "NGraphOperatorBackend212.scala" | "NCF212.scala" | 
                "NGraphOperatorBackendFull213.scala" | "NGraphOperatorBackendAtoL213.scala" |
                "ORTModelBackend212.scala" | "ORTModelBackend213.scala" |
                "ORTOperatorBackend213.scala" | "ORTOperatorBackendAll213.scala" | 
                "ORTOperatorBackendAtoL213.scala"
      }
    ),
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2Compat") else Nil },
    libraryDependencies ++= Seq(
      "org.bytedeco" % "ngraph-platform" % "0.26.0-1.5.3-SNAPSHOT",
      "org.bytedeco" % "onnxruntime-platform" % "1.2.0-1.5.3",
//      "com.microsoft.onnxruntime" % "onnxruntime4j" % "1.0.0-SNAPSHOT"
    ),
//    sources in (Compile, doc) := Seq(),
//    publishArtifact in (Compile, packageDoc) := false
  )
  .jvmSettings(
    crossScalaVersions := Seq(dottyVersion, scala212Version, scala213Version, scala211Version)
  )
//  .jsSettings(
//    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
//  )
//  .nativeSettings(
//    scalaVersion := scala211Version
//  )

lazy val core = (crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Pure) in file("core"))
  .dependsOn(common)
  .settings(
    commonSettings,
    name := "onnx-scala",
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2Compat") else Nil },
//    scalaVersion := scala213Version,
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 11)) => "ONNX.scala" | "ONNX213.scala" | "OpToONNXBytesConverter.scala" | "Tensor.scala"
      case Some((2, 12)) => "ONNX.scala" | "ONNX213.scala" | "OpToONNXBytesConverter.scala" | "Tensor.scala"
      case Some((2, 13)) => "ONNX.scala" | "ONNX212.scala" | "OpToONNXBytesConverter.scala" | "Tensor.scala"
      case _ => "ONNX212.scala" | "ONNX213.scala" | "OpToONNXBytesConverter212.scala"
      }
    )
  )
  .jvmSettings(
    crossScalaVersions := Seq(
      dottyVersion,
      scala212Version,
      scala213Version,
      scala211Version
    ),
//    sources in (Compile, doc) := Seq(),
//    publishArtifact in (Compile, packageDoc) := false, //TODO: Only block this for JS
    libraryDependencies ++= (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, n)) =>
        Seq(
          "org.typelevel" %% "spire" % spireVersion
        )
      case _ =>
        Seq(
          ("org.typelevel" %% "spire" % spireVersion).withDottyCompat(dottyVersion)
        )
    }),
    libraryDependencies ++= Seq(
      "org.bytedeco" % "onnx-platform" % onnxJavaCPPPresetVersion,
      "org.osgi" % "org.osgi.annotation.versioning" % "1.1.0"
    )
  )
  .jsSettings(
    crossScalaVersions := Seq(
      scala212Version,
      scala211Version,
      scala213Version
    ),
    libraryDependencies ++= Seq(
      "org.bytedeco" % "onnx-platform" % onnxJavaCPPPresetVersion
    ),
    libraryDependencies ++= (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case _ =>
        Seq(
          "org.typelevel" %%% "spire" % spireVersion
        )
    })
  )
/*
    .nativeSettings(
      scalaVersion := scala211Version,
      libraryDependencies ++= Seq(
        "org.typelevel" %% "spire" % spireVersion,
        "org.bytedeco" % "onnx-platform" % onnxJavaCPPPresetVersion
      )
    )
*/

lazy val docs = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure) in file("core-docs"))       // new documentation project
  .settings(
    commonSettings,
    mdocVariables := Map(
      "VERSION" -> version.value
   )
  )
  .dependsOn(programGenerator)
  .enablePlugins(MdocPlugin)
  .jvmSettings(
    crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version)
  )

lazy val zio = (crossProject(JVMPlatform)//, JSPlatform)
  .crossType(CrossType.Pure) in file("zio"))
  .dependsOn(backends)
  .settings(
    commonSettings,
    name := "onnx-scala-zio",
//    scalaVersion := scala213Version,
//    sources in (Compile, doc) := Seq(),
//    publishArtifact in (Compile, packageDoc) := false,
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 11)) => "NCFZIO.scala" | "ZIONGraphBackend.scala" 
      case Some((2, 12)) => "NCFZIO.scala" | "ZIONGraphBackend.scala"
      case Some((2, 13)) => "NCFZIO.scala" | "ZIONGraphBackend.scala"
      case _ => ""
      }
    ),
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2Compat") else Nil },
    libraryDependencies ++= (CrossVersion
    .partialVersion(scalaVersion.value) match {
     case Some((2,_)) =>
        Seq(
          "dev.zio" %% "zio" % zioVersion
        )
     case _ =>
        Seq(
         ("dev.zio" %% "zio" % zioVersion).withDottyCompat(dottyVersion)
        )
     })
  )
  .jvmSettings(
    //crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version)
    crossScalaVersions := Seq(dottyVersion, scala212Version, scala213Version, scala211Version)
  )
//  .jsSettings(
//    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
//  )

skip in publish := true
sonatypeProfileName := "com.github.EmergentOrder" 
//sonatypeSessionName := s"[sbt-sonatype] ${name.value} ${version.value}"
//sources in (Compile, packageDoc) := Seq()

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
sonatypeProfileName := "lecaran",
publishMavenStyle := true,
publishConfiguration := publishConfiguration.value.withOverwrite(true),
publishLocalConfiguration := publishLocalConfiguration.value.withOverwrite(true),
publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)
//publishTo := sonatypePublishToBundle.value
)
