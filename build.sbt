import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
val dottyVersion = "0.20.0-RC1"
val scala211Version = "2.11.12"
val scala212Version = "2.12.10"
val scala213Version = "2.13.1"
val spireVersion = "0.17.0-M1"
val zioVersion = "1.0.0-RC16"
val scalametaVersion = "4.2.5"
val onnxJavaCPPPresetVersion = "1.6.0-1.5.2"

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
    publishArtifact in (Compile, packageDoc) := false
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
    publishArtifact in (Compile, packageDoc) := false
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
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2") else Nil },
    libraryDependencies ++= Seq(
      "org.bytedeco" % "ngraph-platform" % "0.26.0-1.5.2",
//      "com.microsoft.onnxruntime" % "onnxruntime4j" % "1.0.0-SNAPSHOT"
    ),
//    sources in (Compile, doc) := Seq(),
    publishArtifact in (Compile, packageDoc) := false
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
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2") else Nil },
//    scalaVersion := scala213Version,
    excludeFilter in unmanagedSources := (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, 13)) => ("ONNX.scala")
      case _ => "ONNX213.scala"
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
    publishArtifact in (Compile, packageDoc) := false, //TODO: Only block this for JS
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
