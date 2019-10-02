import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
val dottyVersion = "0.17.0"
val scala211Version = "2.11.12"
val scala212Version = "2.12.10"
val scala213Version = "2.13.1"
val spireVersion = "0.17.0-M1"
val zioVersion = "1.0.0-RC13"
val scalametaVersion = "4.2.3"
val onnxJavaCPPPresetVersion = "1.6.0-1.5.2-SNAPSHOT"
scalaVersion := scala212Version

lazy val commonSettings = Seq(
  scalaJSUseMainModuleInitializer := true, //Test only
  organization := "org.emergentorder.onnx",
  version := "1.6.0-0.1.0-SNAPSHOT",
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  updateOptions := updateOptions.value.withLatestSnapshots(false),
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation", "-Ywarn-unused", "-Yrangepos"),
  autoCompilerPlugins := true
)

lazy val common = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
  .crossType(CrossType.Pure) in file("common"))
  .settings(commonSettings, name := "onnx-scala-common",
//    excludeFilter in unmanagedSources := (CrossVersion
//      .partialVersion(scalaVersion.value) match {
//      case Some((0, n)) => "UnionType.scala"
//      case _ => ""
//      }
//    )
  )
  .jvmSettings(
//    scalaVersion := scala213Version,
    crossScalaVersions := Seq(
      dottyVersion,
      scala212Version,
      scala213Version,
      scala211Version
    ),
    publishArtifact in (Compile, packageDoc) := false
  )
  .jsSettings(
//    scalaVersion := scala213Version,
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
  )
  .nativeSettings(
    scalaVersion := scala211Version
  )

lazy val commonJS = common.js
  .disablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .disablePlugins(dotty.tools.sbtplugin.DottyIDEPlugin)

lazy val programGenerator = (crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Pure) in file("programGenerator"))
  .dependsOn(backends)
  .settings(
    commonSettings,
    name := "onnx-scala-program-generator",
//    scalaVersion := scala213Version,
    mainClass in (Compile, run) := Some(
      "org.emergentorder.onnx.ONNXProgramGenerator"
    ),
    libraryDependencies ++= Seq(
      ("org.scalameta" %% "scalameta" % scalametaVersion).withDottyCompat(dottyVersion)
    ),
    publishArtifact in (Compile, packageDoc) := false
  )
  .jvmSettings(
    crossScalaVersions := Seq(
//      dottyVersion,
      scala212Version,
      scala211Version,
      scala213Version
    )
  )
  .jsSettings(
    crossScalaVersions := Seq(
      scala212Version,
      scala211Version,
      scala213Version
    )
  )
lazy val backends = (crossProject(JVMPlatform, JSPlatform)
  .crossType(CrossType.Pure) in file("backends"))
  .dependsOn(core)
  .settings(
    commonSettings,
    name := "onnx-scala-backends",
//    scalaVersion := scala213Version,
    libraryDependencies ++= Seq(
      "org.bytedeco" % "ngraph-platform" % "0.25.0-1.5.2-SNAPSHOT"
    ),
    publishArtifact in (Compile, packageDoc) := false
  )
  .jvmSettings(
    crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version)
  )
  .jsSettings(
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
  )
//  .nativeSettings(
//    scalaVersion := scala211Version
//  )

lazy val core = (crossProject(JSPlatform, JVMPlatform)
  .crossType(CrossType.Pure) in file("core"))
  .dependsOn(common)
  .settings(
    commonSettings,
    name := "onnx-scala",
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
    publishArtifact in (Compile, packageDoc) := false,
    libraryDependencies ++= (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, n)) if (n == 13 || n == 11) =>
        Seq(
          "org.typelevel" %% "spire" % spireVersion
        )
      case _ =>
        Seq(
          ("org.typelevel" %% "spire" % spireVersion)
            .withDottyCompat(dottyVersion)
        )
    }),
    libraryDependencies ++= Seq(
      ("org.bytedeco" % "onnx-platform" % onnxJavaCPPPresetVersion)
        .withDottyCompat(dottyVersion)
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
      case Some((2, n)) if (n == 13 || n == 11) =>
        Seq(
          "org.typelevel" %%% "spire" % spireVersion
        )
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

lazy val zio = (crossProject(JVMPlatform, JSPlatform)
  .crossType(CrossType.Pure) in file("zio"))
  .dependsOn(backends)
  .disablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .settings(
    commonSettings,
    name := "onnx-scala-zio",
//    scalaVersion := scala213Version,
    publishArtifact in (Compile, packageDoc) := false,
    libraryDependencies ++= (CrossVersion
      .partialVersion(scalaVersion.value) match {
      case Some((2, n)) if n == 13 =>
        Seq(
          //"org.typelevel" %% "cats-effect" % "2.0.0-M4"
          "dev.zio" %% "zio" % zioVersion
        )
      case _ =>
        Seq(
          "dev.zio" %% "zio" % zioVersion
        )
    })
  )
  .jvmSettings(
    crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version)
  )
  .jsSettings(
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
  )
