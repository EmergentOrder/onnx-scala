import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
val dottyVersion = "0.9.0-RC1"
val scala212Version = "2.12.6"
val scala213Version = "2.13.0-M4"

lazy val commonSettings = Seq(

  organization := "org.emergentorder.onnx",
//  crossScalaVersions := Seq(dottyVersion, "2.10.7", "2.11.12",scala212Version, "2.13.0-M5"),
  version      := "1.2.2-0.1.0-SNAPSHOT",
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation"),
//  wartremoverErrors ++= Warts.allBut(Wart.DefaultArguments, Wart.Nothing, Wart.ToString),
//  wartremoverExcluded += baseDirectory.value / "core" / "src" / "main" / "scala" / "Float16.scala"
)

lazy val common = (crossProject(JSPlatform, JVMPlatform)
    .crossType(CrossType.Pure) in file("common"))
  .settings( commonSettings,
    name := "onnx-scala-common"
  )
  .jvmSettings(
    crossScalaVersions := Seq(dottyVersion, scala212Version, scala213Version)
  )

lazy val commonJS     = common.js.disablePlugins(dotty.tools.sbtplugin.DottyPlugin).disablePlugins(dotty.tools.sbtplugin.DottyIDEPlugin)

lazy val core = (crossProject(JSPlatform, JVMPlatform)
    .crossType(CrossType.Pure) in file("core")).dependsOn(common)
  .settings(commonSettings,
    name := "onnx-scala",
    scalaVersion := scala212Version,
    libraryDependencies ++= Seq("eu.timepit" %%% "singleton-ops" % "0.3.0")
    )
    .jvmSettings(
      crossScalaVersions := Seq(scala212Version, scala213Version),
      libraryDependencies ++= Seq("org.typelevel" % "spire_2.12" % "0.16.0")
    )
    .jsSettings(
      libraryDependencies ++= Seq("org.typelevel" %%% "spire" % "0.16.0")
    )

lazy val coreDotty = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure)).in(file("coreDotty")).dependsOn(common)
  .enablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .settings( commonSettings,
    name := "onnx-scala-dotty",
    scalaVersion := dottyVersion,
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2") else Nil },
    libraryDependencies ++= Seq(
      ("org.typelevel" %% "spire" % "0.16.0").withDottyCompat(dottyVersion),
      ("eu.timepit" %% "singleton-ops" % "0.3.0").withDottyCompat(dottyVersion)
    )
)

lazy val freestyle = (crossProject(JSPlatform, JVMPlatform)
    .crossType(CrossType.Pure) in file("freestyle")).dependsOn(core)
  .disablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .settings( commonSettings,
    name := "onnx-scala-freestyle", 
    scalaVersion := scala212Version,
    publishArtifact in (Compile, packageDoc) := false,
    addCompilerPlugin("org.scalameta" % "paradise" % "3.0.0-M11" cross CrossVersion.full),
    libraryDependencies ++= Seq(
      "io.frees" %% "frees-core" % "0.8.2"
  )
)
