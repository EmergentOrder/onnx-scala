import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
val dottyVersion = "0.10.0-RC1"
val scala211Version = "2.11.12"
val scala212Version = "2.12.7"
val scala213Version = "2.13.0-M5"
//Might want to remove cats ( conflict with Freestyle's version)
val catsVersion = "1.4.0"
//TODO: Replace wartremover with scalafix

lazy val commonSettings = Seq(
  organization := "org.emergentorder.onnx",
  version      := "1.3.0-0.1.0-SNAPSHOT",
  resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  updateOptions := updateOptions.value.withLatestSnapshots(false),
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation"),
  autoCompilerPlugins := true
)

lazy val common = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
    .crossType(CrossType.Pure) in file("common"))
  .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala-common"
  )
  .jvmSettings(
    scalaVersion := scala212Version,
    crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version),
    publishArtifact in (Compile, packageDoc) := false
  )
  .jsSettings(
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
  )
  .nativeSettings(
    scalaVersion := scala211Version
  )

lazy val commonDotty = (crossProject(JVMPlatform)
    .crossType(CrossType.Pure) in file("commonDotty"))
  .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala-common"
  )
  .jvmSettings(
    scalaVersion := dottyVersion,
    publishArtifact in (Compile, packageDoc) := false
  )

lazy val commonJS = common.js.disablePlugins(dotty.tools.sbtplugin.DottyPlugin).disablePlugins(dotty.tools.sbtplugin.DottyIDEPlugin)

lazy val programGenerator = (crossProject(JVMPlatform)
    .crossType(CrossType.Pure) in file("programGenerator")).dependsOn(core)
    .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala-program-generator",
    libraryDependencies ++= Seq("org.bytedeco.javacpp-presets" % "onnx-platform" % "1.3.0-1.4.3"),
    scalaVersion := scala212Version,
    mainClass in (Compile, run) := Some("org.emergentorder.onnx.ONNXProgramGenerator"),
    libraryDependencies ++=  Seq(
                      ("org.scalameta" %% "scalameta" % "1.8.0").withDottyCompat(dottyVersion) 
                  ),
    publishArtifact in (Compile, packageDoc) := false
  )


lazy val core = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
    .crossType(CrossType.Pure) in file("core")).dependsOn(common)
  .enablePlugins(wartremover.WartRemover)
  .settings(commonSettings,
    name := "onnx-scala",
    scalaVersion := scala212Version,
    wartremoverErrors ++= Warts.allBut(Wart.DefaultArguments, Wart.Nothing)
    )
    .jvmSettings(
      crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version),
      scalacOptions ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("-Xsource:2.14" //For opaque types - not merged to 2.13 yet
                                           )
        case _ => Seq(
                  )
      }),

      libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("org.typelevel" % "spire_2.12" % "0.16.0",
                                            "eu.timepit" % "singleton-ops_2.12" % "0.3.1"
                                           )
        case _ => Seq("org.typelevel" %% "spire" % "0.16.0",
                      "eu.timepit" %% "singleton-ops" % "0.3.1"
                  )
      })
    )
    .jsSettings(
      crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version),
      libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("org.typelevel" % "spire_sjs0.6_2.12" % "0.16.0"  excludeAll(
    ExclusionRule(organization = "org.scala-js")),
                                            "eu.timepit" %%% "singleton-ops" % "0.3.1"
                                           )
        case _ => Seq("org.typelevel" %%% "spire" % "0.16.0",
                      "eu.timepit" %%% "singleton-ops" % "0.3.1"
                  )
      })
    )
    .nativeSettings(
      scalaVersion := scala211Version,
      libraryDependencies ++= Seq(
        "org.typelevel" %% "spire" % "0.16.0",
        "eu.timepit" %% "singleton-ops" % "0.3.1"
      )
    )

lazy val coreDotty = (crossProject(JVMPlatform) //TODO: fix fail on common in clean cross-build
  .crossType(CrossType.Pure)).in(file("coreDotty")).dependsOn(commonDotty)
  .enablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala",
    scalaVersion := dottyVersion,
    publishArtifact in (Compile, packageDoc) := false,
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2") else Nil },
    libraryDependencies ++= Seq(
      ("org.typelevel" %% "spire" % "0.16.0").withDottyCompat(dottyVersion),
      ("eu.timepit" %% "singleton-ops" % "0.3.1").withDottyCompat(dottyVersion)
    )
)

lazy val free = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
    .crossType(CrossType.Pure) in file("free")).dependsOn(core)
  .disablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .settings( commonSettings,
    name := "onnx-scala-free", 
    scalaVersion := scala212Version,
    publishArtifact in (Compile, packageDoc) := false,
    scalacOptions ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("-Ymacro-annotations"
                                           )
        case _ => Seq(
                  )
    }),
    libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n != 13 => Seq( compilerPlugin("org.scalameta" % "paradise" % "3.0.0-M11" cross CrossVersion.full) 
                                           )
        case _ => Seq(
                  )
    }),
   libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq(
                                            "org.typelevel" % "cats-free_2.12" % catsVersion,
                                            "org.typelevel" % "cats-effect_2.12" % "1.0.0",
                                            "io.frees" % "frees-core_2.12" % "0.8.2",
                                            "io.frees" % "iota-core_2.12" % "0.3.7"
                                           )
        case _ => Seq(
                      "org.typelevel" %% "cats-free" % catsVersion,
                      "org.typelevel" %% "cats-effect" % "1.0.0",
                      "io.frees" %% "frees-core" % "0.8.2",
                      "io.frees" %% "iota-core" % "0.3.7"
                  )
      })

  )
  .jvmSettings(
    crossScalaVersions := Seq(scala212Version, scala211Version), 
  )
  .jsSettings(
      crossScalaVersions := Seq(scala212Version, scala211Version),
      libraryDependencies ++= Seq(
        "org.typelevel" %%% "cats-free" % catsVersion,
        "org.typelevel" %%% "cats-effect" % "1.0.0",
        "io.frees" %%% "frees-core" % "0.8.2",
        "io.frees" %%% "iota-core" % "0.3.7"
                  ),

    )
  .nativeSettings(
      scalaVersion := scala211Version,
      libraryDependencies ++= Seq(
        "org.typelevel" %% "cats-free" % catsVersion,
        "org.typelevel" %% "cats-effect" % "1.0.0", 
        "io.frees" %% "frees-core" % "0.8.2",
        "io.frees" %% "iota-core" % "0.3.7"
      )
    )
