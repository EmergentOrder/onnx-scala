import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
val dottyVersion = dottyLatestNightlyBuild.get //"0.9.0-RC1"
val scala211Version = "2.11.12"
val scala212Version = "2.12.6"
val scala213Version = "2.13.0-M5"
//Might want to remove cats ( conflict with Freestyle's version)
val catsVersion = "1.4.0"

lazy val commonSettings = Seq(

  organization := "org.emergentorder.onnx",
  version      := "1.3.0-0.1.0-SNAPSHOT",
  resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  resolvers += "Sonatype OSS Staging" at "https://oss.sonatype.org/content/repositories/staging/",
  updateOptions := updateOptions.value.withLatestSnapshots(false),
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation"),
  autoCompilerPlugins := true
)

lazy val common = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
    .crossType(CrossType.Pure) in file("common"))
  .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala-common",
    libraryDependencies ++= Seq("org.bytedeco.javacpp-presets" % "onnx-platform" % "1.3.0-1.4.3-SNAPSHOT")
  )
  .jvmSettings(
    crossScalaVersions := Seq(dottyVersion, scala212Version, scala213Version, scala211Version),
    libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("org.typelevel" % "spire_2.12" % "0.16.0",
                                            "org.scalameta" % "scalameta_2.12" % "1.8.0"
                                           )
        case _ => Seq(("org.typelevel" %% "spire" % "0.16.0").withDottyCompat(dottyVersion),
                      ("org.scalameta" %% "scalameta" % "1.8.0").withDottyCompat(dottyVersion)
                  )
      }),
    publishArtifact in (Compile, packageDoc) := false
  )
  .jsSettings(
    crossScalaVersions := Seq(scala212Version, scala211Version),
    libraryDependencies ++= Seq("org.typelevel" %%% "spire" % "0.16.0",
                                "org.scalameta" %%% "scalameta" % "1.8.0"
      )
  )
  .nativeSettings(
    scalaVersion := scala211Version,
    libraryDependencies ++= Seq(
        "org.typelevel" %% "spire" % "0.16.0",
        "org.scalameta" %% "scalameta" % "1.8.0"
      )
  )

lazy val commonJS     = common.js.disablePlugins(dotty.tools.sbtplugin.DottyPlugin).disablePlugins(dotty.tools.sbtplugin.DottyIDEPlugin)

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
        case Some((2, n)) if n == 13 => Seq(
                                            "eu.timepit" % "singleton-ops_2.12" % "0.3.0"
                                           )
        case _ => Seq(
                      "eu.timepit" %% "singleton-ops" % "0.3.0"
                  )
      })
    )
    .jsSettings(
      crossScalaVersions := Seq(scala212Version, scala211Version),
      libraryDependencies ++= Seq(
        "eu.timepit" %%% "singleton-ops" % "0.3.0"
      )
    )
    .nativeSettings(
      scalaVersion := scala211Version,
      libraryDependencies ++= Seq(
        "eu.timepit" %% "singleton-ops" % "0.3.0"
      )
    )

lazy val coreDotty = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure)).in(file("coreDotty")).dependsOn(common)
  .enablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala",
    scalaVersion := dottyVersion,
    publishArtifact in (Compile, packageDoc) := false,
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2") else Nil },
    libraryDependencies ++= Seq(
      ("org.typelevel" %% "spire" % "0.16.0").withDottyCompat(dottyVersion),
      ("eu.timepit" %% "singleton-ops" % "0.3.0").withDottyCompat(dottyVersion)
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
    crossScalaVersions := Seq(scala212Version), //TODO: restore scala213Version, scala211Version
  )
  .jsSettings(
      crossScalaVersions := Seq(scala212Version, scala211Version),
      libraryDependencies ++= Seq(
        "org.typelevel" %%% "cats-free" % catsVersion,
        "org.typelevel" %%% "cats-effect" % "1.0.0",
        "io.frees" %%% "frees-core" % "0.8.2",
        "io.frees" %%% "iota-core" % "0.3.7"
      )
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


lazy val freeDotty = (crossProject(JVMPlatform)
    .crossType(CrossType.Pure) in file("freeDotty")).dependsOn(coreDotty)
  .enablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala-free",
    scalaVersion := dottyVersion,
    publishArtifact in (Compile, packageDoc) := false,
    libraryDependencies ++= Seq(
      ("io.frees" %% "frees-core" % "0.8.2").withDottyCompat(dottyVersion),
      ("io.frees" %% "iota-core" % "0.3.7").withDottyCompat(dottyVersion),
      ("org.typelevel" %% "cats-free" % catsVersion).withDottyCompat(dottyVersion),
      ("org.typelevel" %% "cats-effect" % "1.0.0").withDottyCompat(dottyVersion),
      (compilerPlugin("org.scalameta" % "paradise_2.12.6" % "3.0.0-M11")
    )
  )

)
