import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
val dottyVersion = "0.16.0-RC3"
val scala211Version = "2.11.12"
val scala212Version = "2.12.8"
val scala213Version = "2.13.0"
val spireVersion = "0.17.0-M1"

scalaVersion := scala212Version

//TODO: Replace wartremover with scalafix

lazy val commonSettings = Seq(
  scalaJSUseMainModuleInitializer := true, //Test only
  organization := "org.emergentorder.onnx",
  version      := "1.5.0-0.1.0-SNAPSHOT", //TODO : Generate APIs for 1.5.0
  resolvers += Resolver.mavenLocal, //TODO: fix issue with mkl-dnn JavaCPP preset resolution
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
    crossScalaVersions := Seq(dottyVersion, scala212Version, scala213Version, scala211Version),
    publishArtifact in (Compile, packageDoc) := false
  )
  .jsSettings(
    scalaVersion := scala212Version,
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
  )
  .nativeSettings(
    scalaVersion := scala211Version 
  )


lazy val commonJS = common.js.disablePlugins(dotty.tools.sbtplugin.DottyPlugin).disablePlugins(dotty.tools.sbtplugin.DottyIDEPlugin)

lazy val programGenerator = (crossProject(JSPlatform, JVMPlatform)
    .crossType(CrossType.Pure) in file("programGenerator")).dependsOn(backends)
    .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala-program-generator",
    scalaVersion := scala212Version,
    run := Defaults.runTask(fullClasspath in Runtime, mainClass in run in Compile, runner in run).evaluated,
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version),
    mainClass in (Compile, run) := Some("org.emergentorder.onnx.ONNXProgramGenerator"),
    libraryDependencies ++=  Seq(
                      ("org.scalameta" %% "scalameta" % "4.1.12").withDottyCompat(dottyVersion)
                  ),
    publishArtifact in (Compile, packageDoc) := false
  )


lazy val backends = (crossProject(JVMPlatform, JSPlatform, NativePlatform)
    .crossType(CrossType.Pure) in file("backends")).dependsOn(core)
    .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala-backends",
    scalaVersion := scala212Version,
   libraryDependencies ++= Seq("org.bytedeco" % "ngraph-platform" % "0.21.0-1.5.1-SNAPSHOT"),
    publishArtifact in (Compile, packageDoc) := false
  )
  .jvmSettings(
    crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version)
  )
 .jsSettings(
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version)
  )
  .nativeSettings(
    scalaVersion := scala211Version
  )



lazy val core = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
    .crossType(CrossType.Pure) in file("core")).dependsOn(common)
  .disablePlugins(wartremover.WartRemover)
  .settings(commonSettings,
    name := "onnx-scala",
    scalaVersion := scala212Version,
    wartremoverErrors ++= Warts.allBut(Wart.DefaultArguments, Wart.Nothing)
    )
    .jvmSettings(
      crossScalaVersions := Seq(dottyVersion, scala212Version, scala213Version, scala211Version),
      scalacOptions ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("-Xsource:2.14" //For opaque types - not merged to 2.13 yet
                                           )
        case _ => Seq(
                  )
      }),
      libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("org.typelevel" % "spire_2.12" % spireVersion
                                      //      "eu.timepit" % "singleton-ops_2.12" % "0.3.1"
                                           )
        case _ => Seq(("org.typelevel" %% "spire" % spireVersion).withDottyCompat(dottyVersion)
 
                     //"eu.timepit" %% "singleton-ops" % "0.3.1"
                  )
      }),
      libraryDependencies ++= Seq(("org.bytedeco" % "onnx-platform" % "1.5.0-1.5.1-SNAPSHOT").withDottyCompat(dottyVersion))
    )
    .jsSettings(
      crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version),

      libraryDependencies ++= Seq("org.bytedeco" % "onnx-platform" % "1.5.0-1.5.1-SNAPSHOT"),
      libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("org.typelevel" % "spire_sjs0.6_2.12" % spireVersion  excludeAll(
    ExclusionRule(organization = "org.scala-js")),
                                           // "eu.timepit" %%% "singleton-ops" % "0.3.1"
                                           )
        case _ => Seq("org.typelevel" %%% "spire" % spireVersion,
                      //"eu.timepit" %%% "singleton-ops" % "0.3.1"
                  )
      })
    )
    .nativeSettings(
      scalaVersion := scala211Version,
      libraryDependencies ++= Seq(
        "org.typelevel" %% "spire" % spireVersion,
        "org.bytedeco" % "onnx-platform" % "1.5.0-1.5.1-SNAPSHOT", 
        //"eu.timepit" %% "singleton-ops" % "0.3.1",
      )
    )

lazy val zio = (crossProject(JVMPlatform, JSPlatform)
    .crossType(CrossType.Pure) in file("zio")).dependsOn(backends)
  .disablePlugins(wartremover.WartRemover) 
  .disablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .settings( commonSettings,
    name := "onnx-scala-zio",
    scalaVersion := scala212Version,
    run := Defaults.runTask(fullClasspath in Runtime, mainClass in run in Compile, runner in run).evaluated,
    crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version), 
    publishArtifact in (Compile, packageDoc) := false,
    scalacOptions ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("-Ymacro-annotations"
                                           )
        case _ => Seq(
                  )
    }),


   libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq(
                                            "org.scalaz" % "scalaz-zio_2.12" % "1.0-RC5"

                                           )
        case _ => Seq(
                      "org.scalaz" %% "scalaz-zio" % "1.0-RC5"
                  )
      })

  )
