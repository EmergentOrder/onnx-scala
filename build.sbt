import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}
val dottyVersion = "0.15.0-RC1"
val scala211Version = "2.11.12"
val scala212Version = "2.12.8"
val scala213Version = "2.13.0-RC3"
//Might want to remove cats ( conflict with Freestyle's version)
val catsVersion = "2.0.0-M1" //"1.6.0"
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
    crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version),
    publishArtifact in (Compile, packageDoc) := false
  )
  .jsSettings(
    scalaVersion := scala212Version,
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
    scalaVersion := scala212Version,
    mainClass in (Compile, run) := Some("org.emergentorder.onnx.ONNXProgramGenerator"),
    libraryDependencies ++=  Seq(
                      ("org.scalameta" %% "scalameta" % "1.8.0").withDottyCompat(dottyVersion) 
                  ),
    publishArtifact in (Compile, packageDoc) := false
  )


lazy val backends = (crossProject(JVMPlatform, JSPlatform, NativePlatform)
    .crossType(CrossType.Pure) in file("backends")).dependsOn(core) //Should split into core and free?
    .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala-backends",
    scalaVersion := scala212Version,
   libraryDependencies ++= Seq("org.bytedeco" % "ngraph-platform" % "0.19.0-1.5.1-SNAPSHOT"),
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
      crossScalaVersions := Seq(scala212Version, scala213Version, scala211Version),
      scalacOptions ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("-Xsource:2.14" //For opaque types - not merged to 2.13 yet
                                           )
        case _ => Seq(
                  )
      }),
      libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("org.typelevel" % "spire_2.12" % "0.16.1"
                                      //      "eu.timepit" % "singleton-ops_2.12" % "0.3.1"
                                           )
        case _ => Seq("org.typelevel" %% "spire" % "0.16.1"
                      //"eu.timepit" %% "singleton-ops" % "0.3.1"
                  )
      }),
      libraryDependencies ++= Seq("org.bytedeco" % "onnx-platform" % "1.5.0-1.5.1-SNAPSHOT") 
    )
    .jsSettings(
      crossScalaVersions := Seq(scala212Version, scala211Version, scala213Version),

      libraryDependencies ++= Seq("org.bytedeco" % "onnx-platform" % "1.5.0-1.5.1-SNAPSHOT"),
      libraryDependencies ++= (CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((2, n)) if n == 13 => Seq("org.typelevel" % "spire_sjs0.6_2.12" % "0.16.1"  excludeAll(
    ExclusionRule(organization = "org.scala-js")),
                                           // "eu.timepit" %%% "singleton-ops" % "0.3.1"
                                           )
        case _ => Seq("org.typelevel" %%% "spire" % "0.16.1",
                      //"eu.timepit" %%% "singleton-ops" % "0.3.1"
                  )
      })
    )
    .nativeSettings(
      scalaVersion := scala211Version,
      libraryDependencies ++= Seq(
        "org.typelevel" %% "spire" % "0.16.1",
        "org.bytedeco" % "onnx-platform" % "1.5.0-1.5.1-SNAPSHOT", 
        //"eu.timepit" %% "singleton-ops" % "0.3.1",
      )
    )

lazy val coreDotty = (crossProject(JVMPlatform)
  .crossType(CrossType.Pure)).in(file("coreDotty")).dependsOn(commonDotty)
  .enablePlugins(dotty.tools.sbtplugin.DottyPlugin)
  .disablePlugins(wartremover.WartRemover)
  .settings( commonSettings,
    name := "onnx-scala",
    scalaVersion := dottyVersion,
    publishArtifact in (Compile, packageDoc) := false,
    scalacOptions ++= { if (isDotty.value) Seq("-language:Scala2") else Nil },
    libraryDependencies ++= Seq(
      ("org.typelevel" %% "spire" % "0.16.1").withDottyCompat(dottyVersion),
      //("eu.timepit" %% "singleton-ops" % "0.3.1").withDottyCompat(dottyVersion)
    )
)

lazy val free = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
    .crossType(CrossType.Pure) in file("free")).dependsOn(backends)
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
                                            "org.typelevel" % "cats-effect_2.12" % "1.2.0",
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
