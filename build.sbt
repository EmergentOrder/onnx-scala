import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

//val dottyVersion = dottyLatestNightlyBuild.get
val dottyVersion     = "3.2.1"
val spireVersion     = "0.18.0"
val scalaTestVersion = "3.2.14"

scalaVersion := dottyVersion

lazy val commonSettings = Seq(
  organization := "org.emergentorder.onnx",
  version      := "0.17.0",
  scalaVersion := dottyVersion,
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://s01.oss.sonatype.org/content/repositories/snapshots",
  updateOptions := updateOptions.value.withLatestSnapshots(false),
  scalacOptions ++= Seq(
    "-explain",
    "-explain-types",
    "-feature",
    "-Xfatal-warnings",
    "-unchecked",
    "-deprecation",
//    "-release:19",
    "-source:3.2"
  ),
  autoCompilerPlugins := true
) ++ sonatypeSettings

lazy val common = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
   .crossType(CrossType.Pure) in file("common"))
   .settings(
     commonSettings,
     name := "onnx-scala-common",
     crossScalaVersions := Seq(
       dottyVersion
     )
   )
   .jsSettings(
     scalaJSStage := FullOptStage
   )

lazy val proto = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
   .crossType(CrossType.Pure) in file("proto"))
   .settings(
     commonSettings,
     name := "onnx-scala-proto",
     crossScalaVersions := Seq(
       dottyVersion
     ),
     Compile / PB.targets := Seq(
       scalapb.gen() -> (Compile / sourceManaged).value / "scalapb"
     ),
     // The trick is in this line:
     Compile / PB.protoSources := Seq(file("proto/src/main/protobuf"))
   )
   .jsSettings(
     scalaJSStage := FullOptStage
   )

lazy val backends = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
   .crossType(CrossType.Pure) in file("backends"))
   .dependsOn(core)
   .settings(
     commonSettings,
     name := "onnx-scala-backends",
     libraryDependencies ++= Seq(
       "com.microsoft.onnxruntime" % "onnxruntime" % "1.13.1"
     ),
     libraryDependencies += ("org.scalatest" %%% "scalatest" % scalaTestVersion) % Test,
     crossScalaVersions                       := Seq(dottyVersion)
   )
   .jvmSettings(
     libraryDependencies += "org.typelevel" %%% "cats-effect-testing-scalatest" % "1.5.0" % Test
   )
   .jsSettings(
     webpack / version                                 := "5.74.0",
     webpackCliVersion                                 := "4.10.0",
     startWebpackDevServer / version                   := "4.11.1",
     scalaJSUseMainModuleInitializer                   := true, // , //Testing
     Compile / npmDependencies += "onnxruntime-node"   -> "1.13.1",
     Compile / npmDependencies += "onnxruntime-common" -> "1.13.1",
     Compile / npmDependencies += "typescript"         -> "4.8.4",
     libraryDependencies += "org.typelevel" %%% "cats-effect-testing-scalatest" % "1.5.0" % Test,
     stOutputPackage                         := "org.emergentorder.onnx",
     scalaJSStage                            := FullOptStage,
     scalaJSLinkerConfig ~= (_.withESFeatures(
       _.withESVersion(org.scalajs.linker.interface.ESVersion.ES2021)
     ))
//     scalaJSLinkerConfig ~= { _.withESFeatures(_.withESVersion(scala.scalajs.LinkingInfo.ESVersion.ES2021)) }
   )
   // For distribution as a library, using ScalablyTypedConverterGenSourcePlugin (vs ScalablyTypedConverterPlugin) is required
   // which slows down the build (particularly the doc build, for publishing) considerably
   // TODO: minimize to reduce build time and size of js output
   .jsConfigure { project => project.enablePlugins(ScalablyTypedConverterGenSourcePlugin) }

lazy val core = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
   .crossType(CrossType.Pure) in file("core"))
   .dependsOn(common)
   .dependsOn(proto)
   .settings(
     commonSettings,
     name := "onnx-scala",
     crossScalaVersions := Seq(
       dottyVersion
     ),
     libraryDependencies ++= (CrossVersion
        .partialVersion(scalaVersion.value) match {
        case _ =>
           Seq(
             ("org.typelevel" %%% "spire"       % spireVersion),
             ("org.typelevel" %%% "cats-effect" % "3.4.0")
           )
     })
   )
   .jsSettings(
     scalaJSStage in Global := FullOptStage
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

publish / skip := true

lazy val sonatypeSettings = Seq(
  sonatypeProfileName                := "org.emergent-order",
  sonatypeCredentialHost             := "s01.oss.sonatype.org",
  sonatypeRepository                 := "https://s01.oss.sonatype.org/service/local",
  ThisBuild / sonatypeCredentialHost := "s01.oss.sonatype.org",
  organization                       := "org.emergent-order",
  homepage                           := Some(url("https://github.com/EmergentOrder/onnx-scala")),
  scmInfo := Some(
    ScmInfo(
      url("https://github.com/EmergentOrder/onnx-scala"),
      "git@github.com:EmergentOrder/onnx-scala.git"
    )
  ),
  developers := List(
    Developer(
      "EmergentOrder",
      "Alex Merritt",
      "lecaran@gmail.com",
      url("https://github.com/EmergentOrder")
    )
  ),
  licenses += ("AGPL-3.0", url("https://www.gnu.org/licenses/agpl-3.0.html")),
  publishMavenStyle         := true,
  publishConfiguration      := publishConfiguration.value.withOverwrite(true),
  publishLocalConfiguration := publishLocalConfiguration.value.withOverwrite(true),
  publishTo := {
     val nexus = "https://s01.oss.sonatype.org/"
     if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
     else Some("releases" at nexus + "service/local/staging/deploy/maven2")
  }
)
