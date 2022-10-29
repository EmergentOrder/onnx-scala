import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

//val dottyVersion = dottyLatestNightlyBuild.get
val dottyVersion     = "3.2.0"
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

lazy val backends = (crossProject(JVMPlatform, JSPlatform)
   .crossType(CrossType.Pure) in file("backends"))
   .dependsOn(core)
   .settings(
     commonSettings,
     name := "onnx-scala-backends",
     libraryDependencies ++= Seq(
       "com.microsoft.onnxruntime" % "onnxruntime" % "1.13.1"
     ),
     libraryDependencies += ("org.scalatest" %%% "scalatest" % scalaTestVersion) % Test,
     crossScalaVersions := Seq(dottyVersion)
   )
   .jvmSettings(
   )
   .jsSettings(
     scalaJSUseMainModuleInitializer                := true, // , //Testing
     Compile / npmDependencies += "onnxruntime-node" -> "1.13.1",
     Compile / npmDependencies += "onnxruntime-common" -> "1.13.1"
   )
   .jsConfigure { project => project.enablePlugins(ScalablyTypedConverterPlugin) }

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
             ("org.typelevel" %%% "spire" % spireVersion)
           )
     })
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
