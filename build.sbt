import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

//val dottyVersion = dottyLatestNightlyBuild.get
val dottyVersion     = "3.1.0-RC3"
val spireVersion     = "0.17.0"
val scalaTestVersion = "3.2.10"

scalaVersion := dottyVersion

lazy val commonSettings = Seq(
//  scalaJSUseMainModuleInitializer := true, //Test only
  organization := "org.emergentorder.onnx",
  version      := "0.15.0",
  scalaVersion := dottyVersion,
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://s01.oss.sonatype.org/content/repositories/snapshots",
  updateOptions := updateOptions.value.withLatestSnapshots(false),
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation"),
  autoCompilerPlugins := true
//  sources in (Compile, doc) := Seq(), //Bug w/ Dotty & JS on doc
) ++ sonatypeSettings

lazy val common = (crossProject(JSPlatform, JVMPlatform)
   .crossType(CrossType.Pure) in file("common"))
   .settings(
     commonSettings,
     name := "onnx-scala-common",
     crossScalaVersions := Seq(
       dottyVersion
     ),
   )

lazy val proto = (crossProject(JSPlatform, JVMPlatform)
   .crossType(CrossType.Pure) in file("proto"))
   .settings(
     commonSettings,
     name := "onnx-scala-proto",
     crossScalaVersions := Seq(
       dottyVersion
     ),
     libraryDependencies -= ("com.thesamet.scalapb" %%% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion),
     libraryDependencies += ("com.thesamet.scalapb" %%% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion)
        .cross(CrossVersion.for3Use2_13),
     PB.targets in Compile := Seq(
       scalapb.gen() -> (sourceManaged in Compile).value
     ),
     // The trick is in this line:
     PB.protoSources in Compile := Seq(file("proto/src/main/protobuf"))
   )

lazy val backends = (crossProject(JVMPlatform, JSPlatform)
   .crossType(CrossType.Pure) in file("backends"))
   .dependsOn(core)
//conditionally enabling/disable based on version, still not working
//  .enablePlugins(ScalablyTypedConverterPlugin)
   .settings(
     commonSettings,
     name := "onnx-scala-backends",
//     excludeFilter in unmanagedSources := (CrossVersion
//        .partialVersion(scalaVersion.value) match {
//          case _ => "Main.scala" | "ONNXJSOperatorBackend.scala"
//     }),
//    scalacOptions ++= { if (isDotty.value) Seq("-source:3.0-migration") else Nil },
     libraryDependencies ++= Seq(
       "com.microsoft.onnxruntime" % "onnxruntime" % "1.9.0"
     ),
     crossScalaVersions := Seq(dottyVersion)
   )
   .jvmSettings(
//TODO: move to utest
     libraryDependencies += ("org.scalatest" %% "scalatest" % scalaTestVersion) % Test
   )
   .jsSettings(scalaJSUseMainModuleInitializer := true, //, //Testing
npmDependencies in Compile += "onnxruntime-web" -> "1.9.0")
//Seems to be a bundling issue, copying things manually seems to work
//TODO NEW: try JS, bundler and converter beta/RC are out
//     npmDependencies in Compile += "onnxjs" -> "0.1.8")
.jsConfigure { project => project.enablePlugins(ScalablyTypedConverterPlugin)}
//ScalaJSBundlerPlugin)} //,ScalablyTypedConverterPlugin) }

lazy val core = (crossProject(JSPlatform, JVMPlatform)
   .crossType(CrossType.Pure) in file("core"))
   .dependsOn(common)
   .dependsOn(proto)
   .settings(
     commonSettings,
     name := "onnx-scala",
//    scalacOptions ++= { if (isDotty.value) Seq("-source:3.0-migration") else Nil }, 
     crossScalaVersions := Seq(
       dottyVersion
     ),
     libraryDependencies ++= (CrossVersion
        .partialVersion(scalaVersion.value) match {
        case _ =>
           Seq(
             ("org.typelevel" %%% "spire" % spireVersion).cross(CrossVersion.for3Use2_13)
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

skip in publish := true

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
