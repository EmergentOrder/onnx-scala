//import org.scalajs.linker.interface.ModuleSplitStyle
import scala.sys.process.Process
//import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

//TODO: figure out why tests got a lot slower after moving to sbt-projectmatrix

//val dottyVersion = dottyLatestNightlyBuild.get
val scala3Version    = "3.8.2-RC2"
val spireVersion     = "0.18.0" //-156-0fe5a6a-20251027T014354Z-SNAPSHOT"
val scalaTestVersion = "3.3.0-alpha.2"

scalaVersion := scala3Version

inThisBuild(
  List(
    scalaVersion      := scala3Version,
    semanticdbEnabled := true,
    semanticdbVersion := scalafixSemanticdb.revision
  )
)

lazy val commonSettings = Seq(
  organization := "org.emergentorder.onnx",
  version      := "0.18.0",
  scalaVersion := scala3Version,
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://s01.oss.sonatype.org/content/repositories/snapshots",
  updateOptions                               := updateOptions.value.withLatestSnapshots(false),
  libraryDependencies += "com.google.protobuf" % "protobuf-java" % "4.34.0-RC2",
  PB.protocVersion                            := "4.34.0-RC2",
//  (Test / parallelExecution) := false,
  scalacOptions ++= Seq(
    // "-new-syntax",
    "-explain",
    "-explain-types",
    "-feature",
    "-unchecked",
    "-deprecation",
//    "-release:25",
    "-rewrite",
//    "-source:future-migration",
    "-source:3.8-migration",
    "-Yimplicit-to-given",
    "-Wunused:all",
    "-Wnonunit-statement",
    "-WunstableInlineAccessors",
    "-Wsafe-init"
  ),
//  versionPolicyIntention := Compatibility.BinaryCompatible, // As long as we are pre 1.0.0, BinaryCompatible for a patch version bump and None for a minor version bump
  versionScheme         := Some("early-semver"),
  mimaPreviousArtifacts := Set("org.emergent-order" %% "onnx-scala-common" % "0.17.0"),
  autoCompilerPlugins   := true
)

//++ sonatypeSettings

lazy val common = (projectMatrix in file("common"))
   .jvmPlatform(scalaVersions = Seq(scala3Version))
//   .jsPlatform(scalaVersions = Seq(scala3Version), scalaJSStage in Global := FullOptStage)
//   .nativePlatform(scalaVersions = Seq(scala3Version))
//(crossProject(JSPlatform, JVMPlatform, NativePlatform)
   .settings(
     commonSettings,
     scalacOptions ++= Seq("-Werror"), // , "-language:future"),
     name := "onnx-scala-common"
   )

lazy val proto = (projectMatrix in file("proto"))
   .jvmPlatform(scalaVersions = Seq(scala3Version))
//   .jsPlatform(scalaVersions = Seq(scala3Version), scalaJSStage in Global := FullOptStage)
//   .nativePlatform(scalaVersions = Seq(scala3Version))
   .settings(
     commonSettings,
     name := "onnx-scala-proto",
     scalacOptions ++= Seq("-Werror"),
     libraryDependencies += ("com.thesamet.scalapb" %% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion % "protobuf")
        .exclude("org.scala-lang.modules", "scala-collection-compat_3"),
     Compile / PB.targets := Seq(
       scalapb.gen(
         scala3Sources = true,
         grpc = false,
         lenses = false
       ) -> (Compile / sourceManaged).value / "scalapb"
     ),
//     // The trick is in this line:
     Compile / PB.protoSources := Seq(file("proto/src/main/protobuf/"))
   )

//val copyIndexTs = taskKey[Unit]("Copy ts types file to target directory")

//val copyPackageNoExports = taskKey[Unit]("Copy package file without exports to target directory")

//val copyPackageNoExportsAgain =
//   taskKey[Unit]("Copy package file without exports to target directory")

//val copyPackageFull = taskKey[Unit]("Copy full package file")

//Enabling NativePlatform here requires custom-built spire
lazy val core = (projectMatrix in file("core"))
   .jvmPlatform(scalaVersions = Seq(scala3Version))
//   .jsPlatform(scalaVersions = Seq(scala3Version), scalaJSStage in Global := FullOptStage)
//   .nativePlatform(scalaVersions = Seq(dottyVersion))
   .dependsOn(common)
   .dependsOn(proto)
   .settings(
     commonSettings,
     scalacOptions ++= Seq("-Werror"), // , "-language:future"),
     name := "onnx-scala",
     libraryDependencies ++= (CrossVersion
        .partialVersion(scalaVersion.value) match {
        case _ =>
           Seq(
             ("org.typelevel" %% "spire"       % spireVersion),
             ("org.typelevel" %% "cats-effect" % "3.7.0-RC1"), // -5d10115"),
             ("org.typelevel" %% "cats-mtl"    % "1.6.0"),
             ("org.typelevel" %% "algebra"     % "2.13.0")
           )
     })
   )

lazy val backends = (projectMatrix in file("backends"))
//TODO: restore  //.enablePlugins(ScalablyTypedConverterExternalNpmPlugin)
   .jvmPlatform(scalaVersions = Seq(scala3Version))
   .dependsOn(core)
   .settings(
     commonSettings,
     scalacOptions ++= Seq("-Werror"), // , "-language:future"),
     name := "onnx-scala-backends",
     libraryDependencies ++= Seq(
       "org.typelevel"            %% "cats-effect-testing-scalatest" % "1.7.0" % Test,
       "com.microsoft.onnxruntime" % "onnxruntime"                   % "1.24.1", // "1.23.0-RC2",
       "com.microsoft.onnxruntime" % "onnxruntime-extensions"        % "0.13.0"
     ),
     libraryDependencies += ("org.scalatest" %% "scalatest" % scalaTestVersion) % Test
//     libraryDependencies += ("org.scalactic" %% "scalactic" % scalaTestVersion),
   )

//lazy val backendsFound = backends
//   .js(scala3Version)
//   .enablePlugins(ScalablyTypedConverterExternalNpmPlugin)
//   .enablePlugins(ScalablyTypedConverterGenSourcePlugin)

// For distribution as a library, using ScalablyTypedConverterGenSourcePlugin (vs ScalablyTypedConverterPlugin) is required
// which slows down the build (particularly the doc build, for publishing) considerably
// TODO: minimize to reduce build time and size of js output
//   .jsConfigure { project => project.enablePlugins(ScalablyTypedConverterExternalNpmPlugin) }
//   .jvmConfigure { project =>
//      project.enablePlugins(JavaAppPackaging) //, GraalVMNativeImagePlugin)
//   } //GraalVMNativeImagePlugin) }
//ScalablyTypedConverterExternalNpmPlugin) }

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

/*
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
 */
