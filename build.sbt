import org.scalajs.linker.interface.ModuleSplitStyle
import scala.sys.process.Process
import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

//val dottyVersion = dottyLatestNightlyBuild.get
val dottyVersion     = "3.7.0-RC4"
val spireVersion     = "0.18.0"
val scalaTestVersion = "3.2.19"

scalaVersion := dottyVersion

lazy val commonSettings = Seq(
  organization := "org.emergentorder.onnx",
  version      := "0.18.0",
  scalaVersion := dottyVersion,
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://s01.oss.sonatype.org/content/repositories/snapshots",
  updateOptions                               := updateOptions.value.withLatestSnapshots(false),
  libraryDependencies += "com.google.protobuf" % "protobuf-java"     % "4.30.0",
  libraryDependencies += "org.scala-lang"      % "scala3-compiler_3" % scalaVersion.value exclude (
    "org.scala-sbt",
    "compiler-interface"
  ),
  scalacOptions ++= Seq(
    "-explain",
    "-explain-types",
    "-feature",
//    "-Xfatal-warnings",
    "-unchecked",
    "-deprecation",
//    "-release:21",
    "-rewrite",
    "-source:3.7-migration"
  ),
  versionPolicyIntention := Compatibility.BinaryCompatible, // As long as we are pre 1.0.0, BinaryCompatible for a patch version bump and None for a minor version bump
  versionScheme         := Some("early-semver"),
  mimaPreviousArtifacts := Set("org.emergent-order" %%% "onnx-scala-common" % "0.17.0"),
  autoCompilerPlugins   := true
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
     scalaJSStage in Global:= FullOptStage
   )

lazy val proto = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
   .crossType(CrossType.Pure) in file("proto"))
   .settings(
     commonSettings,
     name                  := "onnx-scala-proto",
     mimaPreviousArtifacts := Set("org.emergent-order" %%% "onnx-scala-proto" % "0.17.0"),
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
     scalaJSStage in Global := FullOptStage 
   )

val copyIndexTs = taskKey[Unit]("Copy ts types file to target directory")

val copyPackageNoExports = taskKey[Unit]("Copy package file without exports to target directory")

lazy val backends = (crossProject(JSPlatform, JVMPlatform)
   .crossType(CrossType.Pure) in file("backends"))
   .dependsOn(core)
   .settings(
     commonSettings,
     name                  := "onnx-scala-backends",
     mimaPreviousArtifacts := Set("org.emergent-order" %%% "onnx-scala-backends" % "0.17.0"),
     libraryDependencies ++= Seq(
       "com.microsoft.onnxruntime" % "onnxruntime"            % "1.21.1",
       "com.microsoft.onnxruntime" % "onnxruntime-extensions" % "0.13.0"
     ),
     libraryDependencies += ("org.scalatest" %%% "scalatest" % scalaTestVersion) % Test,
     crossScalaVersions                       := Seq(dottyVersion)
   )
   .jvmSettings(
     libraryDependencies += "org.typelevel" %%% "cats-effect-testing-scalatest" % "1.6.0" % Test
   )
   .jsSettings( 
     scalaJSUseMainModuleInitializer := true,
     mainClass := Some("org.emergentorder.onnx.backends.Main"), // "livechart.LiveChart"), // , //Testing
// stuck on web/node 1.15.1 due to this issue: https://github.com/microsoft/onnxruntime/issues/17979
    scalaJSLinkerConfig ~= {
      _.withModuleKind(ModuleKind.ESModule)
//      .withExperimentalUseWebAssembly(true) //wasm works in node, breaks in browser (even when enabled there)
 //       .withModuleSplitStyle(ModuleSplitStyle.FewestModules)
//          .withModuleSplitStyle(ModuleSplitStyle.SmallModulesFor(List("backendsJS")))
       .withESFeatures(
       _.withESVersion(org.scalajs.linker.interface.ESVersion.ES2021))
    },
     // Configure Node.js (at least v23) to support the required Wasm features
     jsEnv := {
     val config = org.scalajs.jsenv.nodejs.NodeJSEnv.Config()
       .withArgs(List(
         "--experimental-wasm-exnref", // required
         "--experimental-wasm-imported-strings", // optional (good for performance)
         "--turboshaft-wasm", // optional, but significantly increases stability
      ))
      new org.scalajs.jsenv.nodejs.NodeJSEnv(config)
    },
    libraryDependencies += "org.scala-js" %%% "scalajs-dom" % "2.8.0",
 //    Compile / npmDependencies += "onnxruntime-web" -> "1.22.0-dev.20250310-fe7634eb6f",
     // ORT web and node are interchangeable, given minor package name changes, and node offers a significant speed-up (at the cost of working on the web)
//     Compile / npmDependencies += "onnxruntime-node"   -> "1.21.0",
//     Compile / npmDependencies += "onnxruntime-common" -> "1.22.0-dev.20250306-aafa8d170a",
//     Compile / npmDependencies += "typescript" -> "5.8.2",
//     Compile / npmDevDependencies += "copy-webpack-plugin" -> "13.0.0",
//     requireJsDomEnv in Test := true,
//     webpack / version := "5.98.0",
//     webpackConfigFile := Some(baseDirectory.value / "webpack.config.js"),
     // this fixes above error I was getting, per github issue in webpack repo
//     webpackCliVersion := "5.1.4",
//     startWebpackDevServer / version := "4.15.2",
//     version in webpack := "5.98.0",
//     version in startWebpackDevServer := "5.2.0",
//     Compile / npmDependencies += "webpack" -> "5.98.0", 
//     Compile / npmDevDependencies += "webpack" -> "5.98.0",
//     Compile / npmDependencies += "webpack-cli" -> "6.0.1",
//     Compile / npmDependencies += "webpack-dev-server" -> "5.2.0",
     sources in (Compile, doc) := Nil,
//     Compile / stMinimize := Selection.AllExcept("onnxruntime-common", "onnxruntime-web", "onnxruntime-node"),
//    Compile / npmDependencies += "typescript"         -> "5.5.4",
     copyIndexTs := {
        import Path._

        val src = new File(".")

        // get the files we want to copy
        val htmlFiles: Seq[File] = Seq(new File("index.d.ts"))

        // use Path.rebase to pair source files with target destination in crossTarget
        val pairs = htmlFiles pair rebase(
          src,
          (baseDirectory).value / "node_modules/onnxruntime-node/dist/"
        )

        // Copy files to source files to target
        IO.copy(
          pairs,
          CopyOptions
             .apply(overwrite = true, preserveLastModified = true, preserveExecutable = false)
        )

     },
     copyPackageNoExports := {
        import Path._

        val src = new File(".")

        // get the files we want to copy
        val htmlFiles: Seq[File] = Seq(new File("package.json"))

        // use Path.rebase to pair source files with target destination in crossTarget
        val pairs = htmlFiles pair rebase(
          src,
          (baseDirectory).value / "node_modules/onnxruntime-common/"
        )

        // Copy files to source files to target
        IO.copy(
          pairs,
          CopyOptions
             .apply(overwrite = true, preserveLastModified = true, preserveExecutable = false)
        )
     },
     Compile / compile := (Compile / compile dependsOn (copyIndexTs, copyPackageNoExports)).value,
//     Test / test := (Test / test dependsOn (copyPackageNoExports)).value,
     libraryDependencies += "org.typelevel" %%% "cats-effect-testing-scalatest" % "1.6.0" % Test,
     stOutputPackage                         := "org.emergentorder.onnx",
     stShortModuleNames                      := true,
     Compile / packageDoc / publishArtifact := false, // This is inordinately slow, only publish doc on release
     scalaJSStage := FullOptStage, 
    externalNpm := {
      Process("npm install", baseDirectory.value).!
//      Process("npm run dev", baseDirectory.value).!
       baseDirectory.value
    }
//     scalaJSLinkerConfig ~= { _.withESFeatures(_.withESVersion(scala.scalajs.LinkingInfo.ESVersion.ES2021)) }
   )
   // For distribution as a library, using ScalablyTypedConverterGenSourcePlugin (vs ScalablyTypedConverterPlugin) is required
   // which slows down the build (particularly the doc build, for publishing) considerably
   // TODO: minimize to reduce build time and size of js output
   .jsConfigure { project => project.enablePlugins(ScalablyTypedConverterExternalNpmPlugin)}
//ScalablyTypedConverterExternalNpmPlugin) }

lazy val core = (crossProject(JSPlatform, JVMPlatform, NativePlatform)
   .crossType(CrossType.Pure) in file("core"))
   .dependsOn(common)
   .dependsOn(proto)
   .settings(
     commonSettings,
     name                  := "onnx-scala",
     mimaPreviousArtifacts := Set("org.emergent-order" %%% "onnx-scala" % "0.17.0"),
     crossScalaVersions := Seq(
       dottyVersion
     ),
     libraryDependencies ++= (CrossVersion
        .partialVersion(scalaVersion.value) match {
        case _ =>
           Seq(
             ("org.typelevel" %%% "spire"       % spireVersion),
             ("org.typelevel" %%% "cats-effect" % "3.6.1")
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
