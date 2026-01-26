import org.scalajs.linker.interface.ModuleSplitStyle
import scala.sys.process.Process
//import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

//TODO: figure out why tests got a lot slower after moving to sbt-projectmatrix

//val dottyVersion = dottyLatestNightlyBuild.get
val dottyVersion     = "3.8.2-RC1" //3.7.4 requires newer sbt-converter 45
val spireVersion     = "0.18.0"    //-156-0fe5a6a-20251027T014354Z-SNAPSHOT"
val scalaTestVersion = "3.3.0-alpha.2"

scalaVersion := dottyVersion

inThisBuild(
  List(
    scalaVersion      := dottyVersion,
    semanticdbEnabled := true,
    semanticdbVersion := scalafixSemanticdb.revision
  )
)

lazy val commonSettings = Seq(
  organization := "org.emergentorder.onnx",
  version      := "0.18.0",
  scalaVersion := dottyVersion,
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://s01.oss.sonatype.org/content/repositories/snapshots",
  updateOptions                               := updateOptions.value.withLatestSnapshots(false),
  libraryDependencies += "com.google.protobuf" % "protobuf-java"     % "4.33.4",
  libraryDependencies += "org.scala-lang"      % "scala3-compiler_3" % scalaVersion.value exclude (
    "org.scala-sbt",
    "compiler-interface"
  ),
//  (Test / parallelExecution) := false,
  scalacOptions ++= Seq(
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
    "-WunstableInlineAccessors"
  ),
  versionPolicyIntention := Compatibility.BinaryCompatible, // As long as we are pre 1.0.0, BinaryCompatible for a patch version bump and None for a minor version bump
  versionScheme         := Some("early-semver"),
  mimaPreviousArtifacts := Set("org.emergent-order" %%% "onnx-scala-common" % "0.17.0"),
  autoCompilerPlugins   := true
)
//++ sonatypeSettings

lazy val common = (projectMatrix in file("common"))
   .jvmPlatform(scalaVersions = Seq(dottyVersion))
   .jsPlatform(scalaVersions = Seq(dottyVersion), scalaJSStage in Global := FullOptStage)
   .nativePlatform(scalaVersions = Seq(dottyVersion))
//(crossProject(JSPlatform, JVMPlatform, NativePlatform)
   .settings(
     commonSettings,
     scalacOptions ++= Seq("-Werror"), // , "-language:future"),
     name := "onnx-scala-common"
   )

lazy val proto = (projectMatrix in file("proto"))
   .jvmPlatform(scalaVersions = Seq(dottyVersion))
   .jsPlatform(scalaVersions = Seq(dottyVersion), scalaJSStage in Global := FullOptStage)
   .nativePlatform(scalaVersions = Seq(dottyVersion))
   .settings(
     commonSettings,
     name := "onnx-scala-proto",
     scalacOptions ++= Seq("-Werror"),
     libraryDependencies += "com.thesamet.scalapb" %% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion % "protobuf",
     Compile / PB.targets := Seq(
       scalapb.gen() -> (Compile / sourceManaged).value / "scalapb"
     ),
     // The trick is in this line:
     Compile / PB.protoSources := Seq(file("proto/src/main/protobuf"))
   )

val copyIndexTs = taskKey[Unit]("Copy ts types file to target directory")

val copyPackageNoExports = taskKey[Unit]("Copy package file without exports to target directory")

val copyPackageNoExportsAgain =
   taskKey[Unit]("Copy package file without exports to target directory")

val copyPackageFull = taskKey[Unit]("Copy full package file")

//Enabling NativePlatform here requires custom-built spire
lazy val core = (projectMatrix in file("core"))
   .jvmPlatform(scalaVersions = Seq(dottyVersion))
   .jsPlatform(scalaVersions = Seq(dottyVersion), scalaJSStage in Global := FullOptStage)
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
             ("org.typelevel" %%% "spire"       % spireVersion),
             ("org.typelevel" %%% "cats-effect" % "3.7.0-RC1"),
             ("org.typelevel" %%% "cats-mtl"    % "1.6.0"),
             ("org.typelevel" %%% "algebra"     % "2.13.0")
           )
     })
   )

lazy val backends = (projectMatrix in file("backends"))
//TODO: restore  //.enablePlugins(ScalablyTypedConverterExternalNpmPlugin)
   .jvmPlatform(scalaVersions = Seq(dottyVersion))
   // axisValues = Seq(config12, VirtualAxis.jvm),Seq())
   .jsPlatform(
     scalaVersions = Seq(dottyVersion),
     Seq(
       scalaJSUseMainModuleInitializer := true,
       scalaJSStage in Global          := FullOptStage,
//              moduleName := name.value + "_js",
       mainClass := Some(
         "org.emergentorder.onnx.backends.Main"
       ), // "livechart.LiveChart"), // , //Testing
       scalaJSLinkerConfig ~= {
          _.withModuleKind(ModuleKind.ESModule)
             .withExperimentalUseWebAssembly(
               true
             ) // wasm works in node, breaks in browser (even when enabled there)
             //       .withModuleSplitStyle(ModuleSplitStyle.FewestModules)
//          .withModuleSplitStyle(ModuleSplitStyle.SmallModulesFor(List("backendsJS")))
             .withESFeatures(_.withESVersion(org.scalajs.linker.interface.ESVersion.ES2021))
       },
       // Configure Node.js (at least v23) to support the required Wasm features
       jsEnv := {
          val config = org.scalajs.jsenv.nodejs.NodeJSEnv
             .Config()
             .withArgs(
               List(
                 "--experimental-wasm-exnref",           // required
                 "--experimental-wasm-imported-strings", // optional (good for performance)
//                 "--turboshaft-wasm", // optional, but significantly increases stability
//         "--version",
                 "--import=extensionless/register"
//         "--experimental-specifier-resolution=node" //TODO: Replace to fix build in recent Node versions
               )
             )
          new org.scalajs.jsenv.nodejs.NodeJSEnv(config)
       },
       libraryDependencies += "org.scala-js" %%% "scalajs-dom"                 % "2.8.1",
       libraryDependencies += "org.scala-js" %%% "scala-js-macrotask-executor" % "1.1.1",
//     Compile / npmDependencies += "onnxruntime-web" -> "1.21.1",
       // ORT web and node are interchangeable, given minor package name changes, and node offers a significant speed-up (at the cost of working on the web)
//     Compile / npmDependencies += "onnxruntime-node"   -> "1.21.1",
//     Compile / npmDependencies += "onnxruntime-common" -> "1.21.1",
//     Compile / npmDependencies += "tsc-alias" -> "1.8.15",
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
//     Compile / sources := Nil,
       doc / sources := Nil,
//     Compile / stMinimize := Selection.AllExcept("onnxruntime-common", "onnxruntime-web", "onnxruntime-node"),
//    Compile / npmDependencies += "typescript"         -> "5.5.4",
       copyIndexTs := {
          import Path._

          val src = new File(".")

          // get the files we want to copy
          val htmlFiles: Seq[File] = Seq(new File("./index.d.ts"))

          // use Path.rebase to pair source files with target destination in crossTarget
          val pairs = Seq(
            (
              htmlFiles(0),
              new File(".sbt/matrix/backendsJS3/node_modules/onnxruntime-node/dist/index.d.ts")
            )
          ) // pair rebase(

          // Copy files to source files to target
          IO.copy(
            pairs,
            CopyOptions
               .apply(overwrite = true, preserveLastModified = true, preserveExecutable = false)
          )

       },
       copyPackageFull := {
          import Path._

          val src = new File(".")

          // get the files we want to copy
          val htmlFiles: Seq[File] = Seq(new File("./package.json"))

          // use Path.rebase to pair source files with target destination in crossTarget
          val pairs: Seq[(File, File)] =
             Seq((htmlFiles(0), new File(".sbt/matrix/backendsJS3/package.json")))

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
          val htmlFiles: Seq[File] = Seq(new File("packageNoExports.json"))

          // use Path.rebase to pair source files with target destination in crossTarget
          val pairs: Seq[(File, File)] = Seq(
            (
              htmlFiles(0),
              new File(".sbt/matrix/backendsJS3/node_modules/onnxruntime-common/package.json")
            )
          )

          // Copy files to source files to target
          IO.copy(
            pairs,
            CopyOptions
               .apply(overwrite = true, preserveLastModified = true, preserveExecutable = false)
          )
       },
       copyPackageNoExportsAgain := {
          import Path._

          val src = new File(".")

          // get the files we want to copy
          val htmlFiles: Seq[File] = Seq(new File("packageNoExports.json"))

          // use Path.rebase to pair source files with target destination in crossTarget
          val pairs: Seq[(File, File)] =
             Seq((htmlFiles(0), new File("node_modules/onnxruntime-common/package.json")))

          // Copy files to source files to target
          IO.copy(
            pairs,
            CopyOptions
               .apply(overwrite = true, preserveLastModified = true, preserveExecutable = false)
          )
       },
       Compile / compile := (Compile / compile dependsOn (copyPackageFull)).value,
       Test / test       := (Test / test dependsOn (
         copyIndexTs,
         copyPackageNoExports,
         copyPackageNoExportsAgain
       )).value,
//     Clean / clean := (Clean / clean dependsOn (copyIndexTs, copyPackageNoExports)).value,
       stOutputPackage                        := "org.emergentorder.onnx",
       stShortModuleNames                     := true,
       Compile / packageDoc / publishArtifact := true,
       externalNpm                            := {
          Process("npm install --ignore-scripts", baseDirectory.value).!
          Process("npm install --ignore-scripts", baseDirectory.value / "../../..").!
//      Process("npm run dev", baseDirectory.value).!
          baseDirectory.value
       }
//     scalaJSLinkerConfig ~= { _.withESFeatures(_.withESVersion(scala.scalajs.LinkingInfo.ESVersion.ES2021)) }
     )
   )
   .dependsOn(core)
   .settings(
     commonSettings,
     scalacOptions ++= Seq("-Werror"), // , "-language:future"),
     name := "onnx-scala-backends",
     libraryDependencies ++= Seq(
       "org.typelevel"           %%% "cats-effect-testing-scalatest" % "1.7.0" % Test,
       "com.microsoft.onnxruntime" % "onnxruntime"                   % "1.23.2", // "1.23.0-RC2",
       "com.microsoft.onnxruntime" % "onnxruntime-extensions"        % "0.13.0"
     ),
     libraryDependencies += ("org.scalatest" %%% "scalatest" % scalaTestVersion) % Test
//     libraryDependencies += ("org.scalactic" %% "scalactic" % scalaTestVersion),
   )

lazy val backendsFound = backends
   .js(dottyVersion)
   .enablePlugins(ScalablyTypedConverterExternalNpmPlugin)

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
