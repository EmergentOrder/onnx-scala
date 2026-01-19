addDependencyTreePlugin
addSbtPlugin("com.eed3si9n" % "sbt-projectmatrix" % "0.11.0")
addSbtPlugin("ch.epfl.scala"      % "sbt-version-policy"            % "3.2.1")
addSbtPlugin("org.scala-js"       % "sbt-scalajs"                   % "1.20.2")
//addSbtPlugin("org.scala-native"   % "sbt-scala-native"              % "0.5.9")
//addSbtPlugin("com.github.sbt"     % "sbt-native-packager"           % "1.11.7")
addSbtPlugin("ch.epfl.scala"      % "sbt-scalafix"                  % "0.14.5")
addSbtPlugin("org.scalameta"      % "sbt-scalafmt"                  % "2.5.6")
//addSbtPlugin("org.xerial.sbt"     % "sbt-sonatype"                  % "3.12.2")
//addSbtPlugin("com.github.sbt"     % "sbt-pgp"                       % "2.3.1")
addSbtPlugin("org.scalameta"      % "sbt-mdoc"                      % "2.8.2")
//addSbtPlugin("ch.epfl.scala" % "sbt-scalajs-bundler" % "0.21.1")
// for Scala.js 1.x.x
addSbtPlugin("org.scalablytyped.converter" % "sbt-converter" % "1.0.0-beta44") //45 not yet published, requires custom build
