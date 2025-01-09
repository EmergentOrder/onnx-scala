addDependencyTreePlugin
addSbtPlugin("ch.epfl.scala"      % "sbt-version-policy"            % "3.2.1")
addSbtPlugin("org.portable-scala" % "sbt-scalajs-crossproject"      % "1.3.2")
addSbtPlugin("org.portable-scala" % "sbt-scala-native-crossproject" % "1.3.2")
addSbtPlugin("org.scala-js"       % "sbt-scalajs"                   % "1.18.1")
addSbtPlugin("org.scala-native"   % "sbt-scala-native"              % "0.5.6")
addSbtPlugin("ch.epfl.scala"      % "sbt-scalafix"                  % "0.13.0")
addSbtPlugin("org.scalameta"      % "sbt-scalafmt"                  % "2.5.2")
addSbtPlugin("org.xerial.sbt"     % "sbt-sonatype"                  % "3.12.2")
addSbtPlugin("com.github.sbt"     % "sbt-pgp"                       % "2.3.1")
addSbtPlugin("org.scalameta"      % "sbt-mdoc"                      % "2.6.2")
// for Scala.js 1.x.x
addSbtPlugin("org.scalablytyped.converter" % "sbt-converter" % "1.0.0-beta44")
