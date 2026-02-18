//Not working with sbt 2
//addDependencyTreePlugin
addSbtPlugin("com.eed3si9n" % "sbt-projectmatrix" % "0.11.0") //No longer needed in sbt 2.x
//addSbtPlugin("ch.epfl.scala" % "sbt-version-policy" % "3.2.1") //Not working with sbt 2.x
//addSbtPlugin("org.scala-js"  % "sbt-scalajs"        % "1.20.2") //Not working with sbt 2.x. PR is open
//addSbtPlugin("org.scala-native"   % "sbt-scala-native"              % "0.5.10") //Not working with sbt 2.x. PR has landed.
//addSbtPlugin("org.scalablytyped.converter" % "sbt-converter" % "1.0.0-beta45")
//addSbtPlugin("org.xerial.sbt"     % "sbt-sonatype"                  % "3.12.2") //Deprecated, use sbt built-in instead

//Working with sbt 2, but not needed
//addSbtPlugin("com.github.sbt"     % "sbt-native-packager"           % "1.11.7")
//addSbtPlugin("com.github.sbt"     % "sbt-pgp"                       % "2.3.1")
//addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.8.2")
//addSbtPlugin("pl.project13.scala" % "sbt-jmh" % "0.4.8")

//working with sbt 2
addSbtPlugin("com.typesafe"  % "sbt-mima-plugin" % "1.1.5")
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix"    % "0.14.5")
addSbtPlugin("org.scalameta" % "sbt-scalafmt"    % "2.5.6")
