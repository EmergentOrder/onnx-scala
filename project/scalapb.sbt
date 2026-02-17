addSbtPlugin("com.thesamet" % "sbt-protoc" % "1.0.8")
//0.1.0-SNAPSHOT")
excludeDependencies ++= Seq(
  ExclusionRule("com.thesamet.scalapb", "protoc-bridge_3"),
  ExclusionRule("com.thesamet.scalapb", "protoc-bridge_2.13"),
  ExclusionRule("org.scala-lang.modules", "scala-collection-compat_3"),
  ExclusionRule("org.scala-lang.modules", "scala-collection-compat_2.13")
)
libraryDependencies += "com.thesamet.scalapb" %% "protoc-bridge" % "0.9.9"
libraryDependencies += ("com.thesamet.scalapb" %% "compilerplugin" % "1.0.0-alpha.3")
 //+0-670082a5+20260212-1249-SNAPSHOT")
