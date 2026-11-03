// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "MotionBricksKit",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "MotionBricksKit", targets: ["MotionBricksKit"]),
        .executable(name: "mbbench", targets: ["mbbench"]),
    ],
    targets: [
        .target(name: "MotionBricksKit", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "mbbench", dependencies: ["MotionBricksKit"], swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
