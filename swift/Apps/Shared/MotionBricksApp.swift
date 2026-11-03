import MotionBricksKit
import SwiftUI

@main
struct MotionBricksApp: App {
    var body: some Scene {
        WindowGroup {
            DemoView(modelURL: Locations.model, feedsURL: Locations.feeds,
                     autorun: ProcessInfo.processInfo.environment["MB_AUTORUN"] == "1")
        }
    }
}

/// Where the app looks for the model and the exported feeds. Only this differs between the two apps.
enum Locations {
    #if os(macOS)
    /// Overridable with MB_MODEL / MB_FEEDS; otherwise the repo this source file was built from.
    static var model: URL? {
        first(env: "MB_MODEL", ["build/coreml/soma_b1.mlpackage", "swift/Models/soma_b1.mlpackage"])
    }
    static var feeds: URL? { first(env: "MB_FEEDS", ["swift/Feeds", "build/swift_feeds"]) }

    private static let repo = URL(fileURLWithPath: #filePath)  // swift/Apps/Shared/MotionBricksApp.swift
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    private static func first(env: String, _ relative: [String]) -> URL? {
        let candidates = [ProcessInfo.processInfo.environment[env].map(URL.init(fileURLWithPath:))]
            + relative.map { repo.appendingPathComponent($0) }
        return candidates.compactMap { $0 }.first { FileManager.default.fileExists(atPath: $0.path) }
    }
    #else
    /// Bundled by scripts/stage_ios_resources.sh; Xcode compiles the .mlpackage to .mlmodelc at build time.
    static var model: URL? { Bundle.main.url(forResource: "soma_b1", withExtension: "mlmodelc") }
    static var feeds: URL? { Bundle.main.url(forResource: "Feeds", withExtension: nil) }
    #endif
}
