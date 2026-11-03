import SwiftUI
#if os(macOS)
import AppKit
#endif

/// The shared benchmark screen: pick compute units, replay the recorded feeds, show accuracy, timing and placement.
public struct BenchmarkView: View {
    @State private var modelURL: URL?
    @State private var feedsURL: URL?
    @State private var compute = ComputeChoice.cpuAndNeuralEngine
    @State private var running = false
    @State private var progress = (done: 0, total: 0)
    @State private var results: [BenchmarkResult] = []
    @State private var error: String?

    private let autorun: Bool

    /// `autorun` starts a run on appear and prints the summary, for scripted checks of the real app.
    public init(modelURL: URL?, feedsURL: URL?, autorun: Bool = false) {
        self.autorun = autorun
        _modelURL = State(initialValue: modelURL)
        _feedsURL = State(initialValue: feedsURL)
    }

    public var body: some View {
        Form {
            Section("Inputs") {
                pathRow("Model", modelURL) { modelURL = $0 }
                pathRow("Feeds", feedsURL) { feedsURL = $0 }
                Picker("Compute units", selection: $compute) {
                    ForEach(ComputeChoice.allCases) { Text($0.rawValue).tag($0) }
                }
                Button(running ? "Running \(progress.done)/\(progress.total)…" : "Run 84 recorded inputs") {
                    Task { await run() }
                }
                .disabled(running || modelURL == nil || feedsURL == nil)
            }
            if let error { Section("Error") { Text(error).foregroundStyle(.red).textSelection(.enabled) } }
            ForEach(Array(results.enumerated().reversed()), id: \.offset) { _, r in
                Section(r.compute.rawValue) {
                    LabeledContent("Accuracy", value: String(format: "mean |Δ| %.4f", r.meanAbsDiff))
                    LabeledContent("Same clip length", value: "\(r.sameLength)/\(r.samples)")
                    LabeledContent("Per call", value: String(format: "%.2f ms median, %.2f ms p90", r.medianMs, r.p90Ms))
                    LabeledContent("Load + compile", value: String(format: "%.1f s", r.loadSeconds))
                    LabeledContent("Placement", value: r.placement.map(String.init(describing:)) ?? "unavailable")
                    Text(r.ok ? "OK" : "MISMATCH").bold().foregroundStyle(r.ok ? .green : .orange)
                }
            }
        }
        .formStyle(.grouped)
        .task { if autorun { await run() } }
        .navigationTitle("MotionBricks SOMA")
    }

    private func pathRow(_ label: String, _ url: URL?, set: @escaping (URL) -> Void) -> some View {
        LabeledContent(label) {
            HStack {
                Text(url?.lastPathComponent ?? "not found").foregroundStyle(url == nil ? .red : .secondary)
                #if os(macOS)
                Button("Choose…") {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true  // .mlpackage, .mlmodelc and the feeds folder are all directories
                    panel.treatsFilePackagesAsDirectories = false
                    if panel.runModal() == .OK, let u = panel.url { set(u) }
                }
                #endif
            }
        }
    }

    private func run() async {
        guard let modelURL, let feedsURL else { return }
        running = true
        error = nil
        defer { running = false }
        do {
            let feeds = try FeedSet(directory: feedsURL)
            progress = (0, feeds.samples.count)
            let choice = compute
            // Run off the main actor so the UI keeps drawing; progress hops back to it.
            let r = try await Task.detached {
                try await Benchmark.run(modelURL: modelURL, feeds: feeds, compute: choice) { done, total in
                    Task { @MainActor in progress = (done, total) }
                }
            }.value
            results.append(r)
            print(r.summary + "\n  placement: " + (r.placement.map(String.init(describing:)) ?? "unavailable"))
        } catch {
            self.error = String(describing: error)
            print("error: \(error)")
        }
    }
}
