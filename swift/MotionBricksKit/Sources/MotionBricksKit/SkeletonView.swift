import Observation
import RealityKit
import SwiftUI
import simd

/// Clips to play: the decoded fp32 reference outputs, plus any generated live on the device.
@MainActor @Observable
final class SkeletonPlayer {
    struct Clip: Identifiable {
        let id = UUID()
        var name: String
        var frames: [[SIMD3<Float>]]
    }

    var clips: [Clip] = []
    var selected: UUID?
    var frame = 0
    var playing = true
    var status = "Loading…"
    var generating = false
    var sampleForGeneration = 0
    private(set) var feeds: FeedSet?
    private var decoder: PoseDecoder?
    private var model: SOMAModel?
    private var timer: Timer?

    var current: Clip? { clips.first { $0.id == selected } ?? clips.first }
    var parents: [Int] { decoder?.rig.parents ?? [] }

    func load(feedsURL: URL?) async {
        guard feeds == nil else { return }
        guard let feedsURL else { status = "Feeds folder not found"; return }
        do {
            let feeds = try await Task.detached { try FeedSet(directory: feedsURL) }.value
            guard let rig = feeds.rig else { status = "No rig.json: run tools/export_swift_rig.py"; return }
            let decoder = PoseDecoder(rig: rig)
            self.feeds = feeds
            self.decoder = decoder
            clips = feeds.samples.enumerated().map { i, s in
                Clip(name: "Recorded #\(i) (fp32 reference)",
                     frames: Self.grounded(decoder.decode(motion: s.referenceMotion, numTokens: s.referenceTokens).frames))
            }
            selected = clips.first?.id
            status = "\(clips.count) clips, \(rig.jointCount) joints"
            startTimer()
        } catch {
            status = String(describing: error)
        }
    }

    /// Runs the network on recorded input `sampleForGeneration`, decodes it and plays it.
    func generate(modelURL: URL?) async {
        guard let modelURL, let feeds, let decoder else { status = "Model not found"; return }
        generating = true
        defer { generating = false }
        do {
            if model == nil {
                status = "Loading model on the Neural Engine…"
                model = try await SOMAModel(url: modelURL, compute: .cpuAndNeuralEngine)
                _ = try model!.predict(feeds.samples[0].input)  // first call pays one-off setup; keep it out of the label
            }
            let i = sampleForGeneration
            let (out, sec) = try model!.predict(feeds.samples[i].input)
            let frames = Self.grounded(decoder.decode(motion: out.motion, numTokens: out.numTokens).frames)
            clips.append(Clip(name: String(format: "Live #%d — %.1f ms on %@", i, sec * 1000, "Neural Engine"), frames: frames))
            selected = clips.last?.id
            frame = 0
            playing = true
            status = String(format: "Generated %d frames in %.1f ms", frames.count, sec * 1000)
        } catch {
            status = String(describing: error)
        }
    }

    private func startTimer() {
        timer?.invalidate()
        let rate = decoder?.rig.sampleRate ?? 30
        timer = Timer.scheduledTimer(withTimeInterval: 1 / rate, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.playing, let c = self.current, !c.frames.isEmpty else { return }
                self.frame = (self.frame + 1) % c.frames.count
            }
        }
    }

    /// Moves a clip so its first root position sits above the origin, keeping heights.
    static func grounded(_ frames: [[SIMD3<Float>]]) -> [[SIMD3<Float>]] {
        guard let root = frames.first?.first else { return frames }
        let shift = SIMD3<Float>(root.x, 0, root.z)
        return frames.map { $0.map { $0 - shift } }
    }
}

/// The main demo screen: a skeleton (joint spheres, bone boxes) playing generated clips at 30 Hz.
public struct SkeletonView: View {
    let modelURL: URL?
    let feedsURL: URL?
    let autorun: Bool
    @State private var player = SkeletonPlayer()

    /// `autorun` generates a clip from input #0 on the Neural Engine as soon as the feeds load.
    public init(modelURL: URL?, feedsURL: URL?, autorun: Bool = false) {
        self.autorun = autorun
        self.modelURL = modelURL
        self.feedsURL = feedsURL
    }

    public var body: some View {
        VStack(spacing: 0) {
            SkeletonScene(player: player)
            controls.padding()
        }
        .task {
            await player.load(feedsURL: feedsURL)
            if autorun { await player.generate(modelURL: modelURL) }
        }
        .navigationTitle("MotionBricks SOMA")
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button(player.playing ? "Pause" : "Play") { player.playing.toggle() }
                    .keyboardShortcut(.space, modifiers: [])
                Picker("Clip", selection: $player.selected) {
                    ForEach(player.clips) { Text($0.name).tag(Optional($0.id)) }
                }
                .onChange(of: player.selected) { player.frame = 0 }
            }
            HStack {
                Stepper("Input #\(player.sampleForGeneration)", value: $player.sampleForGeneration,
                        in: 0...max(0, (player.feeds?.samples.count ?? 1) - 1))
                    .fixedSize()
                Button(player.generating ? "Generating…" : "Generate on Neural Engine") {
                    Task { await player.generate(modelURL: modelURL) }
                }
                .buttonStyle(.borderedProminent)
                .disabled(player.generating || player.feeds == nil)
            }
            Text("\(player.status) · frame \(player.frame + 1)/\(player.current?.frames.count ?? 0)")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// RealityKit scene: ground plane, camera, one sphere per joint and one box per bone, posed from `player`.
private struct SkeletonScene: View {
    let player: SkeletonPlayer
    // Simple orbit camera around the hips: drag to orbit, pinch/scroll-zoom to change distance.
    @State private var yaw: Float = 0.6
    @State private var pitch: Float = 0.25
    @State private var distance: Float = 2.6
    @State private var dragStart: (yaw: Float, pitch: Float)?
    @State private var zoomStart: Float?

    var body: some View {
        RealityView { content in
            let root = Entity()
            root.name = "skeleton"
            content.add(root)
            let ground = ModelEntity(mesh: .generatePlane(width: 3, depth: 3),
                                     materials: [SimpleMaterial(color: .init(white: 0.55, alpha: 1), isMetallic: false)])
            content.add(ground)
            let grid = Entity()
            for k in -3...3 {
                let x = Float(k) * 0.5
                for (w, d, pos) in [(0.005, 3.0, SIMD3<Float>(x, 0.001, 0)), (3.0, 0.005, SIMD3<Float>(0, 0.001, x))] {
                    let line = ModelEntity(mesh: .generatePlane(width: Float(w), depth: Float(d)),
                                           materials: [UnlitMaterial(color: .init(white: 0.4, alpha: 1))])
                    line.position = pos
                    grid.addChild(line)
                }
            }
            content.add(grid)
            let light = DirectionalLight()
            light.light.intensity = 3000
            light.look(at: .zero, from: [2, 4, 3], relativeTo: nil)
            content.add(light)
            let camera = PerspectiveCamera()
            camera.name = "camera"
            content.add(camera)
        } update: { content in
            if let camera = content.entities.first(where: { $0.name == "camera" }) {
                let target = SIMD3<Float>(0, 0.9, 0)
                let eye = target + distance * SIMD3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
                camera.look(at: target, from: eye, relativeTo: nil)
            }
            guard let root = content.entities.first(where: { $0.name == "skeleton" }),
                  let clip = player.current, !clip.frames.isEmpty else { return }
            let joints = clip.frames[min(player.frame, clip.frames.count - 1)]
            let parents = player.parents
            if root.children.isEmpty { Self.build(root, jointCount: joints.count, parents: parents) }
            for (j, p) in joints.enumerated() {
                root.children[j].position = p
                let parent = parents[j]
                guard parent >= 0 else { continue }
                let bone = root.children[joints.count + j]
                let a = joints[parent], d = p - a, len = simd_length(d)
                bone.position = a + d / 2
                bone.scale = [1, 1, max(len, 1e-4)]
                if len > 1e-5 { bone.orientation = simd_quatf(from: [0, 0, 1], to: d / len) }
            }
        }
        .gesture(orbit.simultaneously(with: zoom))
    }

    var orbit: some Gesture {
        DragGesture()
            .onChanged { g in
                let start = dragStart ?? (yaw, pitch)
                dragStart = start
                yaw = start.yaw - Float(g.translation.width) * 0.01
                pitch = min(1.4, max(-0.1, start.pitch + Float(g.translation.height) * 0.01))
            }
            .onEnded { _ in dragStart = nil }
    }

    var zoom: some Gesture {
        MagnifyGesture()
            .onChanged { g in
                let start = zoomStart ?? distance
                zoomStart = start
                distance = min(8, max(0.8, start / Float(g.magnification)))
            }
            .onEnded { _ in zoomStart = nil }
    }

    /// Children 0..<J are joint spheres; J..<2J are bones (child J+j links joint j to its parent).
    static func build(_ root: Entity, jointCount: Int, parents: [Int]) {
        let jointMesh = MeshResource.generateSphere(radius: 0.018)
        let boneMesh = MeshResource.generateBox(width: 0.014, height: 0.014, depth: 1)
        let jointMat = SimpleMaterial(color: .init(red: 0.95, green: 0.55, blue: 0.1, alpha: 1), isMetallic: false)
        let boneMat = SimpleMaterial(color: .init(red: 0.2, green: 0.45, blue: 0.9, alpha: 1), isMetallic: false)
        for _ in 0..<jointCount { root.addChild(ModelEntity(mesh: jointMesh, materials: [jointMat])) }
        for j in 0..<jointCount {
            root.addChild(parents[j] >= 0 ? ModelEntity(mesh: boneMesh, materials: [boneMat]) : Entity())
        }
    }
}

/// Both apps' root: the skeleton demo first, the benchmark as a second tab.
public struct DemoView: View {
    let modelURL: URL?, feedsURL: URL?, autorun: Bool

    public init(modelURL: URL?, feedsURL: URL?, autorun: Bool = false) {
        self.modelURL = modelURL
        self.feedsURL = feedsURL
        self.autorun = autorun
    }

    public var body: some View {
        TabView {
            NavigationStack { SkeletonView(modelURL: modelURL, feedsURL: feedsURL, autorun: autorun) }
                .tabItem { Label("Skeleton", systemImage: "figure.walk") }
            NavigationStack { BenchmarkView(modelURL: modelURL, feedsURL: feedsURL, autorun: autorun) }
                .tabItem { Label("Benchmark", systemImage: "speedometer") }
        }
    }
}
