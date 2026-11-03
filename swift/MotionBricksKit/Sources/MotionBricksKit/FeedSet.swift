import Foundation

/// Recorded network inputs plus fp32 reference outputs, as written by tools/export_swift_feeds.py.
public struct FeedSet: Sendable {
    public struct Sample: Sendable {
        public var input: SOMAInput
        public var referenceMotion: [Float]  // [64, 366]
        public var referenceTokens: Int
    }

    public var samples: [Sample]
    /// Python-decoded joint world positions (runtime space) for some samples: (sample index, [frame][joint]).
    public var parity: [(sample: Int, positions: [[SIMD3<Float>]])] = []
    /// The SOMA rig, if tools/export_swift_rig.py has been run on this folder.
    public var rig: Rig?

    private struct Manifest: Decodable {
        struct Tensor: Decodable { var file: String; var dtype: String; var shape: [Int] }
        struct Parity: Decodable { var sample: Int; var file: String; var shape: [Int] }
        var samples: [[String: Tensor]]
        var parity: [Parity]?
    }

    public init(directory: URL) throws {
        let manifest = try JSONDecoder().decode(
            Manifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        samples = try manifest.samples.map { entry in
            func raw<T>(_ name: String, _: T.Type) throws -> [T] {
                guard let t = entry[name] else { throw SOMAError.badFeeds("missing \(name)") }
                let data = try Data(contentsOf: directory.appendingPathComponent(t.file))
                guard data.count == t.shape.reduce(1, *) * MemoryLayout<T>.size else {
                    throw SOMAError.badFeeds("\(t.file) has \(data.count) bytes for shape \(t.shape)")
                }
                return data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
            }
            return Sample(
                input: SOMAInput(
                    globalRootValues: try raw("global_root_values", Float.self),
                    hasGlobalRootValues: try raw("has_global_root_values", Int32.self),
                    localRootValues: try raw("local_root_values", Float.self),
                    hasLocalRootValues: try raw("has_local_root_values", Int32.self),
                    localPoses: try raw("local_poses", Float.self),
                    hasLocalPoses: try raw("has_local_poses", Int32.self),
                    allowedPredNumTokens: try raw("allowed_pred_num_tokens", Int32.self)),
                referenceMotion: try raw("ref_pred_global_motions", Float.self),
                referenceTokens: Int(try raw("ref_pred_num_tokens", Int32.self)[0]))
        }
        let rigURL = directory.appendingPathComponent("rig.json")
        if FileManager.default.fileExists(atPath: rigURL.path) { rig = try Rig(url: rigURL) }
        parity = try (manifest.parity ?? []).map { p in
            let data = try Data(contentsOf: directory.appendingPathComponent(p.file))
            let flat: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let (frames, joints) = (p.shape[0], p.shape[1])
            guard flat.count == frames * joints * 3 else { throw SOMAError.badFeeds("\(p.file) size") }
            return (p.sample, (0..<frames).map { f in
                (0..<joints).map { j in let o = (f * joints + j) * 3; return SIMD3(flat[o], flat[o + 1], flat[o + 2]) }
            })
        }
    }
}
