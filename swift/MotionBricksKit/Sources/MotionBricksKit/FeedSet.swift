import Foundation

/// Recorded network inputs plus fp32 reference outputs, as written by tools/export_swift_feeds.py.
public struct FeedSet: Sendable {
    public struct Sample: Sendable {
        public var input: SOMAInput
        public var referenceMotion: [Float]  // [64, 366]
        public var referenceTokens: Int
    }

    public var samples: [Sample]

    private struct Manifest: Decodable {
        struct Tensor: Decodable { var file: String; var dtype: String; var shape: [Int] }
        var samples: [[String: Tensor]]
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
    }
}
