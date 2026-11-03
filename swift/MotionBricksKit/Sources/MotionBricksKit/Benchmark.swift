import Foundation
import simd

/// Accuracy and timing over the recorded feeds, matching tools/check_accuracy.py.
public struct BenchmarkResult: Sendable {
    public var compute: ComputeChoice
    public var samples: Int
    public var meanAbsDiff: Double       // mean over samples of mean |Δ| on valid frames
    public var sameLength: Int           // samples whose token count matches the reference
    public var medianMs: Double          // excludes the first (warm-up) call
    public var p90Ms: Double
    public var loadSeconds: Double
    public var placement: Placement?
    /// Joint position error of this model's decoded output against the Python decode of the fp32 reference (m).
    public var poseMeanError: Double?
    public var poseMaxError: Double?
    public var ok: Bool { meanAbsDiff < 0.02 && Double(sameLength) >= 0.95 * Double(samples) }

    public var summary: String {
        String(format: "%@: mean|Δ| %.4f, same length %d/%d, median %.2f ms/call (p90 %.2f), load %.1f s — %@",
               compute.rawValue, meanAbsDiff, sameLength, samples, medianMs, p90Ms, loadSeconds,
               ok ? "OK" : "MISMATCH")
    }

    public var poseSummary: String? {
        guard let m = poseMeanError, let x = poseMaxError else { return nil }
        return String(format: "decoded joints vs Python fp32 decode: mean %.1f mm, max %.1f mm", m * 1000, x * 1000)
    }
}

public enum Benchmark {
    /// Loads the model with `compute`, replays every feed, and compares with the references.
    public static func run(modelURL: URL, feeds: FeedSet, compute: ComputeChoice,
                           progress: (@Sendable (Int, Int) -> Void)? = nil) async throws -> BenchmarkResult {
        let model = try await SOMAModel(url: modelURL, compute: compute)
        var errs: [Double] = [], same = 0, times: [Double] = [], outputs: [Int: SOMAOutput] = [:]
        let paritySamples = Set(feeds.parity.map(\.sample))
        for (i, s) in feeds.samples.enumerated() {
            let (out, sec) = try model.predict(s.input)
            times.append(sec * 1000)
            if out.numTokens == s.referenceTokens { same += 1 }
            if paritySamples.contains(i) { outputs[i] = out }
            let n = min(SOMAOutput.frames, 4 * s.referenceTokens) * SOMAOutput.channels
            var sum = 0.0
            for j in 0..<n { sum += Double(abs(out.motion[j] - s.referenceMotion[j])) }
            errs.append(n > 0 ? sum / Double(n) : 0)
            progress?(i + 1, feeds.samples.count)
        }
        let sorted = times.dropFirst().sorted()
        func pct(_ q: Double) -> Double { sorted.isEmpty ? times.first ?? 0 : sorted[Int(Double(sorted.count - 1) * q)] }
        let placement = try? await Placement.load(compiledURL: model.compiledURL, compute: compute)
        var poseMean: Double?, poseMax: Double?
        if let rig = feeds.rig {
            let decoder = PoseDecoder(rig: rig)
            var sum = 0.0, n = 0, mx = 0.0
            for (i, expected) in feeds.parity {
                guard let out = outputs[i], out.numTokens == feeds.samples[i].referenceTokens else { continue }
                let clip = decoder.decode(motion: out.motion, numTokens: out.numTokens, inRuntimeSpace: true)
                for (f, joints) in clip.frames.enumerated() where f < expected.count {
                    for (j, p) in joints.enumerated() {
                        let e = Double(simd_distance(p, expected[f][j])); sum += e; n += 1; mx = max(mx, e)
                    }
                }
            }
            if n > 0 { poseMean = sum / Double(n); poseMax = mx }
        }
        return BenchmarkResult(compute: compute, samples: errs.count,
                               meanAbsDiff: errs.reduce(0, +) / Double(max(errs.count, 1)), sameLength: same,
                               medianMs: pct(0.5), p90Ms: pct(0.9), loadSeconds: model.loadSeconds,
                               placement: placement, poseMeanError: poseMean, poseMaxError: poseMax)
    }
}
