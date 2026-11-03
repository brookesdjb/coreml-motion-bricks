import Foundation

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
    public var ok: Bool { meanAbsDiff < 0.02 && Double(sameLength) >= 0.95 * Double(samples) }

    public var summary: String {
        String(format: "%@: mean|Δ| %.4f, same length %d/%d, median %.2f ms/call (p90 %.2f), load %.1f s — %@",
               compute.rawValue, meanAbsDiff, sameLength, samples, medianMs, p90Ms, loadSeconds,
               ok ? "OK" : "MISMATCH")
    }
}

public enum Benchmark {
    /// Loads the model with `compute`, replays every feed, and compares with the references.
    public static func run(modelURL: URL, feeds: FeedSet, compute: ComputeChoice,
                           progress: (@Sendable (Int, Int) -> Void)? = nil) async throws -> BenchmarkResult {
        let model = try await SOMAModel(url: modelURL, compute: compute)
        var errs: [Double] = [], same = 0, times: [Double] = []
        for (i, s) in feeds.samples.enumerated() {
            let (out, sec) = try model.predict(s.input)
            times.append(sec * 1000)
            if out.numTokens == s.referenceTokens { same += 1 }
            let n = min(SOMAOutput.frames, 4 * s.referenceTokens) * SOMAOutput.channels
            var sum = 0.0
            for j in 0..<n { sum += Double(abs(out.motion[j] - s.referenceMotion[j])) }
            errs.append(n > 0 ? sum / Double(n) : 0)
            progress?(i + 1, feeds.samples.count)
        }
        let sorted = times.dropFirst().sorted()
        func pct(_ q: Double) -> Double { sorted.isEmpty ? times.first ?? 0 : sorted[Int(Double(sorted.count - 1) * q)] }
        let placement = try? await Placement.load(compiledURL: model.compiledURL, compute: compute)
        return BenchmarkResult(compute: compute, samples: errs.count,
                               meanAbsDiff: errs.reduce(0, +) / Double(max(errs.count, 1)), sameLength: same,
                               medianMs: pct(0.5), p90Ms: pct(0.9), loadSeconds: model.loadSeconds,
                               placement: placement)
    }
}
