// Command-line harness: swift run -c release mbbench <model.mlpackage|.mlmodelc> <feeds dir> [compute...]
import Foundation
import MotionBricksKit

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("usage: mbbench <model.mlpackage|.mlmodelc> <feeds dir> [ne|gpu|cpu|all ...]")
    exit(2)
}
let names: [String: ComputeChoice] = ["ne": .cpuAndNeuralEngine, "gpu": .cpuAndGPU, "cpu": .cpuOnly, "all": .all]
let choices = args.count > 3 ? args[3...].compactMap { names[$0] } : [.cpuAndNeuralEngine, .cpuAndGPU]
let feeds = try FeedSet(directory: URL(fileURLWithPath: args[2]))
print("\(feeds.samples.count) recorded inputs; model \(args[1])")
for c in choices {
    let r = try await Benchmark.run(modelURL: URL(fileURLWithPath: args[1]), feeds: feeds, compute: c)
    print(r.summary)
    print("  placement: \(r.placement.map(String.init(describing:)) ?? "unavailable")")
}
