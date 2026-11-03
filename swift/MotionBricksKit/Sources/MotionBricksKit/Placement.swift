import CoreML
import Foundation

/// How many of the model's operations CoreML prefers to run on each device, from MLComputePlan.
public struct Placement: Sendable, CustomStringConvertible {
    public var neuralEngine = 0, gpu = 0, cpu = 0
    /// Share of estimated compute cost placed on the Neural Engine (0...1), when CoreML reports costs.
    public var neuralEngineCostShare: Double?

    public var total: Int { neuralEngine + gpu + cpu }
    public var description: String {
        var s = "\(neuralEngine)/\(total) ops on Neural Engine, \(gpu) GPU, \(cpu) CPU"
        if let share = neuralEngineCostShare { s += String(format: " (%.1f%% of est. cost on NE)", share * 100) }
        return s
    }

    /// Ops-level device preference from MLComputePlan (macOS 14.4+ / iOS 17.4+, covered by the package minimum).
    public static func load(compiledURL: URL, compute: ComputeChoice) async throws -> Placement? {
        let config = MLModelConfiguration()
        config.computeUnits = compute.units
        let plan = try await MLComputePlan.load(contentsOf: compiledURL, configuration: config)
        guard case .program(let program) = plan.modelStructure,
              let main = program.functions["main"] else { return nil }
        var p = Placement(), neCost = 0.0, allCost = 0.0
        func visit(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                op.blocks.forEach(visit)
                guard let usage = plan.deviceUsage(for: op) else { continue }  // consts etc. have no device
                let cost = plan.estimatedCost(of: op)?.weight ?? 0
                allCost += cost
                switch usage.preferred {
                case .neuralEngine: p.neuralEngine += 1; neCost += cost
                case .gpu: p.gpu += 1
                default: p.cpu += 1
                }
            }
        }
        visit(main.block)
        if allCost > 0 { p.neuralEngineCostShare = neCost / allCost }
        return p
    }
}
