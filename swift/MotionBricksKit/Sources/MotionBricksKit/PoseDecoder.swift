import Foundation
import simd

/// The SOMA skeleton, as written by tools/export_swift_rig.py.
public struct Rig: Sendable, Decodable {
    public struct Transform: Sendable, Decodable {
        var p: [Float]
        var q: [Float]  // x, y, z, w (Warp order)
        public var position: SIMD3<Float> { SIMD3(p[0], p[1], p[2]) }
        public var rotation: simd_quatf { simd_quatf(ix: q[0], iy: q[1], iz: q[2], r: q[3]) }
    }

    public var jointNames: [String]
    public var parents: [Int]
    /// Model joint feeding each pose joint, or -1 to keep the reference local transform.
    public var poseToModel: [Int]
    public var numModelJoints: Int
    public var referenceLocal: [Transform]
    /// SOMA model space (Y up) to NVIDIA's runtime space (Z up, MuJoCo facing).
    public var modelToRuntime: Transform
    public var boundaryFrames: Int
    public var posesPerToken: Int
    public var sampleRate: Double

    public var jointCount: Int { parents.count }

    enum CodingKeys: String, CodingKey {
        case jointNames = "joint_names", parents, poseToModel = "pose_to_model", numModelJoints = "num_model_joints"
        case referenceLocal = "reference_local", modelToRuntime = "model_to_runtime"
        case boundaryFrames = "boundary_frames", posesPerToken = "poses_per_token", sampleRate = "sample_rate"
    }

    public init(url: URL) throws {
        self = try JSONDecoder().decode(Rig.self, from: Data(contentsOf: url))
    }
}

/// Joint world positions for each valid frame of one generated clip.
public struct DecodedClip: Sendable {
    public var frames: [[SIMD3<Float>]]  // [frame][joint]
}

/// Swift port of NVIDIA's decode_motionbricks_output_kernel + compute_world_poses
/// (third_party/motionbricks/motionbricks/model/soma/kernels.py, utils/pose_utils.py).
public struct PoseDecoder: Sendable {
    public let rig: Rig
    public init(rig: Rig) { self.rig = rig }

    /// `inRuntimeSpace` applies model_to_runtime (Z up) as NVIDIA's engine does; false keeps SOMA model space (Y up).
    /// Alignment is identity: a lone clip has no context pose to align to. Endpoint deltas are not applied.
    public func decode(motion: [Float], numTokens: Int, inRuntimeSpace: Bool = false) -> DecodedClip {
        let channels = SOMAOutput.channels, maxFrames = SOMAOutput.frames
        let raw = min(max(numTokens * rig.posesPerToken, 0), maxFrames)
        let count = max(0, raw - 2 * rig.boundaryFrames)
        // Layout: global root [5], child joint positions [3*(J-1)], global 6D rotations [6*J], velocities, contacts.
        let rotationOffset = 5 + (rig.numModelJoints - 1) * 3
        let rootTx = inRuntimeSpace ? rig.modelToRuntime : nil
        var frames: [[SIMD3<Float>]] = []
        frames.reserveCapacity(count)
        for f in 0..<count {
            let base = (rig.boundaryFrames + f) * channels
            func rotation(_ modelJoint: Int) -> simd_quatf {
                let o = base + rotationOffset + modelJoint * 6
                let xRaw = SIMD3(motion[o], motion[o + 1], motion[o + 2])
                let yRaw = SIMD3(motion[o + 3], motion[o + 4], motion[o + 5])
                let x = simd_normalize(xRaw)
                let z = simd_normalize(simd_cross(x, yRaw))
                let y = simd_cross(z, x)
                return simd_quatf(simd_float3x3(columns: (x, y, z)))
            }
            var worldRot = [simd_quatf](repeating: simd_quatf(), count: rig.jointCount)
            var worldPos = [SIMD3<Float>](repeating: .zero, count: rig.jointCount)
            for j in 0..<rig.jointCount {
                let localPos: SIMD3<Float>, localRot: simd_quatf
                if j == 0 {
                    localPos = SIMD3(motion[base], motion[base + 1], motion[base + 2])
                    localRot = rotation(0)
                } else if rig.poseToModel[j] < 0 {
                    localPos = rig.referenceLocal[j].position
                    localRot = rig.referenceLocal[j].rotation
                } else {
                    let parentRot = rotation(rig.poseToModel[rig.parents[j]])
                    localPos = rig.referenceLocal[j].position
                    localRot = simd_normalize(parentRot.inverse * rotation(rig.poseToModel[j]))
                }
                if j == 0 {
                    if let t = rootTx {
                        worldRot[0] = t.rotation * localRot
                        worldPos[0] = t.position + t.rotation.act(localPos)
                    } else {
                        worldRot[0] = localRot
                        worldPos[0] = localPos
                    }
                } else {
                    let p = rig.parents[j]
                    worldRot[j] = worldRot[p] * localRot
                    worldPos[j] = worldPos[p] + worldRot[p].act(localPos)
                }
            }
            frames.append(worldPos)
        }
        return DecodedClip(frames: frames)
    }
}

/// Joint position error of the Swift decode against NVIDIA's Python decode (metres).
public struct PoseParity: Sendable {
    public var clips = 0, frames = 0
    public var meanError: Double = 0, maxError: Double = 0
    public var summary: String {
        String(format: "pose decode vs Python: %d clips, %d frames, mean %.2e m, max %.2e m",
               clips, frames, meanError, maxError)
    }

    /// Decodes each parity sample's fp32 reference motion and compares with the exported Python positions.
    public static func check(feeds: FeedSet, decoder: PoseDecoder) -> PoseParity? {
        guard !feeds.parity.isEmpty else { return nil }
        var r = PoseParity(), sum = 0.0, n = 0
        for (sample, expected) in feeds.parity {
            let s = feeds.samples[sample]
            let clip = decoder.decode(motion: s.referenceMotion, numTokens: s.referenceTokens, inRuntimeSpace: true)
            r.clips += 1
            for (f, joints) in clip.frames.enumerated() where f < expected.count {
                r.frames += 1
                for (j, p) in joints.enumerated() {
                    let e = Double(simd_distance(p, expected[f][j]))
                    sum += e; n += 1
                    r.maxError = max(r.maxError, e)
                }
            }
            if clip.frames.count != expected.count { r.maxError = .infinity }  // frame count must match too
        }
        r.meanError = n > 0 ? sum / Double(n) : 0
        return r
    }
}
