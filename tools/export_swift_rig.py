"""Export the SOMA rig and Python-decoded joint positions for the Swift skeleton view and its parity check.

    python tools/export_swift_rig.py --feeds swift/Feeds            # after tools/export_swift_feeds.py

Writes into the feeds folder:
  rig.json                  joint names, parents, pose_to_model, reference local transforms, model_to_runtime
  NNN_ref_world_positions.bin   float32 [frames, joints, 3] for the first --clips samples, decoded from the fp32
                                reference motion by NVIDIA's own Warp kernels (decode_motionbricks_output_kernel
                                + compute_world_poses), with identity alignment (no context pose to align to)
and adds "parity" to manifest.json. Endpoint-delta post-processing is skipped: it needs the runtime's context pose.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # PYTHONPATH may point at another checkout
import numpy as np  # noqa: E402

from cmb import _compat  # noqa: E402,F401
from cmb.engine import MacSOMAEngine  # noqa: E402
import warp as wp  # noqa: E402
from motionbricks.io.usd import load_skeleton_from_usd  # noqa: E402
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file  # noqa: E402
from motionbricks.utils.pose_utils import compute_world_poses  # noqa: E402
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", type=Path, required=True)
    ap.add_argument("--clips", type=int, default=8)
    args = ap.parse_args()
    wp.init()
    wp.set_device("cpu")
    skeleton = load_skeleton_from_usd(get_asset_file("human", "soma_base_skel_minimal_no_root.usd"),
                                      SOMA_SKELETON_PRIM_PATH)
    to_runtime = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
    engine = MacSOMAEngine(backend=lambda feeds: None, model_to_runtime=to_runtime)
    output = engine.create_output(skeleton)  # binds the skeleton: pose_to_model etc.
    engine._alignment.assign(np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32))

    ref_local = skeleton.reference_local_transforms.numpy()
    rig = {
        "joint_names": [skeleton.joint_name(i) for i in range(skeleton.num_joints)],
        "parents": skeleton.parent_indices.numpy().tolist(),
        "pose_to_model": engine._pose_to_model.numpy().tolist(),
        "num_model_joints": len(engine.contract.joint_names),
        "reference_local": [{"p": t[:3].tolist(), "q": t[3:].tolist()} for t in ref_local],  # q = x, y, z, w
        "model_to_runtime": {"p": list(to_runtime.p), "q": list(to_runtime.q)},
        "boundary_frames": 3, "poses_per_token": 4, "sample_rate": 30.0,
    }
    (args.feeds / "rig.json").write_text(json.dumps(rig, indent=1))

    manifest = json.loads((args.feeds / "manifest.json").read_text())
    parity = []
    world = wp.empty(output.poses.shape, dtype=wp.transform, device="cpu")
    for i in range(min(args.clips, len(manifest["samples"]))):
        s = manifest["samples"][i]
        motion = np.fromfile(args.feeds / s["ref_pred_global_motions"]["file"], dtype="<f4").reshape(1, 64, 366)
        tokens = np.fromfile(args.feeds / s["ref_pred_num_tokens"]["file"], dtype="<i4")
        engine._buffers["pred_global_motions"].assign(motion)
        engine._buffers["pred_num_tokens"].assign(tokens.reshape(1))
        engine._decode_generated_poses(output)
        frames = int(engine._generated_frame_count.numpy()[0])
        compute_world_poses(output.poses, skeleton.parent_indices, world, num_poses=frames)
        pos = world.numpy()[:frames, :, :3].astype("<f4")
        fname = f"{i:03d}_ref_world_positions.bin"
        pos.tofile(args.feeds / fname)
        parity.append({"sample": i, "file": fname, "shape": list(pos.shape)})
    manifest["parity"] = parity
    (args.feeds / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{skeleton.num_joints} joints ({rig['num_model_joints']} model joints); {len(parity)} parity clips")


if __name__ == "__main__":
    main()
