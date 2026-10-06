"""MJCF -> UniMate conditioning -> bounded P2 kinematics (metres, Z-up).

Source models and generated assets are deliberately external to Git. No robot
mesh is relicensed by this module. Use --mjcf with the supplied P2 asset tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Proper rotation: robot X forward, Y left, Z up -> model Z forward, X left, Y up.
ROBOT_TO_MODEL = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/robots/astro_p2.json"


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_model(path):
    model = mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
    types = np.asarray(model.jnt_type)
    if len(types) != 31 or types[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(types[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("P2 requires one leading free joint and exactly 30 hinge joints")
    if model.nq != 37 or model.nv != 36 or not np.all(model.jnt_limited[1:]):
        raise ValueError("Expected nq=37, nv=36 and finite limits on every P2 hinge")
    # The skeleton uses body origins. Reject nonzero hinge pivots rather than
    # silently build offsets that disagree with MuJoCo FK.
    if not np.allclose(model.jnt_pos[1:], 0, atol=1e-9):
        raise ValueError("Nonzero joint pivots need an explicit skeleton adapter")
    return model


def model_signature(model):
    """Digest kinematic values, independent of mesh path / deployment location."""
    h = hashlib.sha256()
    for key in ("body_parentid", "body_pos", "body_quat", "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "qpos0"):
        h.update(np.asarray(getattr(model, key)).tobytes())
    h.update(model.names)
    return h.hexdigest()


def prepare(mjcf, output, config=DEFAULT_CONFIG):
    from data_process.utils.motion_features import build_topology_cond
    from data_process.utils.skeleton import get_skeleton_diameter
    from types import SimpleNamespace

    model = load_model(mjcf)
    cfg = json.loads(Path(config).read_text())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = model.body(cfg["root_body"]).id
    # BFS with explicit virtual terminal joints: UniMate stores a parents
    # rotation on its children, so terminal DOFs otherwise disappear.
    nodes = [(root, -1, None)]
    for idx, (bid, parent, tip) in enumerate(nodes):
        if tip is not None:
            continue
        nodes.extend((int(c), idx, None) for c in range(1, model.nbody) if model.body_parentid[c] == bid)
        name = model.body(bid).name
        if name in cfg["tips"]:
            nodes.append((bid, idx, cfg["tips"][name]))
    if {x[0] for x in nodes} != set(range(1, model.nbody)):
        raise ValueError("Model has bodies outside the pelvis tree")
    names, labels, positions, body_ids, tip_offsets = [], [], [], [], []
    for bid, parent, tip in nodes:
        name = model.body(bid).name
        offset = np.zeros(3) if tip is None else np.array(tip["offset"], dtype=float)
        pos = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ offset
        names.append(name if tip is None else name + "_tip")
        label = cfg["labels"].get(name)
        if tip is not None:
            label = tip["label"]
        elif label is None:
            side, suffix = name.split("_", 1)
            label = side.title() + " " + cfg["side_labels"][suffix]
        labels.append(label)
        positions.append(ROBOT_TO_MODEL @ pos)
        body_ids.append(bid)
        tip_offsets.append(offset)
    parents = np.array([p for _, p, _ in nodes], dtype=np.int64)
    positions = np.array(positions)
    offsets = positions.copy()
    offsets[0] = 0
    offsets[1:] -= positions[parents[1:]]
    scale = 2.0 / get_skeleton_diameter(SimpleNamespace(parents=parents, offsets=offsets))
    positions -= positions[0] * np.array([1., 0., 1.])
    ground = float(positions[:, 1].min())
    positions[:, 1] -= ground
    positions *= scale
    offsets *= scale
    quats = np.tile([1., 0., 0., 0.], (len(nodes), 1))
    cond = build_topology_cond(cfg["name"], parents, offsets, names, labels,
        positions, quats, quats, scale_factor=scale,
        face_joint_idxs=[names.index(cfg["right_hip"]), names.index(cfg["left_hip"])])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "cond.npy", {cfg["name"]: cond})
    np.savez(out / "robot_mapping.npz", body_ids=body_ids, tip_offsets=tip_offsets,
             rest_body_rot=data.xmat.copy().reshape(-1, 3, 3),
             robot_to_model=ROBOT_TO_MODEL, scale=scale,
             joint_names=np.array([model.joint(i).name for i in range(1, model.njnt)]))
    write_json(out / "summary.json", dict(name=cfg["name"], stats_dataset="objaverse",
        source_mjcf=str(Path(mjcf).resolve()), model_signature=model_signature(model),
        n_joints=len(nodes), n_dofs=30, fps=30, scale_factor=scale,
        robot_frame="X forward, Y left, Z up; metres", model_frame="Y up, Z forward; diameter 2",
        model_joint_limit=dict(max_joints=71, fits=len(nodes) <= 71),
        annotation=dict(zip(names, labels)), config=cfg,
        notes=["Virtual tips preserve terminal joint rotations; they are not robot DOFs.",
               "Generated animation requires bounded kinematic projection; dynamics are not guaranteed."]))
    print(f"Prepared {out}: {len(nodes)} skeleton nodes, 30 DOFs, scale={scale:.8f}")
    return cond


def load_asset(asset, mjcf):
    asset = Path(asset)
    info = json.loads((asset / "summary.json").read_text())
    model = load_model(mjcf or info["source_mjcf"])
    if model_signature(model) != info["model_signature"]:
        raise ValueError("MJCF differs from prepared asset; run prepare again")
    cond = np.load(asset / "cond.npy", allow_pickle=True).item()[info["name"]]
    with np.load(asset / "robot_mapping.npz", allow_pickle=False) as f:
        mapping = dict(f)
    return model, info, cond, mapping


def decode_targets(features, cond, mapping):
    from unimate.utils.motion_utils import recover_unimate_anim_from_rot, recover_unimate_joint_pos_from_rot
    from Animation import rotations_global
    feat = np.asarray(features)
    if feat.ndim != 3 or feat.shape[1:] != (len(cond["parents"]), 12) or len(feat) < 2 or not np.isfinite(feat).all():
        raise ValueError("Expected finite denormalized features [T>=2, prepared joints, 12]")
    anim = recover_unimate_anim_from_rot(feat, cond["parents"], cond["tpos_offsets"])
    positions = recover_unimate_joint_pos_from_rot(feat, cond["parents"], cond["tpos_offsets"])
    C = mapping["robot_to_model"]
    target_pos = positions @ C / float(mapping["scale"])
    delta = C.T @ rotations_global(anim).rotation_matrix(cont6d=False) @ C
    target_rot = delta @ mapping["rest_body_rot"][mapping["body_ids"]]
    return target_pos, target_rot


def fit_targets(model, target_pos, target_rot, mapping, temporal=0.02, max_nfev=60):
    """Full-body bounded least squares; root path/orientation kept from generation."""
    ids = mapping["body_ids"]
    tips = mapping["tip_offsets"]
    body_mask = np.linalg.norm(tips, axis=1) == 0
    T = len(target_pos)
    qpos = np.zeros((T, model.nq))
    errors, success = [], []
    data = mujoco.MjData(model)
    lo, hi = model.jnt_range[1:].T
    previous = np.clip(model.qpos0[7:], lo + 1e-8, hi - 1e-8)
    for t in range(T):
        root_q = Rotation.from_matrix(target_rot[t, 0]).as_quat()
        data.qpos[:3] = target_pos[t, 0]
        data.qpos[3:7] = root_q[[3, 0, 1, 2]]
        def residual(q, regularize=True):
            data.qpos[7:] = q
            mujoco.mj_forward(model, data)
            rotations = data.xmat[ids].reshape(-1, 3, 3)
            positions = data.xpos[ids] + np.einsum("nij,nj->ni", rotations, tips)
            pos_err = (positions - target_pos[t]).ravel()
            rot_err = Rotation.from_matrix(np.swapaxes(target_rot[t, body_mask], -1, -2) @ rotations[body_mask]).as_rotvec().ravel()
            result = [pos_err, 0.10 * rot_err]
            if regularize and temporal:
                result.append(temporal * (q - previous))
            return np.concatenate(result)
        result = least_squares(residual, previous, bounds=(lo, hi), max_nfev=max_nfev,
                               ftol=1e-8, xtol=1e-8, gtol=1e-8)
        previous = result.x.copy()
        residual(previous)
        qpos[t] = data.qpos
        positions = data.xpos[ids] + np.einsum("nij,nj->ni", data.xmat[ids].reshape(-1, 3, 3), tips)
        errors.append(np.linalg.norm(positions-target_pos[t], axis=-1))
        success.append(bool(result.success))
    return qpos, np.array(errors), success


def diagnostics(model, qpos, fps, cfg):
    data = mujoco.MjData(model)
    body_pos, body_rot, collision_count, penetration, soles = [], [], [], [], []
    feet = [model.body(n).id for n in cfg["feet"]]
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[1:].copy())
        body_rot.append(data.xquat[1:, [1, 2, 3, 0]].copy())
        depths = [float(c.dist) for c in data.contact if c.dist < -0.001 and
                  model.geom_bodyid[c.geom1] != 0 and model.geom_bodyid[c.geom2] != 0]
        collision_count.append(len(depths))
        penetration.append(max([0.] + [-d for d in depths]))
        foot_min = []
        for bid in feet:
            lows = []
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != bid or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                    continue
                R = data.geom_xmat[g].reshape(3, 3)
                s = model.geom_size[g]
                typ = model.geom_type[g]
                if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    extent = s[0] + abs(R[2, 2]) * s[1]
                elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                    extent = np.abs(R[2]) @ s
                elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
                    extent = s[0]
                else:
                    raise ValueError("Foot clearance requires primitive capsule/box/sphere collision shapes")
                lows.append(data.geom_xpos[g, 2] - extent)
            if not lows:
                raise ValueError("No collision geometries found for foot")
            foot_min.append(min(lows))
        soles.append(foot_min)
    velocities = np.empty((len(qpos), model.nv))
    for t in range(len(qpos)-1):
        mujoco.mj_differentiatePos(model, velocities[t], 1/fps, qpos[t], qpos[t+1])
    velocities[-1] = velocities[-2]
    return dict(body_pos=np.array(body_pos), body_rot=np.array(body_rot), qvel=velocities,
                foot_clearance=np.array(soles), self_collision_count=np.array(collision_count),
                max_self_penetration=np.array(penetration))


def convert(asset, motion, output, mjcf=None, fps=30., temporal=0.02, max_nfev=60, ground=False):
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be positive and finite")
    model, info, cond, mapping = load_asset(asset, mjcf)
    target_pos, target_rot = decode_targets(np.load(motion, allow_pickle=False), cond, mapping)
    qpos, errors, success = fit_targets(model, target_pos, target_rot, mapping, temporal, max_nfev)
    diag = diagnostics(model, qpos, fps, info["config"])
    raw_clearance = float(diag["foot_clearance"].min())
    ground_shift = 0.
    if ground:
        # One constant translation preserves the generated root path and does
        # not flatten jumps or introduce frame-wise vertical jitter.
        ground_shift = -float(diag["foot_clearance"].min())
        qpos[:, 2] += ground_shift
        diag = diagnostics(model, qpos, fps, info["config"])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix != ".npz":
        raise ValueError("Output path must end with .npz")
    np.savez_compressed(out, model_signature=info["model_signature"], fps=fps, qpos=qpos, root_pos=qpos[:, :3],
        root_rot=qpos[:, [4, 5, 6, 3]], dof_pos=qpos[:, 7:], dof_vel=diag["qvel"][:, 6:],
        joint_names=mapping["joint_names"], body_names=np.array([model.body(i).name for i in range(1, model.nbody)]),
        projection_error=errors, **diag)
    report = dict(source_motion=str(Path(motion).resolve()), frames=len(qpos), fps=fps,
        dofs=30, coordinate_system="metres; X forward, Y left, Z up",
        qpos_quaternion="wxyz", root_rot_quaternion="xyzw", body_rot_quaternion="xyzw",
        projection_mean_m=float(errors.mean()), projection_max_m=float(errors.max()),
        solver_success_frames=sum(success), solver_total_frames=len(success),
        max_joint_speed_rad_s=float(np.abs(diag["qvel"][:, 6:]).max()),
        self_collision_frames=int(np.count_nonzero(diag["self_collision_count"])),
        max_self_penetration_m=float(diag["max_self_penetration"].max()),
        raw_minimum_foot_clearance_m=raw_clearance,
        minimum_foot_clearance_m=float(diag["foot_clearance"].min()),
        constant_ground_shift_m=ground_shift, temporal_weight=temporal,
        model_signature=info["model_signature"],
        status="kinematic_candidate_not_dynamics_validated")
    write_json(out.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))
    return report


def render(asset, motion, output, mjcf=None):
    import imageio.v2 as imageio
    model, info, _, _ = load_asset(asset, mjcf)
    spec = mujoco.MjSpec.from_file(str(Path(mjcf or info["source_mjcf"]).resolve()))
    spec.worldbody.add_geom(name="preview_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], rgba=[0.24, 0.28, 0.32, 1])
    spec.worldbody.add_light(pos=[1, -2, 3], dir=[-1, 2, -3])
    model = spec.compile()
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    clip = np.load(motion, allow_pickle=False)
    if "model_signature" in clip and str(clip["model_signature"]) != info["model_signature"]:
        raise ValueError("Motion was exported with a different robot model")
    if not np.array_equal(clip["joint_names"], np.array([model.joint(i).name for i in range(1, model.njnt)])):
        raise ValueError("Motion joint order differs from model")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.7, 135, -15
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(model, height=720, width=960) as renderer, imageio.get_writer(output, fps=float(clip["fps"])) as writer:
        for q in clip["qpos"]:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            camera.lookat[:] = q[:3] + np.array([0, 0, 0.05])
            renderer.update_scene(data, camera=camera, scene_option=option)
            writer.append_data(renderer.render())
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--mjcf", required=True)
    a.add_argument("--output", default="outputs/rig/astro_p2")
    a.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = sub.add_parser("convert")
    a.add_argument("--asset", default="outputs/rig/astro_p2")
    a.add_argument("--motion", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--mjcf")
    a.add_argument("--fps", type=float, default=30)
    a.add_argument("--temporal", type=float, default=0.02)
    a.add_argument("--max-nfev", type=int, default=60)
    a.add_argument("--ground", action="store_true")
    a = sub.add_parser("render")
    a.add_argument("--asset", default="outputs/rig/astro_p2")
    a.add_argument("--motion", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--mjcf")
    args = vars(p.parse_args())
    command = args.pop("command")
    globals()[command](**args)


if __name__ == "__main__":
    main()
