"""Independent MuJoCo -> feature -> P2 round trips using the supplied model.

P2_MJCF must point at the external asset. Without it these integration tests
skip; source assets are not required for importing the rest of UniMate.
"""
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from unimate.robots.p2 import (ROBOT_TO_MODEL as C, prepare, load_asset,
    decode_targets, fit_targets, diagnostics, model_signature)


@pytest.fixture(scope="module")
def robot(tmp_path_factory):
    source = os.environ.get("P2_MJCF")
    if not source or not Path(source).is_file():
        pytest.skip("Set P2_MJCF to the external P2 primitive collision MJCF")
    asset = tmp_path_factory.mktemp("p2")
    prepare(source, asset)
    return load_asset(asset, source)


def feature_roundtrip(robot):
    from Quaternions import Quaternions
    from unimate.utils.motion_utils import compute_unimate_motion_feats
    model, info, cond, mapping = robot
    data = mujoco.MjData(model)
    ids, tips = mapping["body_ids"], mapping["tip_offsets"]
    parents = np.array(cond["parents"])
    known, positions, rotations, local = [], [], [], []
    rng = np.random.default_rng(37)
    lo, hi = model.jnt_range[1:].T
    for t in range(5):
        q = model.qpos0.copy()
        q[:3] = [0.015*t, -0.008*t, 0.75 + 0.005*t]
        quat = Rotation.from_euler("xyz", [0.025*t, -0.018*t, 0.05*t]).as_quat()
        q[3:7] = quat[[3, 0, 1, 2]]
        q[7:] = np.clip(rng.uniform(-0.16, 0.16, 30), lo + 0.005, hi - 0.005)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        R = data.xmat[ids].reshape(-1, 3, 3)
        pos = data.xpos[ids] + np.einsum("nij,nj->ni", R, tips)
        global_delta = C @ (R @ np.swapaxes(mapping["rest_body_rot"][ids], -1, -2)) @ C.T
        loc = global_delta.copy()
        loc[1:] = np.swapaxes(global_delta[parents[1:]], -1, -2) @ global_delta[1:]
        local.append(loc)
        known.append(q)
        positions.append(pos)
        rotations.append(R)
    local = np.array(local)
    quat = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(5, len(ids), 4)
    features = compute_unimate_motion_feats(np.array(positions) @ C.T * float(mapping["scale"]),
        Quaternions(quat[..., [3, 0, 1, 2]]), parents, Quaternions.id(5))
    return features, np.array(known[:-1]), np.array(positions[:-1]), np.array(rotations[:-1])


def test_schema_and_terminal_dofs(robot):
    model, info, cond, mapping = robot
    assert model.nq == 37
    assert len(cond["parents"]) == 36
    assert len(set(mapping["joint_names"])) == 30
    for j in range(1, model.njnt):
        idx = cond["joint_names"].index(model.body(model.jnt_bodyid[j]).name)
        assert idx in cond["parents"], "Every actuator needs a child rotation slot"
    assert np.linalg.det(C) == pytest.approx(1)
    assert np.all(np.array(cond["parents"])[1:] < np.arange(1,36))


def test_mujoco_feature_roundtrip(robot):
    _, _, cond, mapping = robot
    features, known, expected_pos, expected_rot = feature_roundtrip(robot)
    pos, rot = decode_targets(features, cond, mapping)
    # Root recovery places frame zero at X=Y=0, as the known clip does.
    np.testing.assert_allclose(pos, expected_pos, atol=2e-6)
    np.testing.assert_allclose(rot, expected_rot, atol=2e-6)


def test_bounded_full_body_fit(robot):
    model, info, cond, mapping = robot
    features, known, expected_pos, expected_rot = feature_roundtrip(robot)
    pos, rot = decode_targets(features, cond, mapping)
    qpos, error, success = fit_targets(model, pos, rot, mapping, temporal=0, max_nfev=120)
    assert all(success)
    assert error.max() < 2e-4
    np.testing.assert_allclose(qpos[:,7:], known[:,7:], atol=2e-3)
    assert np.all(qpos[:,7:] >= model.jnt_range[1:,0] - 1e-10)
    assert np.all(qpos[:,7:] <= model.jnt_range[1:,1] + 1e-10)
    diag = diagnostics(model, qpos, 30, info["config"])
    assert diag["qvel"].shape == (4,36)
    assert np.isfinite(diag["qvel"]).all()


def test_invalid_features_and_signature(robot):
    model, _, cond, mapping = robot
    for x in [np.zeros((1,36,12)), np.zeros((3,35,12)), np.full((3,36,12), np.nan)]:
        with pytest.raises(ValueError):
            decode_targets(x, cond, mapping)
    before = model_signature(model)
    old = model.jnt_range[1,0]
    model.jnt_range[1,0] += 0.01
    assert model_signature(model) != before
    model.jnt_range[1,0] = old
