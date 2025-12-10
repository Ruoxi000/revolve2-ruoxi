"""Evaluator class implementing custom fitness (height-only fall; orientation events penalize 'upright' if configured)."""

from __future__ import annotations

import math
import numpy as np
import logging

from database_components import Genotype

from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import fitness_functions, terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

import config


# ---------- helpers ----------
def _count_true_runs(mask: np.ndarray, min_frames: int) -> int:
    """Count continuous True-runs of length >= min_frames."""
    count = 0
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= min_frames:
                count += 1
            i = j
        else:
            i += 1
    return count


def _count_events_with_recovery(mask_true: np.ndarray, min_true_frames: int, min_false_frames: int) -> int:
    """
    Count True-runs as events, but require at least `min_false_frames` consecutive False between events.
    Avoids long True-runs being split by noise and prevents long-only-one-penalty.
    """
    n = mask_true.size
    i = 0
    count = 0
    while i < n:
        while i < n and not mask_true[i]:
            i += 1
        if i >= n:
            break
        j = i
        while j < n and mask_true[j]:
            j += 1
        if (j - i) >= min_true_frames:
            count += 1
        # recovery
        k = j
        false_streak = 0
        while k < n and false_streak < min_false_frames:
            if not mask_true[k]:
                false_streak += 1
            else:
                false_streak = 0
            k += 1
        i = k
    return count


def _count_invert_events_u(u_seq: np.ndarray, min_true_frames: int, u_start: float, u_end: float) -> int:
    """
    Count events over u-sequence using hysteresis:
      start when u < u_start ; end when u > u_end ;
    require at least `min_true_frames` samples with (u < u_start).
    We will pass either u_seq (for 'inverted') or -u_seq (for 'upright' events).
    """
    n = u_seq.size
    i = 0
    count = 0
    while i < n:
        while i < n and not (u_seq[i] < u_start):
            i += 1
        if i >= n:
            break
        j = i
        true_len = 0
        while j < n:
            if u_seq[j] < u_start:
                true_len += 1
            elif u_seq[j] > u_end:
                break
            j += 1
        if true_len >= min_true_frames:
            count += 1
        i = j + 1
    return count


def _quat_xyzw(q) -> tuple[float, float, float, float]:
    """Extract quaternion as (x, y, z, w)."""
    if hasattr(q, "x") and hasattr(q, "w"):
        return float(q.x), float(q.y), float(q.z), float(q.w)
    if hasattr(q, "xyzw"):
        x, y, z, w = q.xyzw
        return float(x), float(y), float(z), float(w)
    if hasattr(q, "elements"):
        e = q.elements
        return float(e[0]), float(e[1]), float(e[2]), float(e[3])
    arr = np.array(q, dtype=float).ravel()
    if arr.size != 4:
        return 0.0, 0.0, 0.0, 1.0
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


def _yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Compute yaw from quaternion components."""
    t0 = 2.0 (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1)


def _upright_dot_axis(qx: float, qy: float, qz: float, qw: float, axis: str = "Z") -> float:
    """
    Dot(world+Axis, body+Axis) in [-1,1] for the chosen local body axis.
    Using rotation-matrix diagonal terms from quaternion:
      R11 = 1 - 2*(y^2 + z^2)   (X)
      R22 = 1 - 2*(x^2 + z^2)   (Y)
      R33 = 1 - 2*(x^2 + y^2)   (Z)
    """
    ax = (axis or "Z").upper()
    if ax == "X":
        return 1.0 - 2.0 * (qy*qy + qz*qz)
    elif ax == "Y":
        return 1.0 - 2.0 * (qx*qx + qz*qz)
    else:
        return 1.0 - 2.0 * (qx*qx + qy*qy)


def _norm_height(z: float) -> float:
    """Normalize height to [0,1] using [H_MIN, H_MAX] with clamping."""
    return float(
        np.clip(
            (z - config.H_MIN) / max(1e-12, (config.H_MAX - config.H_MIN)),
            0.0,
            1.0,
        )
    )


class Evaluator(Eval):
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain
    current_generation: int

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
    ) -> None:
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()
        self.current_generation = 0

    def _phase_weights(self, g: int) -> tuple[float, float, float]:
        """Return (w_move, w_yaw, phase_progress) for generation index g."""
        gA_end = int(config.NUM_GENERATIONS * config.STAND_PHASE_FRAC)
        trans = int(config.TRANSITION_LENGTH)
        if g < gA_end:
            return config.W_MOVE_STAND, 0.0, 0.0
        if g < gA_end + trans:
            alpha = (g - gA_end) / max(1, trans)
            move = config.W_MOVE_STAND + alpha * (
                config.W_MOVE_MAX - config.W_MOVE_STAND
            )
            yaw = alpha * config.W_YAW
            return move, yaw, alpha
        return config.W_MOVE_MAX, config.W_YAW, 1.0

    def evaluate(
        self,
        population: list[Genotype],
        return_metrics: bool = False,  # keep breakdown for rerun
    ) -> list[float] | tuple[list[float], list[dict]]:
        """Evaluate multiple robots with custom fitness (height-only fall; orientation events penalize 'upright' if configured)."""
        g = self.current_generation
        w_move, w_yaw, _phase_progress = self._phase_weights(g)
        sim_seconds = float(config.SIM_TIME)

        robots = [genotype.develop() for genotype in population]

        # build scenes
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # simulate
        sim_params = make_standard_batch_parameters(simulation_time=sim_seconds)
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=sim_params,
            scenes=scenes,
        )

        fits: list[float] = []
        metrics_out: list[dict] = []

        for robot, states in zip(robots, scene_states):
            start = int(len(states) * config.FITNESS_START_FRACTION)
            ms_start = states[start].get_modular_robot_simulation_state(robot)
            msN = states[-1].get_modular_robot_simulation_state(robot)

            # 1) XY displacement (single-shot)
            dxy = fitness_functions.xy_displacement(ms_start, msN)

            # 2) Height drop penalty between start and end (normalized)
            try:
                z_start = float(ms_start.get_pose().position[2])
                z_end = float(msN.get_pose().position[2])
            except Exception:
                z_start, z_end = 0.0, 0.0
            h_start_norm = _norm_height(z_start)
            h_end_norm = _norm_height(z_end)
            height_drop = max(0.0, h_start_norm - h_end_norm)
            penalty_height_drop = float(config.HEIGHT_DROP_WEIGHT) * height_drop

            # 3) Yaw change (penalty)
            try:
                pose0 = ms_start.get_pose()
                pose1 = msN.get_pose()
                x0, y0, z0, w0 = _quat_xyzw(getattr(pose0, "orientation", [0, 0, 0, 1]))
                x1, y1, z1, w1 = _quat_xyzw(getattr(pose1, "orientation", [0, 0, 0, 1]))
                dyaw = abs(
                    _yaw_from_quat_xyzw(x1, y1, z1, w1)
                    - _yaw_from_quat_xyzw(x0, y0, z0, w0)
                )
            except Exception:
                dyaw = 0.0

            # 4) Series for height & uprightness
            z_list = []
            u_list = []
            axis = getattr(config, "BODY_UP_AXIS", "Z")
            for s in states[start:]:
                ms = s.get_modular_robot_simulation_state(robot)
                pose = ms.get_pose()
                z_list.append(float(pose.position[2]))
                ori = getattr(pose, "orientation", None)
                if ori is not None:
                    qx, qy, qz, qw = _quat_xyzw(ori)
                    u_list.append(_upright_dot_axis(qx, qy, qz, qw, axis=axis))
            z_seq = np.array(z_list, dtype=float)
            u_seq = np.array(u_list, dtype=float) if len(u_list) == len(z_list) else np.array([], dtype=float)

            # mean height for standing reward
            h_bar = float(np.mean(z_seq)) if z_seq.size > 0 else 0.0

            # 5) FALL: height-only
            fall_mask = (z_seq < float(config.FALL_HEIGHT_THRESHOLD))
            fall_events = _count_events_with_recovery(
                fall_mask.astype(bool),
                min_true_frames=int(getattr(config, "FALL_EVENT_MIN_FRAMES", 2)),
                min_false_frames=int(getattr(config, "FALL_RECOVERY_MIN_FRAMES", 3)),
            )
            fall_frac = float(fall_mask.mean()) if fall_mask.size > 0 else 0.0
            if fall_frac >= float(getattr(config, "FALL_FRAC_BONUS_THRESHOLD", 0.10)):
                fall_events += 1

            # 递进的penalty 等差数列 n*n+1
            fall_penalty = float(config.FALL_PENALTY_PER_EVENT) * float(fall_events * (fall_events + 1) / 2)

            # 6) ORIENTATION events:
            # If INVERT_EVENT_COUNTS_UPRIGHT=True, penalize 'upright' segments (u>0).
            if u_seq.size > 0:
                count_upright = bool(getattr(config, "INVERT_EVENT_COUNTS_UPRIGHT", True))
                u_seq_for_events = -u_seq if count_upright else u_seq  # sign flip to reuse the same comparator
                inv_events = _count_invert_events_u(
                    u_seq_for_events,
                    min_true_frames=int(getattr(config, "INVERT_EVENT_MIN_FRAMES", 2)),
                    u_start=float(getattr(config, "U_INVERT_START", -0.10)),
                    u_end=float(getattr(config, "U_INVERT_END", +0.10)),
                )
                # fraction-based extra count:
                inv_mask = (u_seq > 0.0) if count_upright else (u_seq < 0.0)
                inv_frac = float(inv_mask.mean())
                if inv_frac >= float(getattr(config, "INVERT_FRAC_BONUS_THRESHOLD", 0.10)):
                    inv_events += 1
                invert_penalty = float(config.INVERT_PENALTY_PER_EVENT) * float(inv_events)
            else:
                inv_events = 0
                inv_frac = 0.0
                invert_penalty = 0.0

            # 7) Height normalization (standing reward)
            h_clamp = np.clip(
                h_bar - config.H_MIN, 0.0, config.H_MAX - config.H_MIN
            ) / (config.H_MAX - config.H_MIN)

            total_penalty = fall_penalty + invert_penalty + penalty_height_drop

            # DEBUG: quick check
            logging.debug(
                f"falls={fall_events} (frac={fall_frac:.2f}), "
                f"orient_events={inv_events} (frac={inv_frac:.2f}), "
                f"height_drop={height_drop:.3f}"
            )

            fit = (
                config.W_HEIGHT * h_clamp
                + w_move * dxy
                - w_yaw * dyaw
                - total_penalty
            )
            fits.append(float(fit))

            if return_metrics:  # breakdown for rerun/debug
                metrics_out.append({
                    "dxy": float(dxy),
                    "dyaw": float(dyaw),
                    "h_mean": float(h_bar),
                    "h_start_norm": float(h_start_norm),
                    "h_end_norm": float(h_end_norm),
                    "height_drop": float(height_drop),
                    "penalty_height_drop": float(penalty_height_drop),

                    "fall_events": int(fall_events),
                    "fall_frac": float(fall_frac),
                    "fall_penalty": float(fall_penalty),

                    "orient_events": int(inv_events),          # renamed for clarity in metrics
                    "orient_frac": float(inv_frac),
                    "orient_penalty": float(invert_penalty),
                    "orient_counts_upright": bool(getattr(config, "INVERT_EVENT_COUNTS_UPRIGHT", True)),

                    "total_penalty": float(total_penalty),
                    "u_axis": axis,
                    "u_mean": float(np.mean(u_seq)) if u_seq.size>0 else None,
                    "u_min": float(np.min(u_seq)) if u_seq.size>0 else None,
                    "u_max": float(np.max(u_seq)) if u_seq.size>0 else None,

                    "w_move": float(w_move),
                    "w_yaw": float(w_yaw),
                    "fitness": float(fit),
                })

        if return_metrics:
            return fits, metrics_out
        return fits
