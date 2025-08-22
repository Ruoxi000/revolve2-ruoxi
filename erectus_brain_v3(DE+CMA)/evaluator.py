"""Evaluator class implementing custom fitness."""

from __future__ import annotations

import math
import numpy as np

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
def _count_fall_events(
    z_seq: list[float], hz: float, thr: float, min_dur: float
) -> int:
    """Count fall-events: z<thr lasting at least min_dur seconds."""
    if len(z_seq) == 0:
        return 0
    min_frames = max(1, int(min_dur * hz))
    below = np.array([z < thr for z in z_seq], dtype=bool)
    return _count_true_runs(below, min_frames)


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
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1)


def _upright_dot_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    """
    Dot(world_up, body_up) in [-1,1], where
    body_up is the robot trunk +Z axis expressed in world coordinates.

    With quaternion (x,y,z,w) assumed unit-length, the Z component of body_up is:
        u = 1 - 2*(x^2 + y^2)
    (This equals cos(theta), theta = angle between world +Z and body +Z.)
    """
    return 1.0 - 2.0 * (x * x + y * y)


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

    def evaluate(self, population: list[Genotype]) -> list[float]:
        """Evaluate multiple robots with custom fitness."""
        g = self.current_generation
        w_move, w_yaw, _phase_progress = self._phase_weights(g)  # phase used only for weights
        sim_seconds = float(config.SIM_TIME)

        robots = [genotype.develop() for genotype in population]

        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        sim_params = make_standard_batch_parameters(simulation_time=sim_seconds)
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=sim_params,
            scenes=scenes,
        )

        fits: list[float] = []
        for robot, states in zip(robots, scene_states):
            start = int(len(states) * config.FITNESS_START_FRACTION)
            ms_start = states[start].get_modular_robot_simulation_state(robot)
            msN = states[-1].get_modular_robot_simulation_state(robot)

            # 1) XY displacement (single-shot)
            dxy = fitness_functions.xy_displacement(ms_start, msN)

            # 2) Yaw change (penalty)
            try:
                pose0 = ms_start.get_pose()
                pose1 = msN.get_pose()
                x0, y0, z0, w0 = _quat_xyzw(getattr(pose0, "orientation", [0, 0, 0, 1]))
                x1, y1, z1, w1 = _quat_xyzw(getattr(pose1, "orientation", [0, 0, 0, 1]))
                dyaw = abs(_yaw_from_quat_xyzw(x1, y1, z1, w1) - _yaw_from_quat_xyzw(x0, y0, z0, w0))
            except Exception:
                dyaw = 0.0

            # 3) Height series (mean height for standing reward)
            z_seq = [
                s.get_modular_robot_simulation_state(robot).get_pose().position[2]
                for s in states[start:]
            ]
            h_bar = float(np.mean(z_seq)) if len(z_seq) > 0 else 0.0

            # 4) Fall events (simplified, per-event penalty only)
            sim_hz = 1.0 / sim_params.control_frequency  # frames per second
            fall_count = _count_fall_events(
                z_seq,
                hz=sim_hz,
                thr=config.FALL_HEIGHT_THRESHOLD,
                min_dur=config.FALL_EVENT_MIN_DURATION,
            )
            fall_penalty = config.FALL_PENALTY_PER_EVENT * float(fall_count)

            # 5) Inversion events (new): u_t = dot(world_up, body_up) < 0 sustained
            invert_penalty = 0.0
            try:
                # build uprightness mask over the stable window
                upr_mask = []
                for s in states[start:]:
                    pose = s.get_modular_robot_simulation_state(robot).get_pose()
                    ori = getattr(pose, "orientation", None)
                    if ori is None:
                        upr_mask = []  # orientation unavailable -> no inversion penalty
                        break
                    qx, qy, qz, qw = _quat_xyzw(ori)
                    u = _upright_dot_from_quat_xyzw(qx, qy, qz, qw)  # in [-1,1]
                    upr_mask.append(u > 0.0)  # inverted if body +Z points below world +Z
                if len(upr_mask) > 0:
                    inv_count = _count_true_runs(np.array(upr_mask, dtype=bool),
                                                 max(1, int(config.INVERT_EVENT_MIN_DURATION * sim_hz)))
                    invert_penalty = config.INVERT_PENALTY_PER_EVENT * float(inv_count)
            except Exception:
                invert_penalty = 0.0  # be safe: do not penalize if orientation pipeline breaks

            # 6) Height normalization
            h_clamp = np.clip(
                h_bar - config.H_MIN, 0.0, config.H_MAX - config.H_MIN
            ) / (config.H_MAX - config.H_MIN)

            total_penalty = fall_penalty + invert_penalty

            fit = (
                config.W_HEIGHT * h_clamp
                + w_move * dxy
                - w_yaw * dyaw
                - total_penalty
            )
            fits.append(float(fit))

        return fits
