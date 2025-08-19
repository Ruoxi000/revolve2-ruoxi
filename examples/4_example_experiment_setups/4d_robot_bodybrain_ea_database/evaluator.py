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


def _count_fall_events(z_seq: list[float], hz: float, thr: float, min_dur: float) -> tuple[int, float]:
    """Return (fall_count, fall_fraction) with min-duration hysteresis."""
    if len(z_seq) == 0:
        return 0, 0.0
    min_frames = max(1, int(min_dur * hz))
    below = np.array([z < thr for z in z_seq], dtype=bool)
    frac = float(np.mean(below))
    count = 0
    i = 0
    n = len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) >= min_frames:
                count += 1
            i = j
        else:
            i += 1
    return count, frac


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
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
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
            return 0.0, 0.0, 0.0
        if g < gA_end + trans:
            alpha = (g - gA_end) / max(1, trans)
            return alpha * config.W_MOVE_MAX, alpha * config.W_YAW, alpha
        return config.W_MOVE_MAX, config.W_YAW, 1.0

    def evaluate(self, population: list[Genotype]) -> list[float]:
        """Evaluate multiple robots with custom fitness."""
        g = self.current_generation
        w_move, w_yaw, phase_progress = self._phase_weights(g)
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

            dxy = fitness_functions.xy_displacement(ms_start, msN)

            try:
                pose0 = ms_start.get_pose()
                pose1 = msN.get_pose()
                x0, y0, z0, w0 = _quat_xyzw(getattr(pose0, "orientation", [0, 0, 0, 1]))
                x1, y1, z1, w1 = _quat_xyzw(getattr(pose1, "orientation", [0, 0, 0, 1]))
                dyaw = abs(_yaw_from_quat_xyzw(x1, y1, z1, w1) - _yaw_from_quat_xyzw(x0, y0, z0, w0))
            except Exception:
                dyaw = 0.0

            z_seq = [
                s.get_modular_robot_simulation_state(robot).get_pose().position[2]
                for s in states[start:]
            ]
            h_bar = float(np.mean(z_seq)) if len(z_seq) > 0 else 0.0

            sim_hz = 1.0 / sim_params.control_frequency
            fall_count, fall_frac = _count_fall_events(
                z_seq,
                hz=sim_hz,
                thr=config.FALL_HEIGHT_THRESHOLD,
                min_dur=config.FALL_EVENT_MIN_DURATION,
            )
            penalty = (
                config.FALL_PENALTY_BASE
                * (1.0 + config.FALL_PENALTY_PER_EXTRA * max(0, fall_count - 1))
                + config.FALL_PENALTY_FRAC_WEIGHT * fall_frac
            )
            penalty *= 1.0 + config.FALL_PENALTY_PHASE_GAIN * phase_progress

            h_clamp = np.clip(
                h_bar - config.H_MIN, 0.0, config.H_MAX - config.H_MIN
            ) / (config.H_MAX - config.H_MIN)

            fit = (
                config.W_HEIGHT * h_clamp
                + w_move * dxy
                - w_yaw * dyaw
                - penalty
            )
            fits.append(float(fit))

        return fits
