"""Evaluator – three-phase weights (constant 30s sim), yaw penalty + adaptive fall penalty."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import numpy.typing as npt

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.brain.cpg import (
    BrainCpgNetworkStatic,
    CpgNetworkStructure,
)
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
    # 连续 True 片段计一次事件，且长度>=min_frames
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
    """Extract (x, y, z, w) from a quaternion-like object."""
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
        # 某些后端可能没有 orientation，退化为零四元数（无惩罚）
        return 0.0, 0.0, 0.0, 1.0
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


def _yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Return yaw (rotation around Z) from quaternion in (x,y,z,w)."""
    # ZYX 顺序的经典解
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1)


class Evaluator:
    """Evaluate controllers on a fixed body."""

    _simulator: LocalSimulator
    _terrain: Terrain
    _cpg_network_structure: CpgNetworkStructure
    _body: Body
    _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        cpg_network_structure: CpgNetworkStructure,
        body: Body,
        output_mapping: list[tuple[int, ActiveHinge]],
    ) -> None:
        self._simulator = LocalSimulator(headless=headless, num_simulators=num_simulators)
        self._terrain = terrains.flat()
        self._cpg_network_structure = cpg_network_structure
        self._body = body
        self._output_mapping = output_mapping
        self.current_generation: int = 0

    def _make_robots(self, params_list: Iterable[npt.NDArray[np.float_]]) -> list[ModularRobot]:
        robots: list[ModularRobot] = []
        for params in params_list:
            brain = BrainCpgNetworkStatic.uniform_from_params(
                params=params,
                cpg_network_structure=self._cpg_network_structure,
                initial_state_uniform=math.sqrt(2) * 0.5,
                output_mapping=self._output_mapping,
            )
            robots.append(ModularRobot(body=self._body, brain=brain))
        return robots

    def _phase_weights(self, g: int) -> tuple[float, float, float]:
        """
        三段式权重（仿真时长恒定为 30s）：
        - 前 STAND_PHASE_FRAC：只站稳（w_move=0, w_yaw=0）
        - 中间 TRANSITION_LENGTH 代：线性拉起 w_move/w_yaw
        - 后期：w_move, w_yaw 拉满
        :return: (w_move, w_yaw, phase_progress in [0,1])
        """
        gA_end = int(config.NUM_GENERATIONS * config.STAND_PHASE_FRAC)
        trans = int(config.TRANSITION_LENGTH)

        if g < gA_end:
            return 0.0, 0.0, 0.0
        elif g < gA_end + trans:
            alpha = (g - gA_end) / max(1, trans)
            return alpha * config.W_MOVE_MAX, alpha * config.W_YAW, alpha
        else:
            return config.W_MOVE_MAX, config.W_YAW, 1.0

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]],
        generation: int | None = None,
    ) -> npt.NDArray[np.float_]:
        """Evaluate multiple controller parameter sets."""
        fits, _, _, _, _ = self.evaluate_components(solutions, generation)
        return fits

    def evaluate_components(
        self,
        solutions: list[npt.NDArray[np.float_]],
        generation: int | None = None,
    ) -> tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
    ]:
        """Evaluate controllers and return fitness and its components."""
        g = generation if generation is not None else self.current_generation

        w_move, w_yaw, phase_progress = self._phase_weights(g)
        sim_seconds = float(config.SIM_TIME)  # 恒定 30 秒

        robots = self._make_robots(solutions)

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
        heights: list[float] = []
        velocities: list[float] = []
        yaws: list[float] = []
        penalties: list[float] = []
        for robot, states in zip(robots, scene_states):
            ms0 = states[0].get_modular_robot_simulation_state(robot)
            msN = states[-1].get_modular_robot_simulation_state(robot)

            pose0 = ms0.get_pose()
            pose1 = msN.get_pose()

            # xy 位移
            dxy = fitness_functions.xy_displacement(ms0, msN)

            # 偏航（若后端没有 orientation，就不惩罚）
            try:
                x0, y0, z0, w0 = _quat_xyzw(getattr(pose0, "orientation", [0, 0, 0, 1]))
                x1, y1, z1, w1 = _quat_xyzw(getattr(pose1, "orientation", [0, 0, 0, 1]))
                yaw0 = _yaw_from_quat_xyzw(x0, y0, z0, w0)
                yawN = _yaw_from_quat_xyzw(x1, y1, z1, w1)
                dyaw = abs(yawN - yaw0)
            except Exception:
                dyaw = 0.0

            # 站高（丢弃起始 40% 的下落阶段）
            start = int(len(states) * 0.4)
            z_seq = [
                s.get_modular_robot_simulation_state(robot).get_pose().position[2]
                for s in states[start:]
            ]
            h_bar = float(np.mean(z_seq)) if len(z_seq) > 0 else 0.0

            # 自适应跌倒惩罚
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
            penalty *= (1.0 + config.FALL_PENALTY_PHASE_GAIN * phase_progress)

            # 站高归一化
            h_clamp = np.clip(h_bar - config.H_MIN, 0.0, config.H_MAX - config.H_MIN) / (
                config.H_MAX - config.H_MIN
            )

            height_part = config.W_HEIGHT * h_clamp
            velocity_part = w_move * dxy
            yaw_part = w_yaw * dyaw
            penalty_part = penalty

            fit = height_part + velocity_part - yaw_part - penalty_part

            fits.append(float(fit))
            heights.append(float(height_part))
            velocities.append(float(velocity_part))
            yaws.append(float(yaw_part))
            penalties.append(float(penalty_part))

        return (
            np.array(fits),
            np.array(heights),
            np.array(velocities),
            np.array(yaws),
            np.array(penalties),
        )
