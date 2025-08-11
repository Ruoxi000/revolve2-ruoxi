"""Evaluator – three-phase fitness with yaw penalty."""

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
        raise ValueError("Quaternion-like with size {arr.size}, expected 4")
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


def _yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Return yaw (rotation around Z) from quaternion in (x,y,z,w)."""
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

    def evaluate(
        self,
        solutions: list[npt.NDArray[np.float_]],
        generation: int | None = None,
    ) -> npt.NDArray[np.float_]:
        """Evaluate multiple controller parameter sets."""
        g = generation if generation is not None else self.current_generation

        robots = self._make_robots(solutions)

        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(
                simulation_time=config.SIMULATION_SECONDS
            ),
            scenes=scenes,
        )

        # Phase-based weight switching
        gA_end = config.NUM_GENERATIONS // 3
        trans = config.TRANSITION_LENGTH
        if g < gA_end:
            w_move = 0.0
            w_yaw = 0.0
        elif g < gA_end + trans:
            alpha = (g - gA_end) / trans
            w_move = alpha * config.W_MOVE_MAX
            w_yaw = alpha * config.W_YAW
        else:
            w_move = config.W_MOVE_MAX
            w_yaw = config.W_YAW

        fits: list[float] = []
        for robot, states in zip(robots, scene_states):
            ms0 = states[0].get_modular_robot_simulation_state(robot)
            msN = states[-1].get_modular_robot_simulation_state(robot)

            pose0 = ms0.get_pose()
            pose1 = msN.get_pose()

            dxy = fitness_functions.xy_displacement(ms0, msN)

            x0, y0, z0, w0 = _quat_xyzw(pose0.orientation)
            x1, y1, z1, w1 = _quat_xyzw(pose1.orientation)
            yaw0 = _yaw_from_quat_xyzw(x0, y0, z0, w0)
            yawN = _yaw_from_quat_xyzw(x1, y1, z1, w1)
            dyaw = abs(yawN - yaw0)

            start = int(len(states) * 0.4)
            z_seq = [
                s.get_modular_robot_simulation_state(robot).get_pose().position[2]
                for s in states[start:]
            ]
            h_bar = float(np.mean(z_seq))
            falls = 0
            prev_fell = False
            for z in z_seq:
                fell_now = z < config.FALL_HEIGHT_THRESHOLD
                if fell_now and not prev_fell:
                    falls += 1
                prev_fell = fell_now
            fall_penalty = config.FALL_PENALTY_BASE * falls * (falls + 1) / 2.0

            h_clamp = np.clip(
                h_bar - config.H_MIN, 0.0, config.H_MAX - config.H_MIN
            ) / (config.H_MAX - config.H_MIN)

            fit = (
                config.W_HEIGHT * h_clamp
                + w_move * dxy
                - w_yaw * dyaw
                - fall_penalty
            )
            fits.append(float(fit))

        return np.array(fits)
