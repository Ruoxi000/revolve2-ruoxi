"""Evaluator – 三段式权重切换 + yaw 惩罚."""

from __future__ import annotations
import numpy as np
import math
from math import atan2, asin
from revolve2.experimentation.evolution.abstract_elements import Evaluator as BaseEval
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains, fitness_functions
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from database_components import Genotype
import config


def _quat_to_euler(q):
    """Convert quaternion (x,y,z,w) -> roll,pitch,yaw (rad)."""
    x, y, z, w = q
    # Z-YX
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(t3, t4)
    return roll, pitch, yaw


def _quat_xyzw(q) -> tuple[float, float, float, float]:
    """Try to extract (x,y,z,w) from a pyrr.Quaternion or array-like."""
    # pyrr.Quaternion 常见属性
    if hasattr(q, "x") and hasattr(q, "w"):
        return float(q.x), float(q.y), float(q.z), float(q.w)
    if hasattr(q, "xyzw"):
        x, y, z, w = q.xyzw  # 有些版本暴露 xyzw
        return float(x), float(y), float(z), float(w)
    if hasattr(q, "elements"):
        e = q.elements  # pyrr 常用存储
        return float(e[0]), float(e[1]), float(e[2]), float(e[3])
    # 退化到 array-like
    arr = np.array(q).astype(float).ravel()
    if arr.size != 4:
        raise ValueError(f"Quaternion-like with size {arr.size}, expected 4")
    return arr[0], arr[1], arr[2], arr[3]

def _yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Return yaw (rotation around Z) from quaternion in (x,y,z,w)."""
    # 常用 ZYX 欧拉角转换：yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1)


class Evaluator(BaseEval):
    """Height + xy + yaw + fall."""

    def __init__(self, headless: bool, num_simulators: int) -> None:
        self._sim = LocalSimulator(headless=headless, num_simulators=num_simulators)
        self._terrain = terrains.flat()
        self.current_generation: int = 0

    # ------------------------------------------------------------------
    def evaluate(
        self,
        population: list[Genotype],
        generation: int | None = None,                      # <<< NEW  (main.py 需传入)
    ) -> list[float]:

        g = generation if generation is not None else self.current_generation
        robots = [genotype.develop() for genotype in population]

        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        scene_states = simulate_scenes(
            simulator=self._sim,
            batch_parameters=make_standard_batch_parameters(
                simulation_time=config.SIMULATION_SECONDS
            ),
            scenes=scenes,
        )

        # ---------- 阶段权重 ----------
        gA_end = config.NUM_GENERATIONS // 3
        trans  = config.TRANSITION_LENGTH
        if g < gA_end:                          # 站立期
            w_move = 0.0; w_yaw = 0.0
        elif g < gA_end + trans:               # 过渡
            α = (g - gA_end) / trans
            w_move = α * config.W_MOVE_MAX
            w_yaw  = α * config.W_YAW
        else:                                           # 步态期
            w_move = config.W_MOVE_MAX
            w_yaw  = config.W_YAW

        fits: list[float] = []

        for robot, states in zip(robots, scene_states):
            ms0 = states[0].get_modular_robot_simulation_state(robot)
            msN = states[-1].get_modular_robot_simulation_state(robot)

            pose0 = ms0.get_pose()
            pose1 = msN.get_pose()

            # 位移、速度
            dxy = fitness_functions.xy_displacement(ms0, msN)

            # yaw 变化
            x0, y0, z0, w0 = _quat_xyzw(pose0.orientation)
            x1, y1, z1, w1 = _quat_xyzw(pose1.orientation)
            yaw0 = _yaw_from_quat_xyzw(x0, y0, z0, w0)
            yawN = _yaw_from_quat_xyzw(x1, y1, z1, w1)
            dyaw = abs(yawN - yaw0)

            # 高度序列
            start = int(len(states) * 0.4)
            z_seq = [s.get_modular_robot_simulation_state(robot).get_pose().position[2]
                     for s in states[start:]]
            h_bar = float(np.mean(z_seq))
            fell  = any(z < config.FALL_HEIGHT_THRESHOLD for z in z_seq)

            # clamp 高度奖励
            h_clamp = np.clip(h_bar - config.H_MIN, 0.0,
                              config.H_MAX - config.H_MIN) / (config.H_MAX - config.H_MIN)

            fit = (
                config.W_HEIGHT * h_clamp +
                w_move * dxy -
                w_yaw  * dyaw -
                (config.FALL_PENALTY if fell else 0.0)
            )
            fits.append(float(fit))

        return fits
