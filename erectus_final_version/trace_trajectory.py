"""
tools/trace_trajectory.py
功能：找到最后一代的最佳个体，重跑并记录每一帧的 (x,y,z)，
绘制轨迹图 (Fig 5) 和高度稳定性图 (Fig 6)。
"""
import matplotlib.pyplot as plt
from sqlalchemy import select, desc, func
from sqlalchemy.orm import Session

# Revolve2 Imports
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

# Local Imports (新增 Population)
from database_components import Genotype, Individual, Generation, Population
import config


def main():
    # 1. 打开数据库
    db_engine = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)

    target_gen = config.NUM_GENERATIONS  # 找最后一代

    print(f"Searching for best individual in Generation {target_gen}...")

    with Session(db_engine) as session:
        # --- 修正后的查询逻辑 ---
        stmt = (
            select(Individual, Genotype)
            .join(Population, Individual.population_id == Population.id)
            .join(Generation, Generation.population_id == Population.id)
            .join(Genotype, Individual.genotype_id == Genotype.id)
            .where(Generation.generation_index == target_gen)
            .order_by(desc(Individual.fitness))
            .limit(1)
        )
        row = session.execute(stmt).first()

        # 容错处理：如果指定的代数不存在（比如跑挂了只跑到499代），找最大代数
        if not row:
            print(f"Generation {target_gen} not found. Trying max available generation...")
            subq = select(func.max(Generation.generation_index))
            max_g = session.execute(subq).scalar()
            target_gen = max_g

            # Retry query
            stmt = (
                select(Individual, Genotype)
                .join(Population, Individual.population_id == Population.id)
                .join(Generation, Generation.population_id == Population.id)
                .join(Genotype, Individual.genotype_id == Genotype.id)
                .where(Generation.generation_index == target_gen)
                .order_by(desc(Individual.fitness))
                .limit(1)
            )
            row = session.execute(stmt).first()

        if not row:
            print("Error: Database seems empty.")
            return

        individual, genotype = row
        print(f"Selected Individual ID: {individual.id}, Fitness: {individual.fitness}, Gen: {target_gen}")

        # 2. 发育 (Develop)
        robot = genotype.develop()

    # 3. 运行底层仿真 (获取每一帧数据)
    print("Running full trace simulation...")
    simulator = LocalSimulator(headless=True, num_simulators=1)
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)

    batch_params = make_standard_batch_parameters(simulation_time=config.SIM_TIME)

    scene_states = simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_params,
        scenes=[scene]
    )[0]

    # 4. 提取数据
    times = []
    x_path = []
    y_path = []
    z_height = []

    total_frames = len(scene_states)
    start_frame = int(total_frames * config.FITNESS_START_FRACTION)

    for i, state in enumerate(scene_states):
        robot_state = state.get_modular_robot_simulation_state(robot)
        pose = robot_state.get_pose()

        t = (i / total_frames) * config.SIM_TIME
        times.append(t)
        x_path.append(pose.position.x)
        y_path.append(pose.position.y)
        z_height.append(pose.position.z)

    print(f"Simulation done. {len(times)} frames captured.")

    # 5. 绘图 1: 轨迹图
    plt.figure(figsize=(6, 6))
    plt.plot(x_path[:start_frame], y_path[:start_frame], 'gray', linestyle='--', alpha=0.5, label='Warm-up')
    plt.plot(x_path[start_frame:], y_path[start_frame:], 'b-', linewidth=2, label='Evaluated Path')
    plt.scatter(x_path[start_frame], y_path[start_frame], c='green', marker='o', zorder=5, label='Start')
    plt.scatter(x_path[-1], y_path[-1], c='red', marker='x', zorder=5, label='End')
    plt.title(f"Best Individual Trajectory (Gen {target_gen})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("trajectory_trace.png", dpi=300)
    print("Saved trajectory_trace.png")

    # 6. 绘图 2: 高度稳定性
    plt.figure(figsize=(10, 4))
    plt.plot(times, z_height, label='Core Height', color='purple')
    plt.axhline(y=config.FALL_HEIGHT_THRESHOLD, color='red', linestyle='--', label='Fall Threshold (0.1m)')
    plt.axvspan(0, config.SIM_TIME * config.FITNESS_START_FRACTION, color='gray', alpha=0.2, label='Warm-up Period')
    plt.title("Vertical Stability Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.ylim(0, 0.5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("height_stability.png", dpi=300)
    print("Saved height_stability.png")


if __name__ == "__main__":
    main()