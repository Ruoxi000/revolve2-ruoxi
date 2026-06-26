"""
tools/sample_generations.py
功能：基于 rerun2.py 的数据库读取方式，提取每一代的第一名，重跑仿真，
将 Height 和 Dxy 导出为 CSV，用于绘制论文 Fig 3 & Fig 4。
"""
import logging
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

# 导入 Revolve2 数据库工具
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
# 导入您的数据库模型 (新增 Population)
from database_components import Genotype, Individual, Generation, Population

from evaluator import Evaluator
import config

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # 1. 打开数据库
    db_engine = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)

    # 2. 确定采样点
    # 每 10 代采一次，确保包含第一代和最后一代
    step = 10
    max_gen = config.NUM_GENERATIONS
    sample_gens = list(range(0, max_gen + 1, step))

    print(f"Sampling {len(sample_gens)} generations: {sample_gens}")

    results = []

    # 初始化 Evaluator (Headless)
    evaluator = Evaluator(headless=True, num_simulators=1)

    with Session(db_engine) as session:
        for gen_idx in sample_gens:
            print(f"Processing Generation {gen_idx}...")

            # --- 修正后的查询逻辑 ---
            # 路径：Generation -> Population -> Individual
            stmt = (
                select(Individual, Genotype)
                .join(Population, Individual.population_id == Population.id) # 1. 个体属于种群
                .join(Generation, Generation.population_id == Population.id) # 2. 种群属于某一代
                .join(Genotype, Individual.genotype_id == Genotype.id)       # 3. 个体拥有基因
                .where(Generation.generation_index == gen_idx)               # 4. 筛选代数
                .order_by(desc(Individual.fitness))                          # 5. 取最优
                .limit(1)
            )

            row = session.execute(stmt).first()

            if not row:
                logging.warning(f"Generation {gen_idx} not found or empty.")
                continue

            individual, genotype = row

            # --- 重跑仿真 ---
            # 设置当前代数，以便 Evaluator 使用正确的 Curriculum 权重
            evaluator.current_generation = gen_idx

            # 运行评估
            _, metrics = evaluator.evaluate([genotype], return_metrics=True)
            m = metrics[0]

            # 记录数据
            res = {
                "generation": gen_idx,
                "best_fitness": individual.fitness,
                "h_mean": m['h_mean'],      # 对应 H_max
                "dxy": m['dxy'],            # 对应 D_xy
                "dyaw": m['dyaw'],
                "fall_events": m['fall_events']
            }
            results.append(res)
            print(f"  -> Gen {gen_idx}: Fit={individual.fitness:.2f}, Height={m['h_mean']:.3f}, Dxy={m['dxy']:.3f}")

    # 3. 导出 CSV
    df = pd.DataFrame(results)
    output_file = "evolution_history.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSuccess! Data saved to {output_file}")

if __name__ == "__main__":
    main()