"""
tools/sample_generations.py
功能：提取每一代的第一名，重跑仿真，导出 Height 和 Dxy CSV。
更新：增加了早期（Gen 0-100）的高密度采样点。
"""
import logging
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
# 确保 database_components 路径正确
from database_components import Genotype, Individual, Generation, Population
from evaluator import Evaluator
import config


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # 1. 定义更精细的采样点
    max_gen = config.NUM_GENERATIONS

    # 阶段1 & 2：高密度采样 (前100代，每2代采一次)
    dense_samples = list(range(0, min(101, max_gen + 1), 2))

    # 阶段3：稀疏采样 (100代之后，每10代采一次)
    sparse_samples = list(range(110, max_gen + 1, 10))

    # 合并并去重
    sample_gens = sorted(list(set(dense_samples + sparse_samples)))

    print(f"Sampling strategy definition:")
    print(f" - Dense (0-100): Every 2 gens")
    print(f" - Sparse (100+): Every 10 gens")
    print(f"Total samples to process: {len(sample_gens)}")

    # 2. 打开数据库
    db_engine = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)
    evaluator = Evaluator(headless=True, num_simulators=1)
    results = []

    with Session(db_engine) as session:
        for i, gen_idx in enumerate(sample_gens):
            print(f"[{i + 1}/{len(sample_gens)}] Processing Generation {gen_idx}...")

            # 查询最优个体
            stmt = (
                select(Individual, Genotype)
                .join(Population, Individual.population_id == Population.id)
                .join(Generation, Generation.population_id == Population.id)
                .join(Genotype, Individual.genotype_id == Genotype.id)
                .where(Generation.generation_index == gen_idx)
                .order_by(desc(Individual.fitness))
                .limit(1)
            )
            row = session.execute(stmt).first()

            if not row:
                logging.warning(f"Generation {gen_idx} not found.")
                continue

            individual, genotype = row

            # 重跑仿真 (使用当代的课程权重)
            evaluator.current_generation = gen_idx
            _, metrics = evaluator.evaluate([genotype], return_metrics=True)
            m = metrics[0]

            results.append({
                "generation": gen_idx,
                "h_mean": m['h_mean'],
                "dxy": m['dxy']
            })
            # 使用 \r 实现在同一行打印进度，看起来更整洁
            print(f"  -> Done. Height={m['h_mean']:.3f}, Dxy={m['dxy']:.3f}")

    # 3. 导出 CSV
    # 建议根据当前 config 的数据库名来自动命名 CSV，防止覆盖
    db_name = config.DATABASE_FILE.split('.')[0]
    output_file = f"history_{db_name}.csv"

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nSuccess! Data saved to {output_file}")


if __name__ == "__main__":
    main()