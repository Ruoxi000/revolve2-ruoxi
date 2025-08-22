# rerun_one.py
import logging
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session

import config
from database_components import Genotype, Individual
from evaluator import Evaluator
from revolve2.experimentation.database import open_database_sqlite, OpenMethod
from revolve2.experimentation.logging import setup_logging


def main() -> None:
    setup_logging()
    eng = open_database_sqlite(config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS)

    with Session(eng, expire_on_commit=False) as ses:
        # 取“每个 genotype 的历史最好分”，排序去重后列前 TOPK
        stmt = (
            select(Genotype, func.max(Individual.fitness).label("fitness"))
            .join(Individual, Individual.genotype_id == Genotype.id)
            .group_by(Genotype.id)
            .order_by(desc("fitness"))
            .limit(max(config.RERUN_TOPK, config.RERUN_RANK))
        )
        rows = ses.execute(stmt).all()
        if not rows:
            logging.error("No individuals found in the database.")
            return

        logging.info("Top %d unique individuals by best fitness:", len(rows))
        for i, (g, fit) in enumerate(rows, 1):
            logging.info("%2d) genotype_id=%s  best_fitness=%.6f", i, g.id, float(fit))

        k = config.RERUN_RANK
        if not (1 <= k <= len(rows)):
            logging.error("RERUN_RANK=%d out of range 1..%d", k, len(rows))
            return

        genotype, listed_fit = rows[k - 1]
        logging.info("Selected rank #%d: gid=%s, listed_best_fitness=%.6f", k, genotype.id, float(listed_fit))

        evaluator = Evaluator(headless=bool(config.RERUN_HEADLESS), num_simulators=1)
        if hasattr(evaluator, "current_generation"):
            evaluator.current_generation = getattr(config, "NUM_GENERATIONS", 0)  # 用成熟期权重

        fitness_list = evaluator.evaluate([genotype])  # 一次只跑一个
        logging.info("Re-simulated fitness = %.6f", float(fitness_list[0]))

        if not config.RERUN_HEADLESS:
            try:
                input("Simulation finished. Press <Enter> to exit...")
            except Exception:
                pass


if __name__ == "__main__":
    main()
