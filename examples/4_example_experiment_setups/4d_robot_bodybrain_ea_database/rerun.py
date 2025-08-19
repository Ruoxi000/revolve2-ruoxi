"""Rerun the best robot between all experiments."""

import logging

import config
from database_components import Genotype, Individual
from evaluator import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    # Load the best individual from the database.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    with Session(dbengine) as ses:
        rows = (
            ses.execute(
                select(Genotype, Individual.fitness)
                .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
                .order_by(Individual.fitness.desc())
                .limit(5)
            ).all()
        )

    if len(rows) == 0:
        logging.info("Database is empty.")
        return

    logging.info("Top 5 individuals:")
    for i, (genotype, fitness) in enumerate(rows, start=1):
        logging.info(f"{i}: fitness={fitness} params={genotype}")

    try:
        choice = int(input("Select individual to run [1-5, default 1]: ") or "1")
    except Exception:
        choice = 1
    choice = max(1, min(choice, len(rows)))

    genotype = rows[choice - 1][0]
    logging.info(f"Selected individual {choice} with fitness {rows[choice-1][1]}")

    # Create the evaluator.
    evaluator = Evaluator(headless=False, num_simulators=1)

    # Show the robot.
    evaluator.evaluate([genotype])


if __name__ == "__main__":
    main()
