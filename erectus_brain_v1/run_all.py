"""Run multiple training configurations for comparison."""

import config
import main
from sqlalchemy import select
from sqlalchemy.orm import Session
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from database_components import Individual
from revolve2.standards.modular_robots_v1 import gecko_v1
from revolve2.standards.modular_robots_v2 import gecko_v2

# configurations
bodies = [
    ("v1", gecko_v1()),
    ("v2", gecko_v2()),
]
w_move_max_values = [3.0, 4.0, 5.0]
height_groups = [
    ("low", 0.03, 0.03, 0.20),
    ("high", 0.06, 0.06, 0.25),
]
optimizers = ["de", "cmaes"]

def run_all() -> None:
    """Run all combinations of bodies, speed weights, heights and optimizers."""

    config.NUM_GENERATIONS = 100

    with open("result.txt", "w") as out:
        for body_name, body in bodies:
            for w_move_max in w_move_max_values:
                for height_label, fall_threshold, h_min, h_max in height_groups:
                    for optimizer in optimizers:
                        config.BODY = body
                        config.W_MOVE_MAX = w_move_max
                        config.FALL_HEIGHT_THRESHOLD = fall_threshold
                        config.H_MIN = h_min
                        config.H_MAX = h_max
                        config.OPTIMIZER = optimizer
                        config.DATABASE_FILE = (
                            f"database_{body_name}_W{int(w_move_max)}_H{height_label}_{optimizer}.sqlite"
                        )
                        print(
                            f"Running: body={body_name}, W_MOVE_MAX={w_move_max},",
                            f" heights={height_label}, optimizer={optimizer}"
                        )
                        main.main()
                        db = open_database_sqlite(
                            config.DATABASE_FILE, OpenMethod.OPEN_IF_EXISTS
                        )
                        with Session(db) as ses:
                            best = ses.execute(
                                select(Individual.fitness)
                                .order_by(Individual.fitness.desc())
                                .limit(1)
                            ).scalar_one()
                        out.write(f"{config.DATABASE_FILE},{best}\n")
                        out.flush()

if __name__ == "__main__":
    run_all()
