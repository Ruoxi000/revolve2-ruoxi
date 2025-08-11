"""Main script for the example."""

import logging

import cma
import config
import numpy as np
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
from evaluator import Evaluator
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import seed_from_time
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    active_hinges_to_cpg_network_structure_neighbor,
)


def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # Create an rng seed.
    rng_seed = seed_from_time() % 2**32  # compatible with cma.

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # Find all active hinges in the body
    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)

    # Create a structure for the CPG network and mapping.
    cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
        active_hinges
    )

    evaluator = Evaluator(
        headless=True,
        num_simulators=config.NUM_SIMULATORS,
        cpg_network_structure=cpg_network_structure,
        body=config.BODY,
        output_mapping=output_mapping,
    )

    if config.OPTIMIZER == "cmaes":
        initial_mean = cpg_network_structure.num_connections * [0.5]
        options = cma.CMAOptions()
        options.set("bounds", list(config.BOUNDS))
        options.set("seed", rng_seed)
        opt = cma.CMAEvolutionStrategy(initial_mean, config.INITIAL_STD, options)

        logging.info("Start optimization process (CMA-ES).")
        while opt.countiter < config.NUM_GENERATIONS:
            logging.info(
                f"Generation {opt.countiter + 1} / {config.NUM_GENERATIONS}."
            )
            solutions = opt.ask()
            fitnesses = evaluator.evaluate(solutions, generation=opt.countiter)
            opt.tell(solutions, -fitnesses)

            population = Population(
                individuals=[
                    Individual(genotype=Genotype(parameters), fitness=fitness)
                    for parameters, fitness in zip(solutions, fitnesses)
                ]
            )

            generation = Generation(
                experiment=experiment,
                generation_index=opt.countiter,
                population=population,
            )
            logging.info("Saving generation.")
            with Session(dbengine, expire_on_commit=False) as session:
                session.add(generation)
                session.commit()

    elif config.OPTIMIZER == "de":
        rng = np.random.default_rng(rng_seed)
        dim = cpg_network_structure.num_connections
        bounds_low, bounds_high = config.BOUNDS
        population = rng.uniform(bounds_low, bounds_high, size=(config.POPULATION_SIZE, dim))
        fitnesses = evaluator.evaluate(list(population), generation=0)

        pop = Population(
            individuals=[
                Individual(genotype=Genotype(params), fitness=fit)
                for params, fit in zip(population, fitnesses)
            ]
        )
        generation = Generation(
            experiment=experiment,
            generation_index=0,
            population=pop,
        )
        logging.info("Saving generation.")
        with Session(dbengine, expire_on_commit=False) as session:
            session.add(generation)
            session.commit()

        gen = 0
        while gen < config.NUM_GENERATIONS - 1:
            logging.info(f"Generation {gen + 1} / {config.NUM_GENERATIONS}.")

            trial_vectors = []
            for i in range(config.POPULATION_SIZE):
                indices = list(range(config.POPULATION_SIZE))
                indices.remove(i)
                r1, r2, r3 = rng.choice(indices, 3, replace=False)
                mutant = population[r1] + config.DE_F * (population[r2] - population[r3])
                mutant = np.clip(mutant, bounds_low, bounds_high)

                cross = rng.random(dim) < config.DE_CR
                if not np.any(cross):
                    cross[rng.integers(dim)] = True
                trial = np.where(cross, mutant, population[i])
                trial_vectors.append(trial)

            trial_fitnesses = evaluator.evaluate(trial_vectors, generation=gen + 1)

            for i in range(config.POPULATION_SIZE):
                if trial_fitnesses[i] > fitnesses[i]:
                    population[i] = trial_vectors[i]
                    fitnesses[i] = trial_fitnesses[i]

            gen += 1

            pop = Population(
                individuals=[
                    Individual(genotype=Genotype(params), fitness=fit)
                    for params, fit in zip(population, fitnesses)
                ]
            )
            generation = Generation(
                experiment=experiment,
                generation_index=gen,
                population=pop,
            )
            logging.info("Saving generation.")
            with Session(dbengine, expire_on_commit=False) as session:
                session.add(generation)
                session.commit()

    else:
        raise ValueError(f"Unknown optimizer '{config.OPTIMIZER}'")


def main() -> None:
    """Run the program."""
    # Set up logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for _ in range(config.NUM_REPETITIONS):
        run_experiment(dbengine)


if __name__ == "__main__":
    main()
