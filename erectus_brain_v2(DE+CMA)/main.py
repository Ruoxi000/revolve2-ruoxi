"""Main script: supports 'cmaes' | 'de' | 'hybrid (DE->CMA-ES)'."""

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


def _save_generation(dbengine: Engine, experiment: Experiment, gen_idx: int, params: np.ndarray, fits: np.ndarray) -> None:
    """Helper: save one generation to DB."""
    pop = Population(
        individuals=[
            Individual(genotype=Genotype(vec), fitness=float(fit))
            for vec, fit in zip(params, fits)
        ]
    )
    generation = Generation(
        experiment=experiment,
        generation_index=gen_idx,
        population=pop,
    )
    logging.info("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()


def _run_cma(dbengine: Engine, experiment: Experiment, evaluator: Evaluator, initial_mean: np.ndarray,
             total_gens: int, start_gen_index: int, rng_seed: int, initial_sigma: float | None = None) -> None:
    """Run CMA-ES for `total_gens` generations starting from `initial_mean` (and optional sigma)."""
    dim = initial_mean.size
    options = cma.CMAOptions()
    options.set("bounds", list(config.BOUNDS))
    options.set("seed", int(rng_seed))
    if config.CMA_POPSIZE is not None:
        options.set("popsize", int(config.CMA_POPSIZE))

    sigma = float(initial_sigma) if initial_sigma is not None else float(config.INITIAL_STD)
    opt = cma.CMAEvolutionStrategy(initial_mean.tolist(), sigma, options)

    for local_gen in range(total_gens):
        global_gen = start_gen_index + local_gen
        logging.info(f"[CMA-ES] Generation {global_gen + 1}")
        solutions = opt.ask()
        fitnesses = evaluator.evaluate(solutions, generation=global_gen)
        # CMA-ES uses minimization; we maximize fitness -> tell with negative
        opt.tell(solutions, -fitnesses)
        _save_generation(dbengine, experiment, global_gen, np.asarray(solutions), np.asarray(fitnesses))


def run_experiment(dbengine: Engine) -> None:
    """
    Run an experiment.

    :param dbengine: An openened database with matching initialize database structure.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # RNG seed
    rng_seed = seed_from_time() % 2**32  # CMA 兼容
    rng = np.random.default_rng(rng_seed)

    # 记录实验
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # CPG 网络结构 & 输出映射
    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
    cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    evaluator = Evaluator(
        headless=True,
        num_simulators=config.NUM_SIMULATORS,
        cpg_network_structure=cpg_network_structure,
        body=config.BODY,
        output_mapping=output_mapping,
    )

    dim = cpg_network_structure.num_connections
    low, high = config.BOUNDS

    if config.OPTIMIZER == "cmaes":
        initial_mean = np.full(dim, 0.5, dtype=float)
        _run_cma(
            dbengine=dbengine,
            experiment=experiment,
            evaluator=evaluator,
            initial_mean=initial_mean,
            total_gens=config.NUM_GENERATIONS,
            start_gen_index=0,
            rng_seed=rng_seed,
            initial_sigma=config.INITIAL_STD,
        )

    elif config.OPTIMIZER == "de":
        # 初始化种群
        population = rng.uniform(low, high, size=(config.POPULATION_SIZE, dim))
        fitnesses = evaluator.evaluate(list(population), generation=0)
        _save_generation(dbengine, experiment, 0, population, fitnesses)

        # 迭代
        gen = 0
        while gen < config.NUM_GENERATIONS - 1:
            logging.info(f"[DE] Generation {gen + 1}")
            trial_vectors = []
            for i in range(config.POPULATION_SIZE):
                idxs = list(range(config.POPULATION_SIZE))
                idxs.remove(i)
                r1, r2, r3 = rng.choice(idxs, 3, replace=False)
                mutant = population[r1] + config.DE_F * (population[r2] - population[r3])
                mutant = np.clip(mutant, low, high)
                cross = rng.random(dim) < config.DE_CR
                if not np.any(cross):
                    cross[rng.integers(dim)] = True
                trial = np.where(cross, mutant, population[i])
                trial_vectors.append(trial)

            trial_vectors = np.asarray(trial_vectors)
            trial_fitnesses = evaluator.evaluate(trial_vectors, generation=gen + 1)

            improved = trial_fitnesses > fitnesses
            population[improved] = trial_vectors[improved]
            fitnesses[improved] = trial_fitnesses[improved]

            gen += 1
            _save_generation(dbengine, experiment, gen, population, fitnesses)

    elif config.OPTIMIZER == "hybrid":
        # -------------------
        # Stage 1: DE
        # -------------------
        de_gens = int(config.HYBRID_DE_GENERATIONS)
        cma_gens = int(config.HYBRID_CMA_GENERATIONS)

        # 让 Evaluator 的阶段切换基于“总代数”
        assert config.NUM_GENERATIONS == de_gens + cma_gens, \
            "config.NUM_GENERATIONS 应等于 DE + CMA 的总代数（用于 evaluator 的阶段切换）。"

        population = rng.uniform(low, high, size=(config.POPULATION_SIZE, dim))
        fitnesses = evaluator.evaluate(list(population), generation=0)
        _save_generation(dbengine, experiment, 0, population, fitnesses)

        gen = 0
        while gen < de_gens - 1:
            logging.info(f"[HYBRID-DE] Generation {gen + 1} / {de_gens}")
            trial_vectors = []
            for i in range(config.POPULATION_SIZE):
                idxs = list(range(config.POPULATION_SIZE))
                idxs.remove(i)
                r1, r2, r3 = rng.choice(idxs, 3, replace=False)
                mutant = population[r1] + config.DE_F * (population[r2] - population[r3])
                mutant = np.clip(mutant, low, high)
                cross = rng.random(dim) < config.DE_CR
                if not np.any(cross):
                    cross[rng.integers(dim)] = True
                trial = np.where(cross, mutant, population[i])
                trial_vectors.append(trial)

            trial_vectors = np.asarray(trial_vectors)
            trial_fitnesses = evaluator.evaluate(trial_vectors, generation=gen + 1)

            improved = trial_fitnesses > fitnesses
            population[improved] = trial_vectors[improved]
            fitnesses[improved] = trial_fitnesses[improved]

            gen += 1
            _save_generation(dbengine, experiment, gen, population, fitnesses)

        # -------------------
        # Stage 2: CMA-ES（热启动）
        #   用 DE 末代的 top-K 初始化 CMA 的 mean/std
        # -------------------
        k = int(config.HYBRID_TOPK_FOR_CMA)
        idx_sorted = np.argsort(fitnesses)[::-1]
        top_idx = idx_sorted[:k]
        top_params = population[top_idx]

        mean_init = np.mean(top_params, axis=0)
        std_init_vec = np.std(top_params, axis=0)
        # 防止全 0 方差（卡死），加一点最小步长
        sigma_init = float(max(config.INITIAL_STD * 0.5, np.median(std_init_vec)))

        start_gen_index = gen + 1
        logging.info(f"[HYBRID] Switch to CMA-ES. Start sigma={sigma_init:.4f}")
        _run_cma(
            dbengine=dbengine,
            experiment=experiment,
            evaluator=evaluator,
            initial_mean=mean_init,
            total_gens=cma_gens,
            start_gen_index=start_gen_index,
            rng_seed=rng_seed,
            initial_sigma=sigma_init,
        )

    else:
        raise ValueError(f"Unknown optimizer '{config.OPTIMIZER}'")


def main() -> None:
    """Run the program."""
    setup_logging(file_name="log.txt")
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    Base.metadata.create_all(dbengine)

    for _ in range(config.NUM_REPETITIONS):
        run_experiment(dbengine)


if __name__ == "__main__":
    main()
