"""Main script:
- Supports 'cmaes' | 'de' | 'hybrid (DE -> CMA-ES)'
- Constant 30s sim time in Evaluator
- RevDE/jDE improvements on DE
- Hot-start + mild restarts + top-k re-evaluation for CMA-ES
"""

import logging
import math

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


# ---------- persistence helpers ----------
def _save_generation(dbengine: Engine, experiment: Experiment, gen_idx: int, params: np.ndarray, fits: np.ndarray) -> None:
    """Save a generation to the database."""
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


# ---------- CMA-ES with hot-start + mild restarts ----------
def _run_cma_with_tricks(
    dbengine: Engine,
    experiment: Experiment,
    evaluator: Evaluator,
    initial_mean: np.ndarray,
    total_gens: int,
    start_gen_index: int,
    rng_seed: int,
    initial_sigma: float | None = None,
) -> None:
    """CMA-ES wrapper: hot-start, active+elitist, diagonal start, mild restarts, top-k re-evaluation."""
    dim = initial_mean.size

    # ---- build base options (as dict) ----
    base_opts_obj = cma.CMAOptions()
    base_opts_obj.set("bounds", list(config.BOUNDS))
    base_opts_obj.set("seed", int(rng_seed))
    if config.CMA_ACTIVE:
        base_opts_obj.set("CMA_active", True)
    if config.CMA_ELITIST:
        base_opts_obj.set("CMA_elitist", True)
    if isinstance(config.CMA_DIAGONAL, int) and config.CMA_DIAGONAL > 0:
        base_opts_obj.set("CMA_diagonal", int(config.CMA_DIAGONAL))

    # IMPORTANT: turn CMAOptions into a plain dict (so we can copy/mutate with normal dict ops)
    base_options: dict = dict(base_opts_obj)

    # if user explicitly set popsize, record an integer override; otherwise leave None to use CMA default
    popsize_override: int | None = None
    if config.CMA_POPSIZE is not None:
        popsize_override = int(config.CMA_POPSIZE)

    sigma = float(initial_sigma) if initial_sigma is not None else float(config.INITIAL_STD)

    # ---- restart state ----
    restarts_done = 0
    global_best_f = -1e18
    global_best_x = initial_mean.copy()

    local_gen = 0  # generations within CMA stage (across restarts)

    while local_gen < total_gens:
        budget_left = total_gens - local_gen

        # make a fresh dict of options for this CMA run
        opts = dict(base_options)
        if popsize_override is not None:
            # Now it's a clean int; won't be the CMAOptions string default.
            opts["popsize"] = int(popsize_override)

        logging.info(
            f"[CMA-ES] (restart #{restarts_done}) init: sigma={sigma:.4f}, popsize={opts.get('popsize', 'auto')}"
        )
        opt = cma.CMAEvolutionStrategy(initial_mean.tolist(), sigma, opts)

        no_improve_count = 0
        while budget_left > 0:
            global_gen = start_gen_index + local_gen
            logging.info(f"[CMA-ES] Generation {global_gen + 1}")

            # 1) ask solutions
            solutions = np.asarray(opt.ask(), dtype=float)
            solutions = np.clip(solutions, config.BOUNDS[0], config.BOUNDS[1])

            # 2) evaluate
            fitnesses = evaluator.evaluate(solutions.tolist(), generation=global_gen)

            # optional re-evaluation of top-k to reduce noise
            if config.CMA_TOPK_REEVAL > 0 and config.CMA_REEVAL_TIMES > 1:
                k = min(config.CMA_TOPK_REEVAL, solutions.shape[0])
                idx = np.argsort(fitnesses)[::-1][:k]
                for i in idx:
                    acc = fitnesses[i]
                    for _ in range(config.CMA_REEVAL_TIMES - 1):
                        acc += evaluator.evaluate([solutions[i]], generation=global_gen)[0]
                    fitnesses[i] = acc / float(config.CMA_REEVAL_TIMES)

            # 3) tell (CMA minimizes)
            opt.tell(solutions.tolist(), (-fitnesses).tolist())

            # 4) save to DB
            _save_generation(dbengine, experiment, global_gen, np.asarray(solutions), np.asarray(fitnesses))

            # progress & stagnation tracking
            best_idx = int(np.argmax(fitnesses))
            best_f = float(fitnesses[best_idx])
            if best_f > global_best_f:
                global_best_f = best_f
                global_best_x = solutions[best_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            local_gen += 1
            budget_left -= 1

            # restart if stagnating and we still have budget
            if (
                config.CMA_USE_RESTARTS
                and no_improve_count >= config.CMA_STALL_GENS
                and restarts_done < config.CMA_RESTARTS_MAX
                and local_gen < total_gens
            ):
                logging.info("[CMA-ES] Stagnation detected, restarting with larger popsize and sigma.")
                # hot-start next run from global best
                initial_mean = global_best_x.copy()
                sigma *= float(config.CMA_SIGMA_BOOST)
                # increase population size based on the actual integer popsize of current run
                cur_popsize = int(opt.popsize)
                popsize_override = int(max(cur_popsize * config.CMA_POPSIZE_INC, cur_popsize + 1))
                restarts_done += 1
                break  # break inner loop to restart

        else:
            # budget used up in this run
            break

    logging.info(f"[CMA-ES] Finished. Best fitness in CMA stage: {global_best_f:.4f}")




# ---------- DE with RevDE/jDE tricks ----------
def _de_evolve_with_tricks(
    rng: np.random.Generator,
    evaluator: Evaluator,
    dim: int,
    gens: int,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run DE for 'gens' generations with:
    - jDE-style self-adaptation of F and CR
    - dither (small random F/CR for a subset)
    - elite injection (replace a fraction of worst by jittered elites)
    Returns: (population, fitnesses, last_generation_index)
    """
    N = config.POPULATION_SIZE

    # Initialize population uniformly in bounds
    population = rng.uniform(low, high, size=(N, dim))
    fitnesses = evaluator.evaluate(population.tolist(), generation=0)

    # Per-individual F and CR (jDE)
    F_i = np.full(N, config.DE_F, dtype=float)
    CR_i = np.full(N, config.DE_CR, dtype=float)

    gen = 0
    while gen < gens - 1:
        logging.info(f"[HYBRID-DE] Generation {gen + 1} / {gens}")

        # jDE adaptation: with probability tau, resample F or CR from ranges
        if config.DE_USE_JDE:
            mask_f = rng.random(N) < config.DE_TAU_F
            F_i[mask_f] = rng.uniform(config.DE_F_RANGE[0], config.DE_F_RANGE[1], size=np.sum(mask_f))
            mask_cr = rng.random(N) < config.DE_TAU_CR
            CR_i[mask_cr] = rng.uniform(config.DE_CR_RANGE[0], config.DE_CR_RANGE[1], size=np.sum(mask_cr))

        trial_vectors = np.empty_like(population)
        for i in range(N):
            # choose F, CR for this individual
            F_use = F_i[i]
            CR_use = CR_i[i]

            # dither (small fraction): randomize F/CR within dither ranges
            if config.DE_USE_DITHER and rng.random() < config.DE_DITHER_PROB:
                F_use = float(rng.uniform(*config.DE_DITHER_F_RANGE))
                CR_use = float(rng.uniform(*config.DE_DITHER_CR_RANGE))

            # choose distinct indices
            idxs = list(range(N))
            idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            best_idx = int(np.argmax(fitnesses))

            # mutation strategies
            if config.DE_STRATEGY == "best1bin":
                mutant = population[best_idx] + F_use * (population[r2] - population[r3])
            elif config.DE_STRATEGY == "current2best":
                mutant = population[i] + F_use * (population[best_idx] - population[i]) + F_use * (population[r2] - population[r3])
            else:  # "rand1bin"
                mutant = population[r1] + F_use * (population[r2] - population[r3])

            # binomial crossover
            cross = rng.random(dim) < CR_use
            if not np.any(cross):
                cross[rng.integers(dim)] = True
            trial = np.where(cross, mutant, population[i])
            trial = np.clip(trial, low, high)
            trial_vectors[i] = trial

        # Evaluate trials
        trial_fitnesses = evaluator.evaluate(trial_vectors.tolist(), generation=gen + 1)

        # Greedy selection
        improved = trial_fitnesses > fitnesses
        population[improved] = trial_vectors[improved]
        fitnesses[improved] = trial_fitnesses[improved]

        # Elite injection to maintain diversity & exploit good zones
        if config.DE_ELITE_INJECTION and config.DE_INJECTION_RATE > 0.0:
            k = min(config.DE_ELITE_K, N)
            worst_n = max(1, int(round(N * config.DE_INJECTION_RATE)))
            idx_sorted = np.argsort(fitnesses)
            worst_idx = idx_sorted[:worst_n]
            top_idx = idx_sorted[-k:]
            elites = population[top_idx]
            # tile elites and add small Gaussian noise
            injected = np.tile(elites, (int(np.ceil(worst_n / k)), 1))[:worst_n]
            noise = rng.normal(loc=0.0, scale=config.DE_INJECTION_NOISE, size=injected.shape)
            injected = np.clip(injected + noise, low, high)
            population[worst_idx] = injected
            # After injection we don't evaluate immediately; evaluate next generation

        gen += 1
        _save_generation(dbengine, experiment, gen, population, fitnesses)  # type: ignore  # saved below in run_experiment

    return population, fitnesses, gen


# ---------- experiment driver ----------
def run_experiment(dbengine: Engine) -> None:
    logging.info("----------------")
    logging.info("Start experiment")

    # RNG
    rng_seed = seed_from_time() % 2**32
    rng = np.random.default_rng(rng_seed)

    # Save experiment
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # Build CPG structure & mapping from the fixed body
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

    # Helper to save the very first generation (g=0)
    def save_initial(population: np.ndarray, gen_idx: int = 0):
        fitnesses = evaluator.evaluate(population.tolist(), generation=gen_idx)
        _save_generation(dbengine, experiment, gen_idx, population, fitnesses)
        return population, fitnesses

    if config.OPTIMIZER == "cmaes":
        # Simple CMA-ES from scratch
        initial_mean = np.full(dim, 0.0, dtype=float)  # center
        _run_cma_with_tricks(
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
        # Pure DE with tricks
        population = rng.uniform(low, high, size=(config.POPULATION_SIZE, dim))
        population, fitnesses = save_initial(population, gen_idx=0)

        # Run DE for NUM_GENERATIONS-1 additional generations
        gen = 0
        while gen < config.NUM_GENERATIONS - 1:
            logging.info(f"[DE] Generation {gen + 1} / {config.NUM_GENERATIONS}")
            # reuse the DE routine with 'gens=2' window (do one update then break)
            pop, fits, _ = _de_evolve_with_tricks(
                rng=rng,
                evaluator=evaluator,
                dim=dim,
                gens=2,            # run exactly one update step
                low=low,
                high=high,
            )
            # '_de_evolve_with_tricks' internally saves; refresh population/fitnesses
            population, fitnesses = pop, fits
            gen += 1

    elif config.OPTIMIZER == "hybrid":
        # ---------------------------------------
        # Stage 1: DE (with tricks), length = HYBRID_DE_GENERATIONS
        # ---------------------------------------
        de_gens = int(config.HYBRID_DE_GENERATIONS)
        cma_gens = int(config.HYBRID_CMA_GENERATIONS)
        assert config.NUM_GENERATIONS == de_gens + cma_gens, \
            "NUM_GENERATIONS must equal DE + CMA lengths (used for phase scheduling in Evaluator)."

        # Initial population & save g=0
        population = rng.uniform(low, high, size=(config.POPULATION_SIZE, dim))
        fitnesses = evaluator.evaluate(population.tolist(), generation=0)
        _save_generation(dbengine, experiment, 0, population, fitnesses)

        # DE main loop (using the same routine but expanded for 'de_gens')
        # We adapt the routine here to have visibility on DB saving
        N = config.POPULATION_SIZE
        F_i = np.full(N, config.DE_F, dtype=float)
        CR_i = np.full(N, config.DE_CR, dtype=float)
        gen = 0
        while gen < de_gens - 1:
            logging.info(f"[HYBRID-DE] Generation {gen + 1} / {de_gens}")

            # jDE adaptation
            if config.DE_USE_JDE:
                mask_f = rng.random(N) < config.DE_TAU_F
                F_i[mask_f] = rng.uniform(config.DE_F_RANGE[0], config.DE_F_RANGE[1], size=np.sum(mask_f))
                mask_cr = rng.random(N) < config.DE_TAU_CR
                CR_i[mask_cr] = rng.uniform(config.DE_CR_RANGE[0], config.DE_CR_RANGE[1], size=np.sum(mask_cr))

            trial_vectors = np.empty_like(population)
            for i in range(N):
                F_use = F_i[i]
                CR_use = CR_i[i]
                if config.DE_USE_DITHER and rng.random() < config.DE_DITHER_PROB:
                    F_use = float(rng.uniform(*config.DE_DITHER_F_RANGE))
                    CR_use = float(rng.uniform(*config.DE_DITHER_CR_RANGE))

                idxs = list(range(N))
                idxs.remove(i)
                r1, r2, r3 = rng.choice(idxs, 3, replace=False)
                best_idx = int(np.argmax(fitnesses))

                if config.DE_STRATEGY == "best1bin":
                    mutant = population[best_idx] + F_use * (population[r2] - population[r3])
                elif config.DE_STRATEGY == "current2best":
                    mutant = population[i] + F_use * (population[best_idx] - population[i]) + F_use * (population[r2] - population[r3])
                else:
                    mutant = population[r1] + F_use * (population[r2] - population[r3])

                cross = rng.random(dim) < CR_use
                if not np.any(cross):
                    cross[rng.integers(dim)] = True
                trial = np.where(cross, mutant, population[i])
                trial = np.clip(trial, low, high)
                trial_vectors[i] = trial

            trial_fitnesses = evaluator.evaluate(trial_vectors.tolist(), generation=gen + 1)

            improved = trial_fitnesses > fitnesses
            population[improved] = trial_vectors[improved]
            fitnesses[improved] = trial_fitnesses[improved]

            # Elite injection
            if config.DE_ELITE_INJECTION and config.DE_INJECTION_RATE > 0.0:
                k = min(config.DE_ELITE_K, N)
                worst_n = max(1, int(round(N * config.DE_INJECTION_RATE)))
                idx_sorted = np.argsort(fitnesses)
                worst_idx = idx_sorted[:worst_n]
                top_idx = idx_sorted[-k:]
                elites = population[top_idx]
                injected = np.tile(elites, (int(np.ceil(worst_n / k)), 1))[:worst_n]
                noise = rng.normal(loc=0.0, scale=config.DE_INJECTION_NOISE, size=injected.shape)
                injected = np.clip(injected + noise, low, high)
                population[worst_idx] = injected
                # evaluate injected next generation

            gen += 1
            _save_generation(dbengine, experiment, gen, population, fitnesses)

        # ---------------------------------------
        # Stage 2: CMA-ES (hot-start + mild restarts), length = HYBRID_CMA_GENERATIONS
        # ---------------------------------------
        k = int(config.HYBRID_TOPK_FOR_CMA)
        idx_sorted = np.argsort(fitnesses)[::-1]
        top_idx = idx_sorted[:k]
        top_params = population[top_idx]

        # mean from top-K, sigma from their spread (guarded by minimum INITIAL_STD*0.5)
        mean_init = np.mean(top_params, axis=0)
        std_init_vec = np.std(top_params, axis=0)
        sigma_init = float(max(config.INITIAL_STD * 0.5, np.median(std_init_vec)))

        start_gen_index = gen + 1
        logging.info(f"[HYBRID] Switch to CMA-ES. Hot-start sigma={sigma_init:.4f}")
        _run_cma_with_tricks(
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
