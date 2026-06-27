# 🤖 Robo Erectus: Evolving Upright Life Forms

**Master Project - Vrije Universiteit Amsterdam** **Author:** Ruoxi Ji  
**Supervisors:** Prof. Dr. A. E. Eiben

This repository contains the core source code for the Master Project titled *"Evolving 'Robo Erectus': A Fitness Curriculum for Upright Life Forms"*. Built upon the **Revolve2** framework and powered by the **MuJoCo** physics engine, this project explores the body-brain co-evolution of modular robots.

By introducing an innovative **Phased Curriculum Learning** mechanism, this framework successfully breaks the notorious "crawler trap"—a pervasive local optimum in Evolutionary Robotics where organisms converge on sprawling, low-profile crawling morphologies. It guides modular agents to autonomously evolve stable, upright structures and directed walking gaits without any prior structural knowledge.

---

## 🎥 Results & Multimedia

- **Video Repository:** Demonstrates the dynamic locomotion gaits and typical failure modes of the Top-5 elite individuals.  
  👉 [Access the Google Drive Video Folder](https://drive.google.com/drive/folders/1XdV1Zf4kJGj3DQbbllEYPBozH46sF1FD?usp=sharing)

- **Best Evolution Database:** The full 500-generation evolutionary data is stored in the root directory under the filename `database_new_500gen_best.sqlite` (containing complete genotype mappings, phenotypes, and fitness evaluation histories). You can render and visualize these elite individuals using the provided replay scripts.

---

## 📂 Repository Structure

The main directory contains the implementation of the evolutionary loop and evaluation metrics:

- **`main.py`**: The primary entry point. It initializes the CPPN-NEAT population and manages the steady-state co-evolutionary loop (including tournament selection, crossover, and mutation).
- **`evaluator.py`**: **[Core Contribution]** Custom fitness evaluation module. Features:
  - Dynamic weight scheduling for the **Phased Curriculum** based on the current generation.
  - A state-machine-driven penalty system utilizing a **Hysteresis Filter** to rigorously punish falling and rolling behaviors while filtering contact jitter.
- **`config.py`**: Global experimental configurations, including population size, mutation rates, MuJoCo simulation steps, and generational thresholds for curriculum phases.
- **`plot.py` / `plot_comparison.py`**: Data visualization scripts used to extract evolutionary logs from the SQLite database and generate convergence curves, height trajectories, and comparison charts.
- **`rerun2.py` / `sample_generations.py`**: Physics replay and rendering utilities to load specified individuals (e.g., elites) from the database and visualize their behavior in the MuJoCo 3D simulator.
- **`database_components/`**: Data abstraction layers for database storage inherited from Revolve2. It handles SQLAlchemy mappings and requires no manual modification.

---

## ⚙️ Installation

To set up the workspace, please complete the environment setup sequentially:

1. **Prerequisite: Install Revolve2 Core Platform** Please follow the official installation guidelines provided by the Revolve2 documentation to establish the core platform environment and its underlying physics backends (MuJoCo):  
   👉 [Revolve2 Official Installation Guide](https://ci-group.github.io/revolve2/installation)

2. **Clone this Project Repository**
   ```bash
   git clone https://github.com/Ruoxi000/revolve2-ruoxi.git
   cd revolve2-ruoxi/erectus_final_version

3. **Install Supplementary Dependencies**
   Once the Revolve2 core ecosystem is activated and active within your virtual environment, install the supplementary dependencies required specifically by this evolutionary branch:
   ```bash
   pip install -r requirements.txt

## 📜 License
This project is licensed under the MIT License. Please cite this project and the corresponding Master Thesis if you use or adapt this code in your academic work.
