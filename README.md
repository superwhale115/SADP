# SADP (Symplectic Approximate Dynamic Programming)

## Description
Reinforcement learning (RL) is an advanced technique for solving nonlinear optimal control problems (OCPs). However, its iteration process often encounters severe training instability issues, significantly impairing policy performance. To address these challenges, we propose a symplectic approximate dynamic programming (SADP) algorithm that incorporates the property of symplectic preservation to enhance long-term training stability. By introducing the costate variable in Pontryagin’s minimum principle (PMP), we transform the Lagrange-based OCP into a standard Hamiltonian system and derive its symplectic self-consistency equation and symplectic optimality condition through first-order Euler symplectic discretization. A symplectic policy iteration (SPI) framework is developed for the first time to calculate the solution of the symplectic optimality condition, where the policy evaluation utilizes the symplectic self-consistency equation while policy improvement is achieved by minimizing the Hamiltonian. Furthermore, the designed SADP algorithm integrates the Bellman self-consistency into the symplectic optimality condition, enabling it to utilize zeroth and first-order value information to learn the optimal policy accurately. Simulation and real-world tests demonstrate that SADP exhibits exceptional training stability and superior policy performance, highlighting its potential in handling complex optimal control tasks.

## Installation
Requires:
1. Windows 7 or greater or Linux.
2. Python 3.6 or greater. We recommend using Python 3.8.
3. The installation path must be in English.

You can install through the following steps:
```bash
# clone SADP repository
git clone https://github.com/superwhale115/SADP.git
cd SADP
# create conda environment
conda env create -f sadp_environment.yml
conda activate sadp
# install GOPS
pip install -e .
```

## Quick Start
This is an example of running infinite-horizon Symplectic Approximate Dynamic Programming (INFSADP) on inverted double pendulum environment. 
Train the policy by running:
```bash
python example/infsadp/infsadp_mlp_idpendulum_offserial.py
```
The training results will be saved under the direction ./results
