# SADP (Symplectic Approximate Dynamic Programming)

## Description
Approximate dynamic programming (ADP) is an advanced technique for solving complex nonlinear optimal control problems (OCPs) by approximating the solution of the Hamilton-Jacobi-Bellman (HJB) equation. However, mainstream ADP algorithms often encounter significant approximation errors and instability due to their crude reliance on the self-consistency of value function or its state derivative for policy iteration. To address this issue, we propose the symplectic policy iteration (SPI) framework, which introduces a costate variable derived from Pontryagin's minimum principle (PMP). SPI allows us to perceive the OCP as a Hamiltonian system characterized by symplecticity that correlates with long-term stability. Building upon the SPI framework, we develop a symplectic approximate dynamic programming (SADP) algorithm that leverages both the self-consistency equations of value function and costate variable to enhance accuracy and stability in learning the value function. Compared to mainstream ADP algorithms, SADP demonstrates higher training stability and improved policy performance, highlighting the potential of SPI in developing more advanced ADP algorithms.

## Installation
Requires:
1. Windows 7 or greater or Linux.
2. Python 3.6 or greater. We recommend using Python 3.8.
3. The installation path must be in English.

You can install through the following steps:
```bash
# clone SADP-GOPS repository
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
