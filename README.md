# Action Sequence Crossover with performance-Informed Interpolation for MAP-Elites

This repository implements:
- **ASCII-ME**, introduced in [_Scaling Policy Gradient Quality-Diversity with Massive Parallelization via Behavioral Variations_](https://arxiv.org/abs/2501.18723), GECCO 2025

All experiments can be reproduced within a containerized environment, ensuring reproducibility!

## Overview

<p align="center">
  <img src="assets/high-level_illustration.png" alt="ASCII-ME Algorithm Illustration" width="600">
  <br>
  <em>High-Level Illustration of the ASCII-ME algorithm</em>
</p>

MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) is a quality-diversity algorithm that creates a collection of high-performing solutions that differ according to features defined by the user. 

The ASCII-ME method employs two distinct variation operators within the standard MAP-Elites loop: 
- **Iso+LineDD**, which mutates a parent genotype based on that
of a randomly selected elite. 
- **ASCII**, which interpolates
between the parent's behavior and another behavior sampled
from the buffer, using performance metrics encapsulated in
𝒁 (performance matrix). The behavioral changes are then mapped to the genotypic
space by 𝑱 (Jacobian matrix) to mutate the parent genotype.

**ASCII-ME** is the first Policy Gradient Quality Diversity algorithm that does not rely on Actor-Critic methods, yet capable of evolving Deep Neural Network policies with thousands of parameters at competitive sample and runtime efficiency. This, combined with its strong scalability on a single GPU, underscores the potential of this promising new framework for non Actor-Critic Policy Gradient Quality Diversity methods.

### Baselines

The repository contains the code to run the following algorithms:
- [_ASCII-ME_](https://arxiv.org/abs/2501.18723)
- [_DCRL-ME_](https://arxiv.org/abs/2401.08632)
- [_PGA-ME_](https://dl.acm.org/doi/10.1145/3449639.3459304)
- [_MEMES_](https://arxiv.org/abs/2303.06137)
- [_PPGA_](https://arxiv.org/abs/2305.13795)
- [_ME_](https://arxiv.org/abs/1504.04909)

## Installation

We provide an Apptainer definition file `apptainer/container.def`, that enables to create a containerized environment in which all the experiments and figures can be reproduced.

First, clone the repository:
```bash
git clone https://github.com/konstantinosmitsides/ASCII-ME.git
```

Then, go to the root of the repository with `cd ASCII-ME` and build the container:
```bash
apptainer build --fakeroot --force apptainer/container.sif apptainer/container.def
```

Finally, you can shell within the container:
```bash
apptainer shell --bind $(pwd):/src/ --cleanenv --containall --home /tmp/ --no-home --nv --pwd /src/ --workdir apptainer/ apptainer/container.sif
```

Once you have a shell in the container, you can run the experiments; see the next section.

## Run main experiments

First, follow the previous section to build and shell into a container. Then, to run any algorithms `<algo>`, on any environments `<env>`, use:
```bash
python main.py env=<env> algo=<algo> seed=$RANDOM 
```

For example, to run ASCII-ME on Walker2d Uni:
```bash
python main.py env=walker2d_uni algo=ascii_me seed=$RANDOM
```

During training, the metrics are logged in the `output/` directory.

The configurations for all algorithms and all environments can be found in the `configs/` directory. Alternatively, they can be modified directly in the command line. For example, to increase `num_iterations` to 4000 and `num_critic_training_steps` to 5000 in PGA-ME, you can run:
```bash
python main.py env=ant_uni algo=pga_me seed=$RANDOM num_iterations=4000 algo.num_critic_training_steps=5000
```

To facilitate the replication of all experiments, you can run the bash script `launch_experiments.sh`. This script will run one seed for each algorithm and each environment. Bear in mind, that in the paper, we replicated all experiments with 20 random seeds, so you would need to run `launch_experiments.sh` 20 times to replicate the results. Please note that your results may vary slightly from those reported in the paper due to hardware differences, particularly GPU specifications. However, the overall trends and comparative performance should remain consistent.

## Figures

Once all the experiments are completed, any figures from the paper can be replicated with the following scripts:

- Figure 2: `fig2.py`
- Figure 3: `fig3.py`
- Figure 4: `fig4.py`
- Figure 5: `fig5.py`

## p-values, scalability scores, coefficients of variation, and relative differences

Once all the experiments are completed, all the values used in the `Results & Analysis` section in the paper can be replicated with the following scripts:
- p-values & relative differences: `analysis_p_values.ipynb`
- scalability scores: `analysis_scal_scores.ipynb`
- coefficients of variation: `analysis_cv.ipynb`
