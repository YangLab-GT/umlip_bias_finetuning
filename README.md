# Data repository for "Effects of Bias on Fine-tuning Universal Machine Learned Interatomic Potentials"
Universal machine learned interatomic potentials (uMLIPs) embody a growing area of interest due to their transferability across the periodic table, displaying an error of about 0.6 kcal/mol against the Matbench Discovery test set. However, we show that achieving more accurate predictions on out-of-domain tasks requires fine-tuning. Additionally, we investigate the existence and influence of model biases in molecular dynamics (MD) by examining two approaches for data generation: from multiple MD trajectories in parallel, which we call naive fine-tuning, and from a single MD trajectory with fine-tuning after set intervals, which we call periodic fine-tuning. Our results find that naive fine-tuning generates constrained datasets that fail to represent MD simulations, and thus downstream fine-tuned models fail during extrapolation. In contrast, periodic fine-tuning yields models which are more generalizable and accurate, producing low-error dynamics. These findings indicate the role of uMLIP bias in fine-tuning, and highlights the need for multiple fine-tuning steps. Lastly, we relate unphysical behavior to principal component space, and quantify extrapolations through Q-residual analysis, which are useful as a proxy for epistemic uncertainty for larger simulations.
# Apptainer build & run
To build the docker image from the provided definition file,

```apptainer build umlip_biases.sif apptainer.def```

To run the container

```apptainer run umlip_biases.sif```

Running the container starts a Jupyter server inside the environment. Once it launches, you can access it in your browser at:

```http://localhost:8888/```

