# Dual Conservative Policy Update for Efficient Model-Based Reinforcement Learning

Code for paper "Learning Meta Representations for Agents in Multi-Agent Reinforcement Learning". 

Our code is forked from the [rllib-torch](https://github.com/sebascuri/rllib) repository.

## Requirements

To install create a conda environment:
```bash
$ conda create -n rllib python=3.7
$ conda activate rllib
```

```bash
$ pip install -e .[test,logging,experiments]
```

For Mujoco (license required) Run:
```bash
$ pip install -e .[mujoco]
```

On clusters run:
```bash
$ sudo apt-get install -y --no-install-recommends --quiet build-essential libopenblas-dev python-opengl xvfb xauth
```


## Running an experiment.
To train the DCPU agent in the half-cheetah environment, run:
```bash
$ python run.py
```
To change the training environment and training settings, run:
```bash
$ python run.py --agent-config-file CONFIG_FILE --env-config-file ENV_FILE --seed SEED
```

### License
The RL-Lib is licensed under [MIT License](LICENSE).

## Citing RL-lib

If you use RL-lib in your research please use the following BibTeX entry:
```text
@Misc{Curi2019RLLib,
  author =       {Sebastian Curi},
  title =        {RL-Lib - A pytorch-based library for Reinforcement Learning research.},
  howpublished = {Github},
  year =         {2020},
  url =          {https://github.com/sebascuri/rllib}
}
```
