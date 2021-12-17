"""Model-Based DCPU Agent."""
from itertools import chain

import torch.nn.modules.loss as loss
from torch.optim import Adam

from rllib.algorithms.dcpu import DCPU
from rllib.policy import NNPolicy
from rllib.value_function import NNEnsembleQFunction

from .model_based_agent import ModelBasedAgent
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay, BootstrapExperienceReplay


class DCPUAgent(ModelBasedAgent):
    """Implementation of DCPU Agent."""

    def __init__(
        self,
        policy,
        critic,
        dynamical_model,
        reward_model,
        criterion=loss.MSELoss,
        termination_model=None,
        num_steps=1,
        num_samples=8,
        memory=None,
        *args,
        **kwargs,
    ):

        if memory is None:
            memory = BootstrapExperienceReplay(max_len=100000, num_steps=0)
        self.memory = memory
        self.initial_states_dataset = StateExperienceReplay(
            max_len=1000, dim_state=dynamical_model.dim_state
        )

        algorithm = DCPU(
            policy=policy,
            critic=critic,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            criterion=criterion(reduction="mean"),
            num_steps=num_steps,
            num_samples=num_samples,
            memory=self.memory,
            initial_state_dataset=self.initial_states_dataset,
            initial_distribution=None,
            num_initial_state_samples=0,
            num_initial_distribution_samples=0,
            num_memory_samples=16,
            refresh_interval=2,
            only_sim=False,
            *args,
            **kwargs,
        )

        super().__init__(
            policy_learning_algorithm=algorithm,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            memory=self.memory,
            initial_states_dataset=self.initial_states_dataset,
            *args,
            **kwargs,
        )


        self.optimizer = type(self.optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if (
                    "model" not in name
                    and "target" not in name
                    and "old_policy" not in name
                    and "policy" not in name
                    and p.requires_grad
                )
            ],
            **self.optimizer.defaults,
        )

        self.optimizer_pi = type(self.optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if (
                    ("policy" in name or "critic" in name)
                    and "old_policy" not in name
                    and "target" not in name
                    and p.requires_grad
            )
            ],
            **self.optimizer.defaults,
        )
    @classmethod
    def default(cls, environment, critic=None, policy=None, lr=3e-4, *args, **kwargs):
        """See `AbstractAgent.default'."""
        if critic is None:
            critic = NNEnsembleQFunction.default(environment)
        if policy is None:
            policy = NNPolicy.default(environment)
        optimizer = Adam(chain(policy.parameters(), critic.parameters()), lr=lr)

        return super().default(
            environment=environment,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            *args,
            **kwargs,
        )
