"""ModelBasedAlgorithm."""
import torch

from .abstract_mb_algorithm import AbstractMBAlgorithm
from .derived_algorithm import DerivedAlgorithm
from rllib.value_function.model_based_q_function import ModelBasedQFunction
from rllib.util.losses.pathwise_loss import PathwiseLoss


class Dyna(DerivedAlgorithm, AbstractMBAlgorithm):
    """Dyna Algorithm."""

    def __init__(
        self,
        base_algorithm,
        dynamical_model,
        reward_model,
        num_steps=5,
        num_samples=15,
        termination_model=None,
        only_sim=False,
        *args,
        **kwargs,
    ):
        DerivedAlgorithm.__init__(self, base_algorithm=base_algorithm)
        AbstractMBAlgorithm.__init__(
            self,
            dynamical_model,
            reward_model,
            num_steps=num_steps,
            num_samples=num_samples,
            termination_model=termination_model,
        )
        self.base_algorithm.criterion = type(self.base_algorithm.criterion)(
            reduction="mean"
        )
        """
        self.base_algorithm.pathwise_loss.critic = ModelBasedQFunction(
                dynamical_model=dynamical_model,
                reward_model=reward_model,
                termination_model=termination_model,
                num_samples=self.num_samples,
                num_steps=20,
                policy=self.policy,
          #      value_function=self.value_function,
                value_function=None,
                gamma=self.gamma,
                lambda_=1.0,
                reward_transformer=self.reward_transformer,
                entropy_regularization=self.entropy_loss.eta.item(),
            )
        """
    #    self.pathwise_loss = PathwiseLoss(critic=self.critic, policy=self.policy, q_pol=self.q_list)

        self.only_sim = only_sim



    def forward(self, observation, ref_policy):
        """
        real_loss = self.base_algorithm.forward(observation)
        with torch.no_grad():
            state = observation.state[..., 0, :]
            sim_observation = self.simulate(state, self.policy, stack_obs=True)
        sim_loss = self.base_algorithm.forward(sim_observation)
        if self.only_sim:
            return sim_loss
        return real_loss + sim_loss
        """
        super().forward(None, None)
