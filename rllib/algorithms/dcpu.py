"""Back-Propagation Through Time Algorithm."""
import torch
import torch.nn as nn
from .abstract_algorithm import AbstractAlgorithm
from .abstract_mb_algorithm import AbstractMBAlgorithm
from rllib.dataset.datatypes import Loss, Observation
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.dataset.utilities import unstack_observations


class DCPU(AbstractAlgorithm, AbstractMBAlgorithm):
    """Dual Conservative Policy Update.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        num_steps=1,
        lambda_=1.0,
        termination_model=None,
        memory=None,
        initial_state_dataset=None,
        initial_distribution=None,
        num_initial_state_samples=0,
        num_initial_distribution_samples=0,
        num_memory_samples=0,
        refresh_interval=4,
        only_sim=True,
        *args,
        **kwargs,
    ):
        AbstractAlgorithm.__init__(self, *args, **kwargs)
        AbstractMBAlgorithm.__init__(
            self,
            dynamical_model,
            reward_model,
            num_steps=1,
            num_samples=5,  # 15
            termination_model=termination_model,
        )
        self.initial_distribution = initial_distribution
        self.initial_state_dataset = initial_state_dataset
        self.num_initial_state_samples = num_initial_state_samples
        self.num_initial_distribution_samples = num_initial_distribution_samples
        self.num_memory_samples = num_memory_samples
        self.memory = memory
        self.n_heads = 3
        self.sim_memory_list = []
        for _ in range(self.n_heads):
            sim_memory_head = ExperienceReplay(
                max_len=memory.max_len,
                num_steps=memory.num_steps,
                transformations=memory.transformations,
            )
            self.sim_memory_list.append(sim_memory_head)
        self.refresh_interval = refresh_interval
        self.count = 0
        self.only_sim = only_sim
        self.mse_loss = nn.MSELoss()

    def actor_loss(self, observation, ref_policy):
        """Use the model to compute the gradient loss."""
        return self.pathwise_loss(observation, ref_policy).reduce(self.criterion.reduction)


    def forward(self, observation, ref_policy):
        def base_algo(observation, ref_policy):
            if isinstance(observation, Observation):
                trajectories = [observation]
            else:
                trajectories = observation
            self.reset_info()
            loss = Loss()
            for ind_traj, trajectory in enumerate(trajectories):
                loss += self.actor_loss(trajectory, ref_policy)
                loss += self.critic_loss(trajectory, ref_policy)
                loss += self.regularization_loss(trajectory, ref_policy, len(trajectories))

            return loss / len(trajectories)

        batch_size = observation.reward.shape[0]
        if len(self.sim_memory_list[0]) < batch_size:
            real_loss = base_algo(observation, ref_policy)
            return real_loss
        if not ref_policy:
            sim_observation_list = []
            for head, sim_m in enumerate(self.sim_memory_list):
                sim_observation_list.append(self.sim_memory_list[head].sample_batch(batch_size)[0])
            sim_loss = base_algo(sim_observation_list, ref_policy)
            real_loss = base_algo(observation, ref_policy)
            return sim_loss + real_loss
        else:
            sim_observation_list = []
            for head, sim_m in enumerate(self.sim_memory_list):
                sim_observation_list.append(self.sim_memory_list[head].sample_batch(batch_size)[0])
            sim_loss = base_algo(sim_observation_list, ref_policy)
            real_loss = base_algo(observation, ref_policy)

            return real_loss + sim_loss


    def simulate(
        self, state, policy, head_ind, initial_action=None, logger=None, stack_obs=False
    ):
        """Simulate from initial_states."""
        self.dynamical_model.eval()
        with torch.no_grad():
            trajectory = super().simulate(state, policy, head_ind, stack_obs=stack_obs)

        for observations in trajectory:
            observation_samples = unstack_observations(observations)
            for observation in observation_samples:
                self.sim_memory_list[head_ind].append(observation)

        return trajectory

    def _sample_initial_states(self):
        """Get initial states to sample from."""
        # Samples from experience replay empirical distribution.
        obs, *_ = self.memory.sample_batch(self.num_memory_samples)
        for transform in self.memory.transformations:
            obs = transform.inverse(obs)
        initial_states = obs.state[:, 0, :]  # obs is an n-step return.

        # Samples from empirical initial state distribution.
        if self.num_initial_state_samples > 0:
            initial_states_ = self.initial_state_dataset.sample_batch(
                self.num_initial_state_samples
            )
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        # Samples from initial distribution.
        if self.num_initial_distribution_samples > 0:
            initial_states_ = self.initial_distribution.sample(
                (self.num_initial_distribution_samples,)
            )
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        initial_states = initial_states.unsqueeze(0)
        return initial_states

    def update(self):
        """Update base algorithm."""
        for head in range(self.n_heads):
            self.simulate(state=self._sample_initial_states(), policy=self.policy, head_ind=head)
        super().update()
        self.pathwise_loss.critic.entropy_regularization = self.entropy_loss.eta.item()

    def reset(self):
        """Reset base algorithm."""
        self.count += 1
        if (self.count % self.refresh_interval) == 0:
            for head in range(self.n_heads):
                self.sim_memory_list[head].reset()
        super().reset()
