"""ModelBasedAlgorithm."""
import torch

from rllib.dataset.experience_replay import ExperienceReplay
from rllib.dataset.utilities import unstack_observations
from rllib.util.neural_networks import deep_copy_module, update_parameters

from .dyna import Dyna


class DataAugmentation(Dyna):
    """Data Augmentation Algorithm."""

    def __init__(
        self,
        memory=None,
        initial_state_dataset=None,
        initial_distribution=None,
        num_initial_state_samples=0,
        num_initial_distribution_samples=0,
        num_memory_samples=0,
        refresh_interval=2,
        only_sim=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.initial_distribution = initial_distribution
        self.initial_state_dataset = initial_state_dataset
        self.num_initial_state_samples = num_initial_state_samples
        self.num_initial_distribution_samples = num_initial_distribution_samples
        self.num_memory_samples = num_memory_samples
        self.memory = memory
        self.sim_memory_list = []
        for head in range(1):
            sim_memory_head = ExperienceReplay(
                max_len=memory.max_len,
                num_steps=memory.num_steps,
                transformations=memory.transformations,
            )
            self.sim_memory_list.append(sim_memory_head)
        self.refresh_interval = refresh_interval
        self.count = 0
        self.only_sim = only_sim

    def forward(self, observation, ref_policy):
        """Rollout model and call base algorithm with transitions."""
        batch_size = observation.reward.shape[0]

        if len(self.sim_memory_list[0]) < batch_size:
            real_loss = self.base_algorithm(observation, ref_policy, self.dynamical_model)
            return real_loss
        if not ref_policy:
            sim_observation_list = []
            for head, sim_m in enumerate(self.sim_memory_list):
                sim_observation_list.append(self.sim_memory_list[head].sample_batch(batch_size)[0])
            sim_loss = self.base_algorithm(sim_observation_list, ref_policy, self.dynamical_model)
            real_loss = self.base_algorithm(observation, ref_policy, self.dynamical_model)
            return sim_loss + real_loss
        else:
            real_loss = self.base_algorithm(observation, ref_policy, self.dynamical_model)
            return real_loss

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
        for head in range(1):
            self.simulate(state=self._sample_initial_states(), policy=self.q_list[head], head_ind=head)
        super().update()

    def reset(self):
        """Reset base algorithm."""
        self.count += 1
        if (self.count % self.refresh_interval) == 0:
            for head in range(1):
                self.sim_memory_list[head].reset()
        super().reset()
