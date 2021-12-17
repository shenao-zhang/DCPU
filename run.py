"""Script that demonstrates how to use BPTT using hallucination."""

import argparse
import importlib
import yaml

from rllib.environment import GymEnvironment
from rllib.model import TransformedModel
from rllib.util import set_random_seed
from rllib.util.training.agent_training import evaluate_agent, train_agent

def parse_config_file(file_dir):
    """Parse configuration file."""
    with open(file_dir, "r") as file:
        args = yaml.safe_load(file)
    return args


def main(args):
    """Run experiment."""
    set_random_seed(args.seed)
    env_config = parse_config_file(args.env_config_file)

    environment = GymEnvironment(
        env_config["name"], ctrl_cost_weight=env_config["action_cost"], seed=args.seed
    )
    reward_model = environment.env.reward_model()
    dynamical_model = TransformedModel.default(environment)
    kwargs = parse_config_file(args.agent_config_file)

    agent = getattr(
        importlib.import_module("rllib.agent"), f"{args.agent}Agent"
    ).default(
        environment=environment,
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        thompson_sampling=True,
        **kwargs,
    )
    train_agent(
        agent=agent,
        environment=environment,
        max_steps=env_config["max_steps"],
        num_episodes=args.train_episodes,
        render=False,
        print_frequency=1,
    )

    evaluate_agent(
        agent=agent,
        environment=environment,
        max_steps=env_config["max_steps"],
        num_episodes=args.test_episodes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for DCPU.")
    parser.add_argument(
        "--agent",
        type=str,
        default="DCPU"
    )
    parser.add_argument("--agent-config-file", type=str, default="./exps/mujoco/config/agents/dcpu.yaml")
    parser.add_argument(
        "--env-config-file", type=str, default="./exps/mujoco/config/envs/half-cheetah.yaml"
    )

    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-episodes", type=int, default=800)
    parser.add_argument("--test-episodes", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    main(parser.parse_args())

