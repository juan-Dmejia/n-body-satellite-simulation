import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from orbit_env import SatelliteOrbitEnv


def main():
    parser = argparse.ArgumentParser(description="Train a PPO satellite launch agent.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="satellite_ppo")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    env = SatelliteOrbitEnv()
    check_env(env)

    if args.resume:
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.999,
            verbose=1,
            seed=args.seed,
        )

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=args.resume is None,    
    )
    model.save(args.output)
    env.close()


if __name__ == "__main__":
    main()