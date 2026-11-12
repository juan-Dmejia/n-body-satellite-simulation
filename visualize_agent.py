import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import PPO
from ursina import EditorCamera, Entity, Text, Ursina, color, window

from orbit_env import EARTH_RADIUS, SatelliteOrbitEnv


SCALE_FACTOR = 1e-6
viewer = None


class AgentViewer:
    def __init__(self, model, seed: int, steps_per_second: float):
        self.model = model
        self.seed = seed
        self.env = SatelliteOrbitEnv()
        self.steps_per_second = steps_per_second
        self.step_interval = 1.0 / steps_per_second
        self.accumulator = 0.0
        self.last_frame_time = perf_counter()
        self.paused = False
        self.finished = False
        self.step_count = 0
        self.observation, self.info = self.env.reset(seed=seed)

        window.title = "Satellite PPO Flight"
        self.earth = Entity(
            model="sphere",
            color=color.azure,
            scale=2.0 * EARTH_RADIUS * SCALE_FACTOR,
        )
        self.rocket = Entity(
            model="cube",
            color=color.red,
            scale=(0.16, 0.16, 0.34),
        )
        self.status = Text(
            text="",
            position=(-0.86, 0.46),
            origin=(-0.5, 0.5),
            scale=0.8,
            background=True,
        )
        self.target_markers = self._make_target_markers()
        self.trail = []
        self._sync_scene()
        self._update_status()

    def _make_target_markers(self):
        radius = self.env.target_radius * SCALE_FACTOR
        markers = []
        for angle in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
            markers.append(
                Entity(
                    model="sphere",
                    color=color.yellow,
                    scale=0.035,
                    position=(radius * np.cos(angle), radius * np.sin(angle), 0.0),
                )
            )
        return markers

    def update(self):
        now = perf_counter()
        self.accumulator += min(now - self.last_frame_time, 0.1)
        self.last_frame_time = now

        if self.paused or self.finished:
            return

        while self.accumulator >= self.step_interval and not self.finished:
            self.accumulator -= self.step_interval
            self._advance_policy()

    def _advance_policy(self):
        action, _ = self.model.predict(self.observation, deterministic=True)
        self.observation, _, terminated, truncated, self.info = self.env.step(action)
        self.step_count += 1
        self._sync_scene()

        if self.step_count % 8 == 0 or terminated or truncated:
            self.trail.append(
                Entity(
                    model="sphere",
                    color=color.orange,
                    scale=0.045,
                    position=self.rocket.position,
                )
            )

        self.finished = terminated or truncated
        self._update_status()
        if self.finished:
            print(
                f"Episode ended: {self.info.get('termination_reason', 'unknown')} | "
                f"success={self.info['success']} | "
                f"altitude={self.info['altitude'] / 1000.0:.1f} km | "
                f"eccentricity={self.info['eccentricity']:.4f} | "
                f"fuel={self.info['fuel_remaining']:.1f} kg"
            )

    def _sync_scene(self):
        earth_position = self.env.engine.bodies["earth"][:3] * SCALE_FACTOR
        rocket_position = self.env.engine.bodies["rocket"][:3] * SCALE_FACTOR
        self.earth.position = tuple(earth_position)
        self.rocket.position = tuple(rocket_position)

    def _update_status(self):
        state = self.env.engine.bodies["rocket"]
        earth = self.env.engine.bodies["earth"]
        radius = np.linalg.norm(state[:3] - earth[:3])
        speed = np.linalg.norm(state[3:6] - earth[3:6])
        result = "SUCCESS: target orbit" if self.info["success"] else "In flight"
        if self.finished and not self.info["success"]:
            result = f"Ended: {self.info.get('termination_reason', 'unknown')}"
        pause_label = "paused" if self.paused else "running"
        self.status.text = (
            f"{result}  |  {pause_label}\n"
            f"Step {self.step_count}/{self.env.max_steps}  "
            f"Altitude {self.info['altitude'] / 1000.0:,.1f} km  "
            f"Speed {speed / 1000.0:,.2f} km/s  "
            f"Fuel {self.info['fuel_remaining']:,.0f} kg\n"
            f"Eccentricity {self.info['eccentricity']:.4f}  "
            f"Target altitude {self.env.target_radius - EARTH_RADIUS:,.0f} m\n"
            "Space: pause/resume    R: restart    Mouse: orbit camera"
        )

    def input(self, key):
        if key == "space" and not self.finished:
            self.paused = not self.paused
            self._update_status()
        elif key == "r":
            self.reset()

    def reset(self):
        for marker in self.trail:
            marker.disable()
            del marker
        self.trail.clear()
        self.observation, self.info = self.env.reset(seed=self.seed)
        self.accumulator = 0.0
        self.last_frame_time = perf_counter()
        self.paused = False
        self.finished = False
        self.step_count = 0
        self._sync_scene()
        self._update_status()


def update():
    if viewer is not None:
        viewer.update()


def input(key):
    if viewer is not None:
        viewer.input(key)


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained satellite PPO policy.")
    parser.add_argument("--model", default="satellite_ppo.zip")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps-per-second", type=float, default=30.0)
    args = parser.parse_args()

    if args.steps_per_second <= 0.0:
        parser.error("--steps-per-second must be greater than zero")
    model_path = Path(args.model)
    if not model_path.is_file() and Path(f"{args.model}.zip").is_file():
        model_path = Path(f"{args.model}.zip")
    if not model_path.is_file():
        parser.error(f"Model not found: {args.model}. Train an agent first.")

    global viewer
    model = PPO.load(str(model_path))
    app = Ursina()
    EditorCamera()
    viewer = AgentViewer(model, args.seed, args.steps_per_second)
    app.run()
    viewer.env.close()


if __name__ == "__main__":
    main()
