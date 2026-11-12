import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine import Engine, G
from rocket import RocketController


EARTH_MASS = 5.97219e24
EARTH_RADIUS = 6_371_000.0
ROCKET_MASS = 5_000.0
DRY_MASS = 1_000.0
INITIAL_FUEL_MASS = ROCKET_MASS - DRY_MASS
SPECIFIC_IMPULSE = 650.0
STANDARD_GRAVITY = 9.80665


class SatelliteOrbitEnv(gym.Env):
    """Train continuous throttle and steering for a planar Earth orbit."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 2.0,
        max_steps: int = 10_000,
        max_thrust: float = 110_000.0,
        target_altitude: float = 200_000.0,
    ):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.max_thrust = max_thrust
        self.target_radius = EARTH_RADIUS + target_altitude
        self.mu = G * EARTH_MASS
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -2.0, -2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.engine = Engine()
        self.controller = None
        self.steps = 0
        self.fuel_mass = INITIAL_FUEL_MASS
        self.previous_orbit_error = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.engine = Engine()
        self.engine.add_body("earth", np.zeros(3), EARTH_MASS)
        self.engine.add_body(
            "rocket", np.array([EARTH_RADIUS, 0.0, 0.0]), ROCKET_MASS
        )
        self.controller = RocketController("rocket", "earth")
        self.engine.controller = self.controller
        self.steps = 0
        self.fuel_mass = INITIAL_FUEL_MASS
        self.previous_orbit_error = self._orbit_error()
        return self._get_observation(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(f"Action is outside the action space: {action}")

        throttle = (float(action[0]) + 1.0) / 2.0
        steering_angle = float(action[1]) * (np.pi / 2.0)
        radial, tangent, _, _, _ = self._relative_state()
        thrust_direction = (
            np.cos(steering_angle) * radial + np.sin(steering_angle) * tangent
        )
        self.controller.change_propulsion_angle(thrust_direction)
        requested_fuel = (
            throttle
            * self.max_thrust
            / (SPECIFIC_IMPULSE * STANDARD_GRAVITY)
            * self.dt
        )
        fuel_burned = min(self.fuel_mass, requested_fuel)
        effective_throttle = throttle
        if requested_fuel > 0.0:
            effective_throttle *= fuel_burned / requested_fuel
        self.engine.bodies["rocket"][6] = self.fuel_mass + DRY_MASS - fuel_burned / 2.0
        self.controller.current_throttle = effective_throttle * self.max_thrust
        self.engine.step(self.dt)
        self.fuel_mass -= fuel_burned
        self.engine.bodies["rocket"][6] = self.fuel_mass + DRY_MASS
        self.steps += 1

        observation = self._get_observation()
        info = self._get_info()
        orbit_error = self._orbit_error()
        reward = 10.0 * (self.previous_orbit_error - orbit_error)
        reward -= 0.001 + 0.1 * fuel_burned / INITIAL_FUEL_MASS
        self.previous_orbit_error = orbit_error

        terminated = False
        if not np.all(np.isfinite(observation)):
            terminated = True
            reward -= 100.0
            info["termination_reason"] = "invalid_state"
        elif info["altitude"] <= 0.0:
            terminated = True
            reward -= 100.0
            info["termination_reason"] = "impact"
        elif info["specific_energy"] >= 0.0:
            terminated = True
            reward -= 25.0
            info["termination_reason"] = "escape"
        elif self._is_target_orbit(info):
            terminated = True
            reward += 100.0
            info["success"] = True
            info["termination_reason"] = "target_orbit"

        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            info["termination_reason"] = "time_limit"
        return observation, float(reward), terminated, truncated, info

    def _relative_state(self):
        earth = self.engine.bodies["earth"]
        rocket = self.engine.bodies["rocket"]
        relative_position = rocket[:3] - earth[:3]
        relative_velocity = rocket[3:6] - earth[3:6]
        radius = float(np.linalg.norm(relative_position))
        radial = relative_position / radius
        tangent = np.array([-radial[1], radial[0], 0.0])
        radial_speed = float(np.dot(relative_velocity, radial))
        tangential_speed = float(np.dot(relative_velocity, tangent))
        return radial, tangent, radius, radial_speed, tangential_speed

    def _orbit_error(self):
        _, _, radius, radial_speed, tangential_speed = self._relative_state()
        circular_speed = np.sqrt(self.mu / radius)
        return (
            abs(radius - self.target_radius) / self.target_radius
            + abs(radial_speed) / circular_speed
            + abs(tangential_speed - circular_speed) / circular_speed
        )

    def _get_observation(self):
        _, _, radius, radial_speed, tangential_speed = self._relative_state()
        escape_speed = np.sqrt(2.0 * self.mu / radius)
        return np.array(
            [
                (radius - EARTH_RADIUS) / EARTH_RADIUS,
                radial_speed / escape_speed,
                tangential_speed / escape_speed,
                self.fuel_mass / INITIAL_FUEL_MASS,
                self.steps / self.max_steps,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        _, _, radius, radial_speed, tangential_speed = self._relative_state()
        relative_position = (
            self.engine.bodies["rocket"][:3] - self.engine.bodies["earth"][:3]
        )
        relative_velocity = (
            self.engine.bodies["rocket"][3:6] - self.engine.bodies["earth"][3:6]
        )
        specific_energy = float(
            0.5 * np.dot(relative_velocity, relative_velocity) - self.mu / radius
        )
        angular_momentum = np.cross(relative_position, relative_velocity)
        eccentricity_vector = (
            np.cross(relative_velocity, angular_momentum) / self.mu
            - relative_position / radius
        )
        eccentricity = float(np.linalg.norm(eccentricity_vector))
        info = {
            "altitude": radius - EARTH_RADIUS,
            "radial_speed": radial_speed,
            "tangential_speed": tangential_speed,
            "specific_energy": specific_energy,
            "eccentricity": eccentricity,
            "fuel_remaining": self.fuel_mass,
            "success": False,
        }
        if specific_energy < 0.0:
            semi_major_axis = -self.mu / (2.0 * specific_energy)
            info["perigee_altitude"] = semi_major_axis * (1.0 - eccentricity) - EARTH_RADIUS
            info["apogee_altitude"] = semi_major_axis * (1.0 + eccentricity) - EARTH_RADIUS
        return info

    @staticmethod
    def _is_target_orbit(info):
        return (
            info["specific_energy"] < 0.0
            and info["eccentricity"] < 0.08
            and info["perigee_altitude"] >= 120_000.0
            and info["apogee_altitude"] <= 500_000.0
        )