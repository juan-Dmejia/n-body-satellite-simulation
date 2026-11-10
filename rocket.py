import numpy as np

class RocketController:
    def __init__(self, rocket_body: str, planet: str):
        self.rocket_body = rocket_body
        self.planet = planet

        self.propulsion = np.array([0.0, 0.0, 0.0])
        self.current_throttle = 0.0
        self.current_angle_vector = np.array([0.0, 1.0, 0.0])  # Initial pointing up

    def change_propulsion_angle(self, direction_vector: np.ndarray):
        # Normalize and update the direction the engine is pointing
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            self.current_angle_vector = direction_vector / norm

    def get_propulsion_force(self):
        return self.current_angle_vector * self.current_throttle