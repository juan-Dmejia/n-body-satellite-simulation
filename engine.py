import numpy as np

G = 6.6743e-11

class Engine:
    def __init__(self):
        self.bodies = {}

    def add_body(self, name: str, coords: np.ndarray, mass: float, orbit_body: str = None):
        velocity = np.array([0.0,0.0,0.0])

        
        if orbit_body is not None:
            orbit_state = self.bodies[orbit_body]

            central_mass = orbit_state[6:]

            dist_vector = coords - orbit_state[:3]
            dist_magnitude = np.linalg.norm(dist_vector)

            speed = np.sqrt((G * central_mass) / dist_magnitude)

            z_axis = np.array([0.0, 0.0, 1.0])
            tangent_vector = np.cross(z_axis, dist_vector)

            unit_tangent = tangent_vector / np.linalg.norm(tangent_vector)
            velocity = speed * unit_tangent

        self.bodies[name] = np.array([*coords, *velocity, mass])

    def delete_body(self, name: str):
        del self.bodies[name]

    def get_force_gravity(self, body1: np.ndarray, body2: np.ndarray):
        pos1 = body1[:3]
        pos2 = body2[:3]
        mass1 = body1[6:]
        mass2 = body2[6:]

        dist_vector = pos1 - pos2 # vector between two bodies
        dist_mag = np.linalg.norm(dist_vector) # magnitude of vector bet. the bodies

        unit_vector = dist_vector / dist_mag    # direction vector for force

        force_gravity = (G * (mass1 * mass2) / (dist_mag**2)) * unit_vector # vector for force due to gravity
        return force_gravity

    def apply_forces(self, dt):
        for name1, state1 in self.bodies:

            mass = state1[6:]
            force_net = 0.0

            for name2, state2 in self.bodies:

                if name1 == name2:
                    continue
                
                force_net += self.get_force_gravity(state1, state2)

            acceleration_vector =  force_net / mass
            state1[3:6] += acceleration_vector * dt

    def update_positions(self, dt):
        for name, state in self.bodies:
            state[:3] += state[3:6] * dt

    def step(self, dt):
        self.apply_forces(dt)
        self.update_positions(dt)
        for name, state in self.bodies:
            print(f'{name}, state: {state}', end = "\r", flush = True)

