import numpy as np

G = 6.6743e-11

# dictionary: {key: value}
# {string_name: np.array([])}
# i.e {earth: [x, y, z, vx, vy, vz, m], moon: [x, y, z, vx, vy, vz, m]}

class Engine:
    def __init__(self):
        self.bodies = {}

    def add_body(self, name: str, coords: np.ndarray, mass: float, orbit_body: str = None):
        velocity = np.array([0.0, 0.0, 0.0])

        
        if orbit_body is not None:
            # sets a velocity vector perpendicular to central body to 'kickstart' the orbit

            orbit_state = self.bodies[orbit_body]

            central_mass = orbit_state[6]

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
        # returns force of attraction in Newtons between two bodies

        # take x,y,z coordinates of both bodies by splicing their 'state' vector
        pos1 = body1[:3]
        pos2 = body2[:3]
        # do the same for the mass
        mass1 = body1[6]
        mass2 = body2[6]

        # get vector ([x,y,z]) between the two celestial bodies
        dist_vector = pos2 - pos1

        # get norm (magnitude) of distance vector, (numpy has a built in function so we don't have to do math)
        dist_norm = np.linalg.norm(dist_vector) 

        # get unit vector (vector in same direction as distance_vector but with magnitude of 1)
        dist_unit_vector = dist_vector / dist_norm 

        # compute force_gravity_vector ([fx, fy, fz]) with physics formula
        # we multiply it by the dist_unit_vector to turn it from a magnitude into a vector
        force_gravity = (G * (mass1 * mass2) / (dist_norm**2)) * dist_unit_vector
        return force_gravity

    def apply_forces(self, dt):
        # for every celestial body in the engine, we will compute its force_gravity with every other body in the engine
        # then convert the force into their acceleration

        # name1 = string, state1 = np.array of that bodies stats
        for name1, state1 in self.bodies.items():
            # get mass of body we are checking and set f_net = 0
            mass = state1[6]
            force_net = 0.0

            for name2, state2 in self.bodies.items():
                # skip if it's the same body
                if name1 == name2:
                    continue

                # get force_gravity using class method, then add to force_net
                # force_net is a vector ([fx, fy, fz])
                force_net += self.get_force_gravity(state1, state2)

            # compute acceleration with physics formula and add it to our bodies velocity stats
            acceleration_vector = force_net / mass
            state1[3:6] += acceleration_vector * dt

    def update_positions(self, dt):
        # once forces are applied, this method will update each bodies x,y,z coordinates by adding their new velocity vectors
        for name, state in self.bodies.items():
            # OPTIONAL (off for now): skip updating earth's position for now so it acts as an anchor
            #if name == 'earth':
            #    continue
            
            state[:3] += state[3:6] * dt

    def step(self, dt):
        # this method will run every 'tick'

        self.apply_forces(dt) # first compute force -> acceleration -> velocity
        self.update_positions(dt) # then update position by adding velocity for that tick

    def burn_fuel(self, name: str, fuel_amount: float, thrust_direction: np.ndarray, dt: float):
        # Get the body's state array
        state = self.bodies[name]
        
        # 1. Reduce the total mass (index 6) by how much fuel was burned
        state[6] -= fuel_amount
        
        # 2. Calculate acceleration from thrust (Force / Mass)
        thrust_force = thrust_direction * 100000.0  # arbitrary engine power scaling
        acceleration_vector = thrust_force / state[6]
        
        # 3. Add that acceleration to the rocket's velocity (indices 3, 4, 5)
        state[3:6] += acceleration_vector * dt