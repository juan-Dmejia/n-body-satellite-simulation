from engine import Engine
from rocket import RocketController
from ursina import *
import time
import numpy as np

# set up the 3D scene
app = Ursina()
# Free-roaming camera (Right-click + drag to rotate, scroll to zoom)
EditorCamera()

# Scalar Constants
M_EARTH = 5.97219e24
M_MOON = 7.34767e22
DIST = 384400000    # distance in meters
# Scale factor to convert massive meters down to readable 3D graphics coordinates
SCALE_FACTOR = 1e-7
ROCKET_DIST = 6371000  # Earth radius

# create physics engine and set delta time
engine = Engine()
dt = 2.0



# add celestial bodies to engine
engine.add_body('earth', np.array([0.0,0.0,0.0]), M_EARTH)
engine.add_body('moon', np.array([DIST, 0.0, 0.0]), M_MOON, orbit_body = 'earth')

# add rocket and create controller
engine.add_body('rocket', np.array([ROCKET_DIST, 0.0, 0.0]), mass=5000.0)
engine.controller = RocketController('rocket', 'earth')


# Create visual 3D spheres
earth_mesh = Entity(model="sphere", color=color.green, scale=3.0)
moon_mesh = Entity(model="sphere", color=color.light_gray, scale=1.0)
rocket_mesh = Entity(model="cube", color=color.red, scale=.5)

flight_time = 0.0

time_vertical = 20.0
vertical_propulsion = 110000
time_turning = 80.0
turning_propulsion = 85000


def update():
    global flight_time
    dt_physics = 2.0  # or whatever your step size is
    flight_time += dt_physics

    # --- SIMPLE OPEN-LOOP FLIGHT PROFILE ---
    if flight_time < time_vertical:
        # Phase 1: Lift off vertically to clear the pad/lower atmosphere
        engine.controller.change_propulsion_angle(np.array([0.0, 1.0, 0.0]))
        # Full throttle
        engine.controller.current_throttle = vertical_propulsion  # adjust to your mass/thrust scale

    elif time_vertical <= flight_time < time_turning + time_vertical:
        # Phase 2: The Gravity Turn (pitching sideways to build horizontal velocity)
        # Slowly tilt the angle vector from vertical [0, 1, 0] towards horizontal [1, 0, 0]
        tilt_progress = (flight_time - time_vertical) / time_turning  # 0 to 1 over 60 seconds
        # Example interpolation (or a simple curve):
        x_component = tilt_progress
        y_component = 1.0 - tilt_progress
        engine.controller.change_propulsion_angle(
            np.array([x_component, y_component, 0.0])
        )
        engine.controller.current_throttle = turning_propulsion
    else:
        # Phase 3: Main engine cutoff (MECO) - coast and check your orbit!
        engine.controller.current_throttle = 0.0


    # run an engine step 10 times for every frame rendered by Ursina
    # simulates a 'fast-forward' effect so we can actually see changes
    for i in range(10):
        engine.step(dt)

    # live terminal update
    moon_pos = engine.bodies['moon'][:3]
    earth_pos = engine.bodies['earth'][:3]
    rocket_pos = engine.bodies['rocket'][:3]
    rocket_dist = np.linalg.norm(rocket_pos - earth_pos)
    dist_vector = moon_pos - earth_pos
    current_distance = np.linalg.norm(dist_vector)

    # print coords to terminal
    print(
        f"Rocket: ({rocket_pos[0]:,.0f}, {rocket_pos[1]:,.0f}, {rocket_pos[2]:,.0f}) | "
        f"Earth: ({earth_pos[0]:,.0f}, {earth_pos[1]:,.0f}, {earth_pos[2]:,.0f}) | "
        f"Dist: {current_distance:,.0f}m | "
        f"Rocket: {rocket_dist:,.0f}m",
        end="\r",
        flush=True,
    )


    # Apply scaling and update the 3D visual objects' positions
    earth_mesh.position = tuple(earth_pos * SCALE_FACTOR)
    moon_mesh.position = tuple(moon_pos * SCALE_FACTOR)
    rocket_mesh.position = tuple(rocket_pos * SCALE_FACTOR)

# Ursina looks for an update() function to run every frame
app.run()