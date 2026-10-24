from engine import Engine
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

# create physics engine and set delta time
engine = Engine()
dt = 60.0

# add celestial bodies to engine
engine.add_body('earth', np.array([0.0,0.0,0.0]), M_EARTH)
engine.add_body('moon', np.array([DIST, 0.0, 0.0]), M_MOON, 'earth')

# Create visual 3D spheres
earth_mesh = Entity(model="sphere", color=color.green, scale=3.0)
moon_mesh = Entity(model="sphere", color=color.light_gray, scale=1.0)

def update():
    # run an engine step 10 times for every frame rendered by Ursina
    # simulates a 'fast-forward' effect so we can actually see changes
    for i in range(10):
        engine.step(dt)

    # live terminal update
    moon_pos = engine.bodies['moon'][:3]
    earth_pos = engine.bodies['earth'][:3]
    dist_vector = moon_pos - earth_pos
    current_distance = np.linalg.norm(dist_vector)

    # print coords to terminal
    print(
        f"Moon: ({moon_pos[0]:,.0f}, {moon_pos[1]:,.0f}, {moon_pos[2]:,.0f}) | "
        f"Earth: ({earth_pos[0]:,.0f}, {earth_pos[1]:,.0f}, {earth_pos[2]:,.0f}) | "
        f"Dist: {current_distance:,.0f}m",
        end="\r",
        flush=True,
    )


    # Apply scaling and update the 3D visual objects' positions
    earth_mesh.position = tuple(earth_pos * SCALE_FACTOR)
    moon_mesh.position = tuple(moon_pos * SCALE_FACTOR)

# Ursina looks for an update() function to run every frame
app.run()