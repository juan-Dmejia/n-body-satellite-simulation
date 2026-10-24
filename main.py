from engine import Engine
import time
import numpy as np

# Simple test case of engine, I will probably work on graphics next so we at least get a 3d visualization of it
# if nothing else, I think that would still count as a completed project

M_EARTH = 5.97219e24
M_MOON = 7.34767e22

DIST = 384400000


engine = Engine()
dt = 60.0

engine.add_body('earth', np.array([0.0,0.0,0.0]), M_EARTH)
engine.add_body('moon', np.array([DIST, 0.0, 0.0]), M_MOON)

try:
    while True:
        engine.step(dt)

        # live terminal update
        moon_pos = engine.bodies['moon'][:3]
        earth_pos = engine.bodies['earth'][:3]
        dist = moon_pos - earth_pos
        # Calculate current distance from earth
        current_distance = np.linalg.norm(dist)

        print(
            f"Moon: ({moon_pos[0]:,.0f}, {moon_pos[1]:,.0f}, {moon_pos[2]:,.0f}) | "
            f"Earth: ({earth_pos[0]:,.0f}, {earth_pos[1]:,.0f}, {earth_pos[2]:,.0f}) | "
            f"Dist: {current_distance:,.0f}m",
            end="\r",
            flush=True,
        )

        # throttle to limit cpu speed
        time.sleep(0.01)

except KeyboardInterrupt:
    # CTRL + C will stop the loop
    print("\nSimulation Stopped.") 