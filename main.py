from engine import Engine
import time
import numpy as np

# Simple test case of engine, I will probably work on graphics next so we at least get a 3d visualization of it
# if nothing else, I think that would still count as a completed project

M_EARTH = 5.97219e24
M_MOON = 7.34767e22

DIST = 384400


eng = Engine()
dt = 1.0

eng.add_body('earth', np.array([0.0,0.0,0.0]), M_EARTH)
eng.add_body('moon', np.array([DIST, 0.0, 0.0]), M_MOON, 'earth')

try:
    while True:
        eng.step(dt)

        # live terminal update
        moon_pos = eng.bodies['moon'][:3]
        # Calculate current distance from the origin (Earth)
        current_distance = np.linalg.norm(moon_pos)

        print(
            f"Moon Coords -> X: {moon_pos[0]:,.1f}m | Y: {moon_pos[1]:,.1f}m | Z: {moon_pos[2]:,.1f}m | current distance: {current_distance:,.1f}m",
            end="\r",
            flush = True,
        )

        # throttle to limit cpu speed
        time.sleep(0.01)

except KeyboardInterrupt:
    # CTRL + C will stop the loop
    print("\nSimulation Stopped.") 