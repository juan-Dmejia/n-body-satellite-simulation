from engine import Engine
import numpy as np

M_EARTH = 5.97219e24
M_MOON = 7.34767e22

DIST = 384400


eng = Engine()

eng.add_body('earth', np.array([0.0,0.0,0.0]), M_EARTH)
eng.add_body('moon', np.array([DIST, 0.0, 0.0]), M_MOON, 'earth')

print(eng.bodies)