#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import collections
import time
import pdb

State = collections.namedtuple('State', ['q', 'q_d'])

class Grid():
    def __init__(self, size, init_value = 0, step_size = 0.1):
        self.size = size
        self.step_size = step_size
        self.grid_size = int(self.size / self.step_size)
        self.grid = np.ones((self.grid_size, self.grid_size)) * init_value
        self.init_value = init_value

    def getRange(self):
        return np.arange(-int(self.size/2), int(self.size/2), self.step_size)

    def getIndex(self, position):
        return int(np.round((position + self.size/2) / self.step_size))

    def __getitem__(self, key):
        x1 = int(np.floor((key[0] + self.size/2) / self.step_size))
        x = (key[0] + self.size/2) / self.step_size
        x2 = int(np.ceil((key[0] + self.size/2) / self.step_size))
        y1 = int(np.floor((key[1] + self.size/2) / self.step_size))
        y = (key[1] + self.size/2) / self.step_size
        y2 = int(np.ceil((key[1] + self.size/2) / self.step_size))
        
        if not (0 <= x1 < x2 < self.grid_size and 0 <= y1 < y2 < self.grid_size):
            # print("Out of bound: key = " + str(key[0]) + ", " + str(key[1]) + ")")
            return self.init_value

        return bilinear_interpolation(x, y, [
            (x1, y1, self.grid[y1][x1]),
            (x1, y2, self.grid[y2][x1]),
            (x2, y1, self.grid[y1][x2]),
            (x2, y2, self.grid[y2][x2])
        ])

    def __setitem__(self, key, value):
        x = self.getIndex(key[0])
        y = self.getIndex(key[1])
        self.grid[y][x] = value

# https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def cost_to_go(state, u, tolerance = 0.1):
    if abs(state.q) < tolerance and abs(state.q_d) < tolerance:
        return 0
    else:
        return 1

def transition_func(u, state, t, step_size = 0.1):
    q = state.q + state.q_d*t + u*0.5*t**2
    q_d = state.q_d + u*t
    return State(q, q_d)

def roundToSize(value, step_size = 1):
    rounded = np.round(value / step_size) * step_size
    # if abs(rounded - value) > 0.45*step_size:
        # print(str(value) + " rounded to " + str(rounded) + ", step size = " + str(step_size))
    return rounded

def runValueIteration():
    INITIAL_COST = 1e3
    STATE_SIZE = 20
    STATE_STEP = 0.1
    TIME_STEP = 0.1

    cost_grid = Grid(STATE_SIZE, init_value = INITIAL_COST, step_size = STATE_STEP)
    controller_grid = Grid(STATE_SIZE, init_value = 0, step_size = STATE_STEP)
    cost_grid[(0, 0)] = 0

    fig, (p1, p2) = plt.subplots(1, 2)
    p1.set_title("Cost")
    cost_plot = p1.imshow(cost_grid.grid, cmap='rainbow', vmin = 0, vmax = 100, origin='lower')
    p2.set_title("Controller")
    controller_plot = p2.imshow(controller_grid.grid, cmap='Set1', vmin = -1, vmax = 1, origin='lower', interpolation='nearest')

    for iteration in range(100):
        t1 = time.time()
        for q in cost_grid.getRange():
            for q_d in cost_grid.getRange():
                for u in [0, -1, 1]:
                    next_state = transition_func(u, State(q, q_d), TIME_STEP, step_size = STATE_STEP)
                    next_state_cost = cost_grid[next_state]
                    cost = cost_to_go(State(q, q_d), u, tolerance = STATE_STEP/2) + next_state_cost
                    if cost < cost_grid[(q, q_d)]:
                        cost_grid[(q, q_d)] = cost
                        controller_grid[(q, q_d)] = u
        print("Time for 1 iteration : " + str(time.time() - t1) + "s")

        print("Iteration = " + str(iteration))
        cost_plot.set_data(cost_grid.grid)
        cost_plot.set_clim()
        controller_plot.set_data(controller_grid.grid)
        plt.pause(0.1)

        # pdb.set_trace()

    pdb.set_trace()

runValueIteration()
