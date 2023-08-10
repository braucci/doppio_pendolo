import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Parameters
g = 9.81  # gravity
L1 = 1.0  # length of pendulum 1
L2 = 1.0  # length of pendulum 2
m1 = 1.0  # mass of pendulum 1
m2 = 1.0  # mass of pendulum 2

def dy(y, t, L1, L2, m1, m2):
    delta = y[2] - y[0]
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    den2 = (L2 / L1) * den1

    dy0 = y[1]
    dy1 = ((m2 * L2 * y[3] ** 2 * np.sin(delta) * np.cos(delta) +
            m2 * g * np.sin(y[2]) * np.cos(delta) +
            m2 * L2 * y[3] ** 2 * np.sin(delta) -
            (m1 + m2) * g * np.sin(y[0])) / den1)
    dy2 = y[3]
    dy3 = ((-L1 / L2 * y[1] ** 2 * np.sin(delta) * np.cos(delta) +
            (m1 + m2) * g * np.sin(y[0]) * np.cos(delta) -
            (m1 + m2) * L1 * y[1] ** 2 * np.sin(delta) -
            (m1 + m2) * g * np.sin(y[2])) / den2)

    return [dy0, dy1, dy2, dy3]

# Initial conditions
y0 = [np.pi / 4, 0, np.pi / 4, 0]

# Time array
t = np.linspace(0, 10, 500)

# Solve the differential equation
y = odeint(dy, y0, t, args=(L1, L2, m1, m2))

# Convert to Cartesian coordinates
x1 = L1 * np.sin(y[:, 0])
y1 = -L1 * np.cos(y[:, 0])
x2 = x1 + L2 * np.sin(y[:, 2])
y2 = y1 - L2 * np.cos(y[:, 2])

# Animate
fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)

# Salva l'animazione come GIF
writer = PillowWriter(fps=20)
ani.save("double_pendulum.gif", writer=writer)

plt.show()
