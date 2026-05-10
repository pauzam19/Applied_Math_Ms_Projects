# =================================================
# ADVECTION-DIFFUSION SIMULATION: THREE CASES
# Case 1: No diffusion
# Case 2: With diffusion
# Case 3: With diffusion and obstacle
#=================================================
# LIBRARIES
#=================================================

import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# DOMAIN (Rectangular and finite gridpoints, as mesh generation)
# ==================================================

Lx, Ly = 2.0, 2.0        # physical size of the domain: 0 <= x <= Lx, 0 <= y <= Ly
Nx, Ny = 100, 100        # number of grid points in x and y

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

# X[j, i] and Y[j, i] give the physical coordinates of grid point (i, j)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]

# ==================================================
# INITIAL CONDITION (smoke or temperature blob is centered at (x0, y0) = (0.5,1.0) with width sigma = 0.1)
# ==================================================

x0, y0 = 0.5, 1.0        # center of the initial smoke/temperature blob
sigma = 0.15              # width of the Gaussian blob


def initial_condition():
    """Return the initial Gaussian blob u(x,y,0)."""
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / sigma**2)


# ==================================================
# VELOCITY FIELD AND NUMERICAL PARAMETERS
# ==================================================

# Constant prescribed velocity field v = (a,b)
a = 1.0                  # velocity in the x-direction
b = 0.0                  # velocity in the y-direction

dt = 0.0005              # time step size
Nt = 700               # number of time steps
T = dt * Nt              # final time



# ==================================================
# OBSTACLE: SIMPLIFIED AIRFOIL
# ==================================================

xc, yc = 1.0, 1.0        # center of the circular obstacle
r = 0.2                  # radius of the circular obstacle

# Boolean mask: True inside the obstacle, False outside
mask = (X - xc)**2 + (Y - yc)**2 < r**2

# ==================================================
# CASE DEFINITIONS
# ==================================================

cases = [
    {
        "name": "Case 1: No diffusion",
        "D": 0.0,
        "use_obstacle": False,
    },
    {
        "name": "Case 2: With diffusion",
        "D": 0.01,
        "use_obstacle": False,
    },
    {
        "name": "Case 3: With diffusion + obstacle",
        "D": 0.01,
        "use_obstacle": True,
    },
]

# ==================================================
# TIME STEP FUNCTION (Define the a function that advances the soultion by one time step using finite difference approximations for the derivatives)
# ==================================================


def step(u, D, use_obstacle):
    """
    Advance the solution by one time step using:
    - Forward Euler in time
    - Upwind finite differences for advection
    - Central finite differences for diffusion

    Array indexing convention:
    u[j, i] corresponds to u(x_i, y_j).
    Therefore:
    - i is the x-index
    - j is the y-index
    """

    u_new = u.copy()

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):

            # If the obstacle is active, force concentration to zero inside it.
            if use_obstacle and mask[j, i]:
                u_new[j, i] = 0.0
                continue

            # ---------- First derivatives: upwind scheme ----------
            # For positive a, information comes from the left.
            # For negative a, information comes from the right.
            if a >= 0:
                du_dx = (u[j, i] - u[j, i - 1]) / dx
            else:
                du_dx = (u[j, i + 1] - u[j, i]) / dx

            # For positive b, information comes from below.
            # For negative b, information comes from above.
            if b >= 0:
                du_dy = (u[j, i] - u[j - 1, i]) / dy
            else:
                du_dy = (u[j + 1, i] - u[j, i]) / dy

            # ---------- Second derivatives: central differences ----------
            d2u_dx2 = (u[j, i + 1] - 2.0 * u[j, i] + u[j, i - 1]) / dx**2
            d2u_dy2 = (u[j + 1, i] - 2.0 * u[j, i] + u[j - 1, i]) / dy**2

            # ---------- Advection-diffusion equation ----------
            # PDE: u_t + a*u_x + b*u_y = D*(u_xx + u_yy)
            u_new[j, i] = (
                u[j, i]
                - dt * (a * du_dx + b * du_dy)
                + dt * D * (d2u_dx2 + d2u_dy2)
            )

    # Enforce the obstacle condition again after the update.
    if use_obstacle:
        u_new[mask] = 0.0

    return u_new


# ==================================================
# SIMULATION
# ==================================================


def run_simulation(D, use_obstacle):
    """Run one simulation case and return the initial and final fields."""

    u_initial = initial_condition()

    # If the obstacle exists, remove concentration inside the obstacle at t = 0.
    if use_obstacle:
        u_initial[mask] = 0.0

    u = u_initial.copy()

    for n in range(Nt):
        u = step(u, D, use_obstacle)

    return u_initial, u


# ==================================================
# RUN ALL CASES
# ==================================================

results = []

for case in cases:
    u_initial_case, u_final_case = run_simulation(
        D=case["D"],
        use_obstacle=case["use_obstacle"],
    )
    results.append((case, u_initial_case, u_final_case))

# ==================================================
# PLOTS
# ==================================================


levels = np.linspace(0.0, 1.0, 61)

for case, u_initial_case, u_final_case in results:

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(case["name"], fontsize=14)

    # ---------- Initial state ----------
    c0 = axes[0].contourf(
        X,
        Y,
        u_initial_case,
        levels=levels,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Initial State")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Mark the initial blob center.
    axes[0].plot(x0, y0, "wo", markeredgecolor="black", markersize=5)

    # ---------- Final state ----------
    c1 = axes[1].contourf(
        X,
        Y,
        u_final_case,
        levels=levels,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"Final State, T = {T:.2f}")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    # Mark the expected advective center without obstacle.
    xfinal = x0 + a * T
    yfinal = y0 + b * T
    axes[1].plot(xfinal, yfinal, "wo", markeredgecolor="black", markersize=5)

    # Draw the filled obstacle only for the obstacle case.
    if case["use_obstacle"]:
        for ax in axes:
            obstacle = plt.Circle(
                (xc, yc),
                r,
                facecolor="lightgray",
                edgecolor="black",
                fill=True,
                linewidth=1.0,
                zorder=10,
            )
            ax.add_patch(obstacle)

    # Add one colorbar for this figure.
    fig.colorbar(c1, ax=axes.ravel().tolist(), label="Concentration")

plt.show()
