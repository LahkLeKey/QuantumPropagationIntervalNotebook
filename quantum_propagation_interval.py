"""
Quantum Meets Black Holes: Simulate and Visualize the Universe's Mysteries
--------------------------------------------------------------------------

This notebook integrates the original functionality (quantum physics, relativistic effects, and black hole visualizations) 
with a new zoomed-out view of the universe, complete with sample galaxies.

Features:
- Quantum propagation intervals, gravitational redshifts, and Lorentz factors.
- Visualizations of black hole event horizons and rotational energies.
- Relationships between physical quantities through pair plots.
- A zoomed-out 3D universe populated with simulated galaxies.

You can run this notebook locally or on platforms like Kaggle with Python installed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Constants
c = 299_792_458  # Speed of light in m/s
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
R_earth = 6.371e6  # Radius of Earth in meters
M_sun = 1.989e30  # Mass of the Sun in kg
M_bh = 10 * M_sun  # Example black hole mass
R_schwarzschild = 2 * G * M_bh / c**2  # Schwarzschild radius
LY = 9.461e15  # Light-year in meters
UNIVERSE_SCALE = 1e6 * LY  # Universe scale in light-years
GALAXY_COUNT = 200  # Number of galaxies

# Functions
def calculate_quantum_interval(mass):
    """Calculate quantum propagation interval."""
    E = (mass * c / 2) ** 2
    interval = E / (mass * c)
    return interval

def lorentz_correction(v):
    """Calculate Lorentz factor."""
    return 1 / np.sqrt(1 - (v / c) ** 2)

def gravitational_redshift(mass, distance):
    """Calculate gravitational redshift factor."""
    potential = -G * mass / distance
    return 1 / np.sqrt(1 + (2 * potential / c**2))

def rotational_kinetic_energy(mass, radius, omega):
    """Calculate rotational kinetic energy."""
    I = (2 / 5) * mass * radius**2
    return 0.5 * I * omega**2

# Simulated Quantum and Relativistic Data
distances = np.array([R_earth, 2 * R_earth, 10 * R_earth])
velocities = np.array([0.1 * c, 0.5 * c, 0.9 * c, 0.99 * c])
masses = np.array([1e25, 2e25, 5e25])

simulation_data = {
    "Mass (kg)": [],
    "Distance (m)": [],
    "Velocity (m/s)": [],
    "Quantum Interval (s)": [],
    "Gravitational Redshift": [],
    "Lorentz Factor": [],
    "Rotational Kinetic Energy (J)": []
}

for mass in masses:
    cached_quantum_interval = calculate_quantum_interval(mass)
    for distance in distances:
        cached_redshift = gravitational_redshift(mass, distance)
        for velocity in velocities:
            cached_lorentz_factor = lorentz_correction(velocity)
            cached_kinetic_energy = rotational_kinetic_energy(mass, R_earth, velocity / R_earth)
            simulation_data["Mass (kg)"].append(mass)
            simulation_data["Distance (m)"].append(distance)
            simulation_data["Velocity (m/s)"].append(velocity)
            simulation_data["Quantum Interval (s)"].append(cached_quantum_interval)
            simulation_data["Gravitational Redshift"].append(cached_redshift)
            simulation_data["Lorentz Factor"].append(cached_lorentz_factor)
            simulation_data["Rotational Kinetic Energy (J)"].append(cached_kinetic_energy)

simulation_df = pd.DataFrame(simulation_data)

# Key Highlights
max_quantum_interval = simulation_df["Quantum Interval (s)"].max()
max_lorentz_factor = simulation_df["Lorentz Factor"].max()
max_rotational_energy = simulation_df["Rotational Kinetic Energy (J)"].max()

print(f"✨ Maximum Quantum Interval: {max_quantum_interval:.2e} seconds")
print(f"✨ Maximum Lorentz Factor: {max_lorentz_factor:.2f}")
print(f"✨ Maximum Rotational Kinetic Energy: {max_rotational_energy:.2e} Joules")

# Galaxy Simulation for Universe View
np.random.seed(42)  # For reproducibility
galaxy_positions = np.random.uniform(-UNIVERSE_SCALE, UNIVERSE_SCALE, (GALAXY_COUNT, 3))
galaxy_sizes = np.random.uniform(0.1, 5, GALAXY_COUNT)  # Arbitrary galaxy sizes
galaxy_colors = np.random.rand(GALAXY_COUNT, 3)  # RGB colors for diversity

# Visualizations
def plot_universe():
    """Plot a zoomed-out view of the universe with sample galaxies."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        galaxy_positions[:, 0],
        galaxy_positions[:, 1],
        galaxy_positions[:, 2],
        s=galaxy_sizes * 20,
        c=galaxy_colors,
        alpha=0.8,
        edgecolor='k'
    )
    ax.set_xlim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_ylim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_zlim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_title("Zoomed-Out View of the Universe", fontsize=16)
    ax.set_xlabel("X (Light-Years)")
    ax.set_ylabel("Y (Light-Years)")
    ax.set_zlabel("Z (Light-Years)")
    plt.show()

def plot_combined():
    """Combined quantum and relativistic plots."""
    distances_plot = np.sort(simulation_df["Distance (m)"].unique())
    velocities_plot = np.sort(simulation_df["Velocity (m/s)"].unique())
    quantum_intervals = simulation_df.groupby("Distance (m)")["Quantum Interval (s)"].mean()
    redshifts = simulation_df.groupby("Distance (m)")["Gravitational Redshift"].mean()
    lorentz_factors = simulation_df.groupby("Velocity (m/s)")["Lorentz Factor"].mean()
    rotational_energies = simulation_df.groupby("Velocity (m/s)")["Rotational Kinetic Energy (J)"].mean()

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.plot(distances_plot, quantum_intervals, label="Quantum Interval", color="blue", marker="o")
    ax2.plot(distances_plot, redshifts, label="Gravitational Redshift", color="red", linestyle="--", marker="x")
    ax1.set_title("Quantum Interval vs Gravitational Redshift")
    ax1.grid(True)

    ax3 = axs[1]
    ax4 = ax3.twinx()
    ax3.plot(velocities_plot, lorentz_factors, label="Lorentz Factor", color="green", marker="v")
    ax4.plot(velocities_plot, rotational_energies, label="Rotational Energy", color="orange", linestyle="-", marker="^")
    ax3.set_title("Lorentz Factor vs Rotational Energy")
    ax3.grid(True)
    plt.tight_layout()
    plt.show()

# Execute Visualizations
plot_combined()
plot_universe()
