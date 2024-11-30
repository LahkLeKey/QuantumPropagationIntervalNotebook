"""
Quantum Meets Black Holes: Simulate and Visualize the Universe's Mysteries
--------------------------------------------------------------------------

This notebook integrates the original functionality (quantum physics, relativistic effects, and black hole visualizations) 
with new features including an enhanced zoomed-out view of the universe, chaotic spacetime net, and additional visualizations.

Features:
- Inverse Meter-Hertz (NIST), gravitational redshifts, and Lorentz factors.
- Visualizations of black hole event horizons and rotational energies.
- Relationships between physical quantities through pair plots.
- A zoomed-out 3D universe populated with simulated galaxies.
- Spacetime net with connections and curvature based on quantum intervals.
- Chaotic spacetime net entangling galaxies for deeper exploration.

Run this notebook locally or on platforms like Kaggle with Python installed.
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
GALAXY_COUNT = 100  # Number of galaxies

# Functions
def calculate_quantum_interval(mass):
    """Calculate quantum propagation interval (Inverse Meter-Hertz NIST)."""
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
    "Inverse Meter-Hertz (NIST)": [],
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
            simulation_data["Inverse Meter-Hertz (NIST)"].append(cached_quantum_interval)
            simulation_data["Gravitational Redshift"].append(cached_redshift)
            simulation_data["Lorentz Factor"].append(cached_lorentz_factor)
            simulation_data["Rotational Kinetic Energy (J)"].append(cached_kinetic_energy)

simulation_df = pd.DataFrame(simulation_data)

# Key Highlights
peak_quantum_interval = simulation_df["Inverse Meter-Hertz (NIST)"].max()
peak_lorentz_factor = simulation_df["Lorentz Factor"].max()
peak_rotational_energy = simulation_df["Rotational Kinetic Energy (J)"].max()

print(f"✨ Peak Inverse Meter-Hertz (NIST): {peak_quantum_interval:.2e}")
print(f"✨ Peak Lorentz Factor: {peak_lorentz_factor:.2f}")
print(f"✨ Peak Rotational Kinetic Energy: {peak_rotational_energy:.2e} Joules")

# Galaxy Simulation for Universe View
np.random.seed(42)  # For reproducibility
galaxy_positions = np.random.uniform(-UNIVERSE_SCALE, UNIVERSE_SCALE, (GALAXY_COUNT, 3))
galaxy_sizes = np.random.uniform(0.1, 5, GALAXY_COUNT)  # Arbitrary galaxy sizes
galaxy_colors = np.random.rand(GALAXY_COUNT, 3)  # RGB colors for diversity

# Enhanced Chaotic Spacetime Net
def create_chaotic_spacetime_net():
    """Generates a chaotic spacetime net connecting all galaxies."""
    connections = []
    curvatures = []

    for i in range(len(galaxy_positions)):
        for j in range(len(galaxy_positions)):
            if i != j:  # Avoid self-connections
                connections.append((i, j))
                curvatures.append(1 / (np.linalg.norm(galaxy_positions[i] - galaxy_positions[j]) + 1e-10))  # Inverse distance
    return connections, curvatures

chaotic_connections, chaotic_curvatures = create_chaotic_spacetime_net()

# Enhanced Visualizations
def plot_chaotic_universe_with_spacetime_net():
    """Plot the zoomed-out universe with galaxies and a chaotic spacetime net."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Galaxies
    ax.scatter(
        galaxy_positions[:, 0],
        galaxy_positions[:, 1],
        galaxy_positions[:, 2],
        s=galaxy_sizes * 20,
        c=galaxy_colors,
        alpha=0.8,
        edgecolor='k',
        label="Galaxies"
    )

    # Plot Chaotic Spacetime Net
    for (i, j), curvature in zip(chaotic_connections, chaotic_curvatures):
        galaxy1 = galaxy_positions[i]
        galaxy2 = galaxy_positions[j]

        # Curve based on chaotic curvature
        t = np.linspace(0, 1, 50)
        curve_x = galaxy1[0] + t * (galaxy2[0] - galaxy1[0])
        curve_y = galaxy1[1] + t * (galaxy2[1] - galaxy1[1])
        curve_z = galaxy1[2] + t * (galaxy2[2] - galaxy1[2]) - curvature * np.sin(t * np.pi)  # Chaotic bending

        ax.plot(curve_x, curve_y, curve_z, color='gray', alpha=0.1, linewidth=0.3)

    # Set Axis Labels and Limits
    ax.set_xlim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_ylim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_zlim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_title("Chaotic Universe with Spacetime Net", fontsize=16)
    ax.set_xlabel("X (Light-Years)")
    ax.set_ylabel("Y (Light-Years)")
    ax.set_zlabel("Z (Light-Years)")
    ax.legend()
    plt.show()

# Combined Plot
def plot_combined():
    """Combined Inverse Meter-Hertz (NIST) and relativistic plots."""
    distances_plot = np.sort(simulation_df["Distance (m)"].unique())
    velocities_plot = np.sort(simulation_df["Velocity (m/s)"].unique())
    quantum_intervals = simulation_df.groupby("Distance (m)")["Inverse Meter-Hertz (NIST)"].mean()
    redshifts = simulation_df.groupby("Distance (m)")["Gravitational Redshift"].mean()
    lorentz_factors = simulation_df.groupby("Velocity (m/s)")["Lorentz Factor"].mean()
    rotational_energies = simulation_df.groupby("Velocity (m/s)")["Rotational Kinetic Energy (J)"].mean()

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.plot(distances_plot, quantum_intervals, label="Inverse Meter-Hertz (NIST)", color="blue", marker="o")
    ax2.plot(distances_plot, redshifts, label="Gravitational Redshift", color="red", linestyle="--", marker="x")
    ax1.set_title("Inverse Meter-Hertz (NIST) vs Gravitational Redshift")
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
plot_chaotic_universe_with_spacetime_net()
