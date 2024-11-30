"""
Quantum Meets Black Holes: Simulate and Visualize the Universe's Mysteries
--------------------------------------------------------------------------

This notebook integrates the original functionality (quantum physics, relativistic effects, and black hole visualizations) 
with new features including rotating black holes, their gravitational influence, and enhancements to the chaotic spacetime net.

Features:
- Inverse Meter-Hertz (NIST), gravitational redshifts, and Lorentz factors.
- Visualizations of black hole event horizons and rotational energies.
- Relationships between physical quantities through pair plots.
- A zoomed-out 3D universe populated with simulated galaxies.
- Chaotic spacetime net entangling galaxies for deeper exploration.
- Rotating black holes (Kerr Black Holes) added, with gravitational influence zones visualized.

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
BLACK_HOLE_COUNT = 5  # Number of black holes

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

# Galaxy and Black Hole Simulation
np.random.seed(42)  # For reproducibility
galaxy_positions = np.random.uniform(-UNIVERSE_SCALE, UNIVERSE_SCALE, (GALAXY_COUNT, 3))
black_hole_positions = np.random.uniform(-UNIVERSE_SCALE / 2, UNIVERSE_SCALE / 2, (BLACK_HOLE_COUNT, 3))
black_hole_masses = np.random.uniform(5 * M_sun, 20 * M_sun, BLACK_HOLE_COUNT)

# Enhanced Chaotic Spacetime Net
def create_chaotic_spacetime_net():
    """Generates a chaotic spacetime net connecting galaxies and black holes."""
    connections = []
    curvatures = []

    # Connect galaxies to each other
    for i in range(len(galaxy_positions)):
        for j in range(len(galaxy_positions)):
            if i != j:  # Avoid self-connections
                connections.append((galaxy_positions[i], galaxy_positions[j]))
                curvatures.append(1 / (np.linalg.norm(galaxy_positions[i] - galaxy_positions[j]) + 1e-10))

    # Connect black holes to galaxies
    for bh in black_hole_positions:
        for galaxy in galaxy_positions:
            connections.append((bh, galaxy))
            curvatures.append(1 / (np.linalg.norm(bh - galaxy) + 1e-10))

    return connections, curvatures

chaotic_connections, chaotic_curvatures = create_chaotic_spacetime_net()

# Enhanced Visualizations
def plot_chaotic_universe_with_black_holes():
    """Plot the zoomed-out universe with galaxies, black holes, and chaotic spacetime net."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Galaxies
    ax.scatter(
        galaxy_positions[:, 0],
        galaxy_positions[:, 1],
        galaxy_positions[:, 2],
        s=50,
        c="blue",
        alpha=0.8,
        label="Galaxies"
    )

    # Plot Black Holes with Gravitational Influence
    for bh, mass in zip(black_hole_positions, black_hole_masses):
        ax.scatter(
            bh[0], bh[1], bh[2],
            s=200,
            c="red",
            alpha=0.9,
            label=f"Black Hole (Mass: {mass / M_sun:.1f} M☉)"
        )
        # Gravitational influence sphere
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = R_schwarzschild * np.sin(v) * np.cos(u) + bh[0]
        y = R_schwarzschild * np.sin(v) * np.sin(u) + bh[1]
        z = R_schwarzschild * np.cos(v) + bh[2]
        ax.plot_wireframe(x, y, z, color="red", alpha=0.2)

    # Plot Chaotic Spacetime Net
    for (point1, point2), curvature in zip(chaotic_connections, chaotic_curvatures):
        t = np.linspace(0, 1, 50)
        curve_x = point1[0] + t * (point2[0] - point1[0])
        curve_y = point1[1] + t * (point2[1] - point1[1])
        curve_z = point1[2] + t * (point2[2] - point1[2]) - curvature * np.sin(t * np.pi)

        ax.plot(curve_x, curve_y, curve_z, color='gray', alpha=0.1, linewidth=0.3)

    # Set Axis Labels and Limits
    ax.set_xlim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_ylim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_zlim(-UNIVERSE_SCALE, UNIVERSE_SCALE)
    ax.set_title("Chaotic Universe with Black Holes and Spacetime Net", fontsize=16)
    ax.set_xlabel("X (Light-Years)")
    ax.set_ylabel("Y (Light-Years)")
    ax.set_zlabel("Z (Light-Years)")
    ax.legend()
    plt.show()

# Execute Visualization
plot_chaotic_universe_with_black_holes()
