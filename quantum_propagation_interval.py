"""
Quantum Meets Black Holes: Simulate and Visualize the Universe's Mysteries
--------------------------------------------------------------------------

This notebook demonstrates the interplay between quantum mechanics and relativity.
We simulate quantum propagation intervals, gravitational redshifts, and relativistic effects.
We also visualize how spacetime bends around a black hole.

- Discover how quantum mechanics and general relativity intersect.
- See how changes in mass, distance, and velocity impact physical quantities.
- Explore 3D visualizations of black hole event horizons and relativistic effects.

You can run this notebook locally or on platforms like Kaggle with Python installed.
Enjoy your journey into the depths of physics!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings

# Suppress specific FutureWarnings related to deprecated Pandas options
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Constants
c = 299_792_458  # Speed of light in m/s
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
R_earth = 6.371e6  # Radius of Earth in meters
omega_earth = 7.2921159e-5  # Angular velocity of Earth in rad/s
M_sun = 1.989e30  # Mass of the Sun in kg
M_bh = 10 * M_sun  # Example black hole mass
R_schwarzschild = 2 * G * M_bh / c**2  # Schwarzschild radius

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

# Simulation Data
distances = np.array([R_earth, 2 * R_earth, 10 * R_earth])
velocities = np.array([0.1 * c, 0.5 * c, 0.9 * c, 0.99 * c])
angular_velocities = np.array([omega_earth, 2 * omega_earth, 5 * omega_earth])
masses = np.array([1e25, 2e25, 5e25])

# Simulate Results
simulation_data = {
    "Mass (kg)": [],
    "Distance (m)": [],
    "Velocity (m/s)": [],
    "Angular Velocity (rad/s)": [],
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
            for angular_velocity in angular_velocities:
                cached_kinetic_energy = rotational_kinetic_energy(mass, R_earth, angular_velocity)
                simulation_data["Mass (kg)"].append(mass)
                simulation_data["Distance (m)"].append(distance)
                simulation_data["Velocity (m/s)"].append(velocity)
                simulation_data["Angular Velocity (rad/s)"].append(angular_velocity)
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

# Combined Plots
def plot_combined():
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    distances_plot = np.sort(simulation_df["Distance (m)"].unique())
    velocities_plot = np.sort(simulation_df["Velocity (m/s)"].unique())

    quantum_intervals = simulation_df.groupby("Distance (m)")["Quantum Interval (s)"].mean()
    redshifts = simulation_df.groupby("Distance (m)")["Gravitational Redshift"].mean()
    lorentz_factors = simulation_df.groupby("Velocity (m/s)")["Lorentz Factor"].mean()
    rotational_energies = simulation_df.groupby("Velocity (m/s)")["Rotational Kinetic Energy (J)"].mean()

    # Ensure matching dimensions for plotting
    if len(velocities_plot) > len(rotational_energies):
        velocities_plot = velocities_plot[:len(rotational_energies)]

    # Left Plot
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.plot(distances_plot, quantum_intervals, label="Quantum Interval", color="blue", marker="o")
    ax2.plot(distances_plot, redshifts, label="Gravitational Redshift", color="red", linestyle="--", marker="x")
    ax1.set_xlabel("Distance from Mass (m)", fontsize=12)
    ax1.set_ylabel("Quantum Interval (s)", color="blue", fontsize=12)
    ax2.set_ylabel("Gravitational Redshift Factor", color="red", fontsize=12)
    ax1.set_title("Quantum Interval vs Gravitational Redshift", fontsize=14)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True)

    # Right Plot
    ax3 = axs[1]
    ax4 = ax3.twinx()
    ax3.plot(velocities_plot, lorentz_factors, label="Lorentz Factor", color="green", marker="v")
    ax4.plot(velocities_plot, rotational_energies, label="Rotational Energy", color="orange", linestyle="-", marker="^")
    ax3.set_xlabel("Velocity (m/s)", fontsize=12)
    ax3.set_ylabel("Lorentz Factor", color="green", fontsize=12)
    ax4.set_ylabel("Rotational Energy (J)", color="orange", fontsize=12)
    ax3.set_title("Lorentz Factor vs Rotational Energy", fontsize=14)
    ax3.legend(loc="upper left")
    ax4.legend(loc="upper right")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

# Event Horizon Visualization
def plot_event_horizon():
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Funnel
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, 1, 200)
    u, v = np.meshgrid(u, v)
    x_funnel = R_schwarzschild * np.cos(u) * (1 - v**2)
    y_funnel = R_schwarzschild * np.sin(u) * (1 - v**2)
    z_funnel = -v**2 * R_schwarzschild - R_schwarzschild
    ax.plot_surface(x_funnel, y_funnel, z_funnel, cmap='inferno', alpha=0.6)

    # Black Hole
    phi, theta = np.mgrid[0:2.0 * np.pi:100j, 0:np.pi:50j]
    x_hole = R_schwarzschild * np.sin(theta) * np.cos(phi)
    y_hole = R_schwarzschild * np.sin(theta) * np.sin(phi)
    z_hole = R_schwarzschild * np.cos(theta)
    ax.plot_surface(x_hole, y_hole, z_hole, color='cyan', alpha=0.8)

    # Labels
    ax.set_title("Dynamic Event Horizon Visualization", fontsize=16)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (m)", fontsize=12)

    plt.show()

# Pairplot Visualization of Relationships
sns.set(style="whitegrid")
def plot_pairplot():
    print("📊 Generating Pairplot Visualization...")
    pairplot_df = simulation_df[["Mass (kg)", "Distance (m)", "Velocity (m/s)", "Quantum Interval (s)", "Gravitational Redshift", "Lorentz Factor", "Rotational Kinetic Energy (J)"]]
    pairplot = sns.pairplot(pairplot_df, diag_kind="kde", plot_kws={"alpha": 0.6}, height=2.0, aspect=1.2)
    
    # Add correlation coefficients
    for i, ax in enumerate(pairplot.axes.flat):
        if ax is not None:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel and ylabel and xlabel != ylabel:
                corr = pairplot_df[xlabel].corr(pairplot_df[ylabel])
                ax.annotate(f'Corr: {corr:.2f}', xy=(0.1, 0.9), xycoords='axes fraction', 
                            ha='center', va='center', fontsize=10, color='blue')
    
    plt.suptitle("Relationships Between Physical Quantities", y=1.02, fontsize=16)
    print("🎉 Pairplot Visualization Generated! (It may take a few seconds to render)")
    plt.show()

# Run Visualizations
plot_combined()
plot_event_horizon()
plot_pairplot()
