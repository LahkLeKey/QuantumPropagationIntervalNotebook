# Note: This script can be run on Kaggle.com
# - Kaggle offers a free and powerful environment for running Python code with pre-installed libraries like NumPy, Pandas, and Matplotlib.
# - To use this script:
#   1. Go to https://www.kaggle.com/code.
#   2. Click "New Notebook," paste this code, and run all cells for a complete presentation of results and plots.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
c = 299_792_458  # Speed of light in m/s
hertz_inverse_meter_relationship = 3.3356409519815204e-9  # Defined by NIST
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M_earth = 5.972e24  # Mass of Earth in kg
R_earth = 6.371e6  # Radius of Earth in meters
M_sun = 1.989e30  # Mass of Sun in kg
M_bh = 10 * M_sun  # Example black hole mass
R_schwarzschild = 2 * G * M_bh / c**2  # Schwarzschild radius

# Function to calculate quantum propagation interval
def calculate_quantum_interval(mass):
    E = (mass * c / 2) ** 2
    entangled_value = 2 * np.sqrt(E)
    interval = entangled_value / (mass * c)
    return E, entangled_value, interval

# Function to apply relativistic Lorentz correction
def lorentz_correction(v):
    gamma = 1 / np.sqrt(1 - (v / c) ** 2)
    return gamma

# Function to calculate gravitational redshift effect
def gravitational_redshift(mass, distance):
    if distance <= R_schwarzschild:
        return np.nan
    potential = -G * mass / distance
    redshift = 1 / np.sqrt(1 + (2 * potential / c**2))
    return redshift

# Black Hole Visualization in 3D

def plot_black_hole_escape_3d():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D funnel (black hole warp)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 1, 100)
    u, v = np.meshgrid(u, v)
    x = np.cos(u) * (1 - v**2)
    y = np.sin(u) * (1 - v**2)
    z = -v**2 * 2

    ax.plot_surface(x, y, z, cmap='inferno', alpha=0.8)

    # Create the 3D mirrored plane structure
    net_x, net_y = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    net_z = 0.1 * (net_x**2 - net_y**2)
    ax.plot_wireframe(net_x, net_y, net_z, color='blue', alpha=0.5)

    # Add escaping path to a mirrored 3D plane
    escape_x = np.linspace(0, 2, 100)
    escape_y = np.sin(escape_x * np.pi)
    escape_z = np.linspace(-2, 1, 100)
    mirrored_escape_z = -escape_z  # Mirrored plane
    ax.plot(escape_x, escape_y, escape_z, color='green', linewidth=2, label="Escape Path")
    ax.plot(escape_x, escape_y, mirrored_escape_z, color='purple', linewidth=2, linestyle='--', label="Mirrored Path")

    # Add labels and title
    ax.set_title("Black Hole Escape Visualization: 3D Funnel Escaping to Mirrored 3D Plane", fontsize=14)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.legend()

    plt.show()

# Simulation setup
masses = [1.0e20, 1.11112e25, 2.22223e25, 3.33334e25, 4.44445e25, M_bh]
velocities = [0.1 * c, 0.5 * c, 0.9 * c, 0.99 * c]
distances = [R_earth, 2 * R_earth, 10 * R_earth, R_schwarzschild, 2 * R_schwarzschild]

# Simulation results
mass_data = {
    "Mass (kg)": [], "Distance (m)": [], "Velocity (m/s)": [], "Quantum Interval (s)": [],
    "Gravitational Redshift": [], "Lorentz Factor": [], "Energy (E = mc^2, J)": [],
    "Black Hole Escape Condition": []
}

for mass in masses:
    for distance in distances:
        for velocity in velocities:
            E, _, interval = calculate_quantum_interval(mass)
            gamma = lorentz_correction(velocity)
            redshift = gravitational_redshift(mass, distance)
            is_black_hole = distance <= R_schwarzschild
            mass_data["Mass (kg)"].append(mass)
            mass_data["Distance (m)"].append(distance)
            mass_data["Velocity (m/s)"].append(velocity)
            mass_data["Quantum Interval (s)"].append(interval)
            mass_data["Gravitational Redshift"].append(redshift)
            mass_data["Lorentz Factor"].append(gamma)
            mass_data["Energy (E = mc^2, J)"].append(E)
            mass_data["Black Hole Escape Condition"].append(is_black_hole)

# Convert results to DataFrame
results_df = pd.DataFrame(mass_data)

# Combined and annotated plots
def plot_combined_advanced_results():
    distances_plot = results_df["Distance (m)"].unique()
    velocities_plot = results_df["Velocity (m/s)"].unique()
    quantum_intervals = results_df.groupby("Distance (m)")["Quantum Interval (s)"].mean()
    redshifts = results_df.groupby("Distance (m)")["Gravitational Redshift"].mean()

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Quantum Interval vs Distance and Gravitational Redshift
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.plot(distances_plot, quantum_intervals, label="Quantum Interval", color="blue", marker="o")
    ax2.plot(distances_plot, redshifts, label="Gravitational Redshift", color="red", linestyle="--", marker="x")
    ax1.set_xlabel("Distance from Mass (m)")
    ax1.set_ylabel("Quantum Interval (s)", color="blue")
    ax2.set_ylabel("Gravitational Redshift Factor", color="red")
    ax1.set_title("Quantum Interval vs Gravitational Redshift")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid()

    # Lorentz Factor vs Velocity
    lorentz_factors = results_df.groupby("Velocity (m/s)")["Lorentz Factor"].mean()
    ax3 = axs[1]
    ax3.plot(velocities_plot, lorentz_factors, label="Lorentz Factor", color="green", marker="v")
    ax3.set_xlabel("Velocity (m/s)")
    ax3.set_ylabel("Lorentz Factor")
    ax3.set_title("Lorentz Factor vs Velocity")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.show()

# Run the plots
plot_combined_advanced_results()
plot_black_hole_escape_3d()
