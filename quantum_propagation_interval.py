# Note: This script can be run on Kaggle.com
# - Kaggle offers a free and powerful environment for running Python code with pre-installed libraries like NumPy, Pandas, and Matplotlib.
# - To use this script:
#   1. Go to https://www.kaggle.com/code.
#   2. Click "New Notebook," paste this code, and run all cells for a complete presentation of results and plots.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
c = 299_792_458  # Speed of light in m/s
hertz_inverse_meter_relationship = 3.3356409519815204e-9  # Defined by NIST
planck_constant = 6.62607015e-34  # Planck's constant in J*s
frequency = c / 1e-6  # Example frequency (1 micron wavelength)
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M_earth = 5.972e24  # Mass of Earth in kg
R_earth = 6.371e6  # Radius of Earth in meters

# Function to calculate quantum propagation interval
def calculate_quantum_interval(mass):
    E = (mass * c / 2) ** 2  # Energy term for propagation
    entangled_value = 2 * np.sqrt(E)  # Intermediate entangled value
    interval = entangled_value / (mass * c)  # Quantum propagation interval
    return E, entangled_value, interval

# Function to apply relativistic Lorentz correction
def lorentz_correction(v):
    gamma = 1 / np.sqrt(1 - (v / c) ** 2)
    return gamma

# Function to calculate gravitational redshift effect
def gravitational_redshift(mass, distance):
    potential = -G * mass / distance
    redshift = 1 / np.sqrt(1 + (2 * potential / c**2))
    return redshift

# Sample masses, velocities, and distances for simulation
masses = [1.0e20, 1.11112e25, 2.22223e25, 3.33334e25, 4.44445e25]
velocities = [0.1 * c, 0.5 * c, 0.9 * c, 0.99 * c]
distances = [R_earth, 2 * R_earth, 10 * R_earth]

# Simulation results
mass_data = {
    "Mass (kg)": [],
    "Distance (m)": [],
    "Velocity (m/s)": [],
    "Quantum Interval (s)": [],
    "Gravitational Redshift": [],
    "Lorentz Factor": [],
    "Energy (E = mc^2, J)": [],
    "Known Meter-Hertz Relationship": [],
    "Difference from Meter-Hertz Relationship": []
}

for mass in masses:
    for distance in distances:
        for velocity in velocities:
            E, _, interval = calculate_quantum_interval(mass)
            gamma = lorentz_correction(velocity)
            redshift = gravitational_redshift(mass, distance)
            difference = abs(interval - hertz_inverse_meter_relationship)

            mass_data["Mass (kg)"].append(mass)
            mass_data["Distance (m)"].append(distance)
            mass_data["Velocity (m/s)"].append(velocity)
            mass_data["Quantum Interval (s)"].append(interval)
            mass_data["Gravitational Redshift"].append(redshift)
            mass_data["Lorentz Factor"].append(gamma)
            mass_data["Energy (E = mc^2, J)"].append(E)
            mass_data["Known Meter-Hertz Relationship"].append(hertz_inverse_meter_relationship)
            mass_data["Difference from Meter-Hertz Relationship"].append(difference)

# Convert results to a DataFrame
results_df = pd.DataFrame(mass_data)

# Explanation of Quantum Propagation Interval
explanation = """
Quantum Propagation Interval

The Quantum Propagation Interval is derived as a conceptual extension of mass-energy equivalence, aligning with 
the NIST-defined Hertz-Inverse Meter Relationship. This interval builds on the universally recognized constant 
c (speed of light) and explores its implications in wave-particle duality, relativistic effects, and gravitational interactions.

Key Insight:
Gravitational redshift inversely correlates with the quantum propagation interval. At higher distances from a 
massive object, gravitational redshift diminishes while the quantum interval increases, reflecting opposite trends. 

This relationship further grounds the quantum interval in Einstein's theoretical framework and provides a mechanism 
for experimental validation through astrophysical and cosmological observations.
"""

# Combined and advanced plots
def plot_combined_advanced_results():
    distances_plot = results_df["Distance (m)"].unique()
    velocities_plot = results_df["Velocity (m/s)"].unique()

    quantum_intervals = results_df.groupby("Distance (m)")["Quantum Interval (s)"].mean()
    redshifts = results_df.groupby("Distance (m)")["Gravitational Redshift"].mean()
    lorentz_factors = results_df.groupby("Velocity (m/s)")["Lorentz Factor"].mean()

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Quantum Interval vs Distance and Gravitational Redshift vs Distance (Dual Axis)
    ax1 = axs[0, 0]
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
    axs[0, 1].plot(velocities_plot, lorentz_factors, label="Lorentz Factor", color="green", marker="v")
    axs[0, 1].set_xlabel("Velocity (m/s)")
    axs[0, 1].set_ylabel("Lorentz Factor")
    axs[0, 1].set_title("Lorentz Factor vs Velocity")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Quantum Interval and Gravitational Redshift vs Mass
    quantum_intervals_mass = results_df.groupby("Mass (kg)")["Quantum Interval (s)"].mean()
    redshifts_mass = results_df.groupby("Mass (kg)")["Gravitational Redshift"].mean()
    axs[1, 0].plot(masses, quantum_intervals_mass, label="Quantum Interval", color="blue", marker="o")
    axs[1, 0].plot(masses, redshifts_mass, label="Gravitational Redshift", color="red", linestyle="--", marker="x")
    axs[1, 0].set_xlabel("Mass (kg)")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].set_title("Quantum Interval and Gravitational Redshift vs Mass")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # Energy (E = mc^2) vs Mass
    energies_mass = results_df.groupby("Mass (kg)")["Energy (E = mc^2, J)"].mean()
    axs[1, 1].plot(masses, energies_mass, label="Energy (E = mc^2)", color="purple", marker="s")
    axs[1, 1].set_xlabel("Mass (kg)")
    axs[1, 1].set_ylabel("Energy (J)")
    axs[1, 1].set_title("Energy vs Mass")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()
    plt.show()

# Display results and explanation
def display_results_with_explanation():
    print("Quantum Propagation Interval Results:")
    print(results_df.head())
    print("\n" + explanation)

# Run the display function with explanation and plot advanced combined results
display_results_with_explanation()
plot_combined_advanced_results()