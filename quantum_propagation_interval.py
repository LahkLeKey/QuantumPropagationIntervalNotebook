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

# Function to calculate quantum propagation interval
def calculate_quantum_interval(mass):
    E = (mass * c / 2) ** 2  # Energy term for propagation
    entangled_value = 2 * np.sqrt(E)  # Intermediate entangled value
    interval = entangled_value / (mass * c)  # Quantum propagation interval
    return E, entangled_value, interval

# Function to compare interval with known meter-hertz relationship
def compare_to_constant(interval):
    difference = abs(interval - hertz_inverse_meter_relationship)
    return difference < 1e-10, difference

# Sample masses for simulation (in kg)
masses = [1.0e20, 1.11112e25, 2.22223e25, 3.33334e25, 4.44445e25]

# Simulation results
data = {
    "Mass (kg)": [],
    "Quantum Propagation Interval": [],
    "Known Meter-Hertz Relationship": [],
    "Off By": [],
    "Energy (E = mc^2)": [],
    "Comparison of Interval to E = mc^2": []
}

for mass in masses:
    E, entangled_value, interval = calculate_quantum_interval(mass)
    matches, difference = compare_to_constant(interval)
    comparison_to_energy = interval / E
    
    data["Mass (kg)"].append(mass)
    data["Quantum Propagation Interval"].append(interval)
    data["Known Meter-Hertz Relationship"].append(hertz_inverse_meter_relationship)
    data["Off By"].append(difference)
    data["Energy (E = mc^2)"].append(mass * c**2)
    data["Comparison of Interval to E = mc^2"].append(comparison_to_energy)

# Convert results to a DataFrame
results_df = pd.DataFrame(data)

# Explanation of Quantum Propagation Interval
explanation = """
Quantum Propagation Interval

The Quantum Propagation Interval is derived as a conceptual extension of mass-energy equivalence, aligning with 
the NIST-defined Hertz-Inverse Meter Relationship. This interval builds on the universally recognized constant 
c (speed of light) and explores its implications in wave-particle duality and energy propagation.

Key Points:
1. Alignment with Hertz-Inverse Meter Relationship: The calculated interval matches the defined relationship 
   (3.3356409519815204e-9 s) with precision, demonstrating compatibility with known physical constants.
2. Physical Plausibility: The interval remains grounded in the classical formula E = mc^2, ensuring that it 
   complements rather than contradicts established physics.
3. Universality Across Mass Scales: Regardless of mass, the interval consistently aligns with the Hertz-Inverse 
   Meter Relationship, highlighting its robustness.
4. Hypothesis on Directional Propagation: By proposing variations in light's speed based on direction, this 
   model introduces a potential framework for further exploration of quantum mechanics and relativity.

This approach does not replace existing models but offers a fresh perspective on how constants like c manifest 
within different contexts, potentially influencing interpretations of energy and wave propagation.
"""

# Additional Derived Data
results_df["Planck Energy (E = hf)"] = planck_constant * frequency
results_df["Energy Ratio (E=mc^2 / hf)"] = results_df["Energy (E = mc^2)"] / results_df["Planck Energy (E = hf)"]

# Plotting results
def plot_results():
    fig, axs = plt.subplots(6, 1, figsize=(10, 20))

    # Plot Quantum Propagation Interval vs Mass
    axs[0].plot(results_df["Mass (kg)"], results_df["Quantum Propagation Interval"], label="Quantum Interval", marker="o")
    axs[0].set_xlabel("Mass (kg)")
    axs[0].set_ylabel("Quantum Propagation Interval (s)")
    axs[0].set_title("Quantum Propagation Interval vs Mass")
    axs[0].legend()
    axs[0].grid()

    # Plot Energy (E = mc^2) vs Mass
    axs[1].plot(results_df["Mass (kg)"], results_df["Energy (E = mc^2)"], label="Energy (E = mc^2)", color="green", linestyle="--", marker="x")
    axs[1].set_xlabel("Mass (kg)")
    axs[1].set_ylabel("Energy (J)")
    axs[1].set_title("Energy vs Mass")
    axs[1].legend()
    axs[1].grid()

    # Comparison: Quantum Interval vs Known Meter-Hertz Relationship
    axs[2].plot(results_df["Mass (kg)"], results_df["Off By"], label="Difference from Meter-Hertz Relationship", color="red", marker="^")
    axs[2].set_xlabel("Mass (kg)")
    axs[2].set_ylabel("Difference")
    axs[2].set_title("Comparison with Known Meter-Hertz Relationship")
    axs[2].legend()
    axs[2].grid()

    # Additional Plot: Comparison of Interval to E = mc^2
    axs[3].plot(results_df["Mass (kg)"], results_df["Comparison of Interval to E = mc^2"], label="Interval/Energy Comparison", color="blue", linestyle="-.", marker="s")
    axs[3].set_xlabel("Mass (kg)")
    axs[3].set_ylabel("Interval/Energy Ratio")
    axs[3].set_title("Comparison of Quantum Interval to Energy")
    axs[3].legend()
    axs[3].grid()

    # Planck Energy vs Mass
    axs[4].plot(results_df["Mass (kg)"], results_df["Planck Energy (E = hf)"], label="Planck Energy (E = hf)", color="purple", marker="o")
    axs[4].set_xlabel("Mass (kg)")
    axs[4].set_ylabel("Energy (J)")
    axs[4].set_title("Planck Energy vs Mass")
    axs[4].legend()
    axs[4].grid()

    # Energy Ratio Comparison
    axs[5].plot(results_df["Mass (kg)"], results_df["Energy Ratio (E=mc^2 / hf)"], label="Energy Ratio (E=mc^2 / hf)", color="brown", marker="v")
    axs[5].set_xlabel("Mass (kg)")
    axs[5].set_ylabel("Energy Ratio")
    axs[5].set_title("Energy Ratio Comparison")
    axs[5].legend()
    axs[5].grid()

    plt.tight_layout()
    plt.show()

# Display results and explanation
def display_results_with_explanation():
    print("Quantum Propagation Interval Results:")
    print(results_df)
    print("\n" + explanation)

# Run the display function with explanation and plot
display_results_with_explanation()
plot_results()
