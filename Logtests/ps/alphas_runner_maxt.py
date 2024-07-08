import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the values of asif and min_t to iterate over
asif_values = [0.0025, 0.0025, 0.0025, 0.0025, 0.0025]
min_t_values = [1e-34, 1e-34, 1e-34, 1e-34, 1e-34]
max_t_values = [1e-2, 1e-6, 1e-10, 1e-14, 1e-18]
evs = 100000

# Define a function to run the main script with specific asif and min_t values
def run_main_script(asif, min_t, max_t):
    command = [
        'python3', 'shower_LL_run_thrust.py',
        '--asif', str(asif),
        '--max_t', str(max_t),
        '--min_t', str(min_t),
        '--e', str(evs)
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script with asif={asif}, min_t={min_t}, and max_t={max_t}: {e}")

# Run the main script for each combination of asif and min_t
results = []
for i in range(len(asif_values)):
    asif = asif_values[i]
    min_t = min_t_values[i]
    max_t = max_t_values[i]
    run_main_script(asif, min_t, max_t)
    # Read the resulting lambda_2 from params.txt
    if os.path.exists('params.txt'):
        with open('params.txt', 'r') as f:
            lines = f.readlines()
            lambda_2_value = float(lines[0].split(":")[1].strip())
            results.append((asif, min_t, lambda_2_value, max_t))

# Save results to a file
with open('all_params.txt', 'w') as f:
    for asif, min_t, lambda_2_value, max_t in results:
        f.write(f"asif: {asif}, min_t: {min_t}, lambda_2: {lambda_2_value}, max_t: {max_t}\n")

# Extract max_t and lambda_2 values for plotting
max_t_plot_values = [result[3] for result in results]
lambda_2_plot_values = [result[2] for result in results]

# Plot lambda_2 vs. max_t with a logarithmic x-axis
plt.figure()
plt.plot(max_t_plot_values, lambda_2_plot_values, marker='o', label='Data')

# Perform a linear fit
coef = np.polyfit(np.log(max_t_plot_values), lambda_2_plot_values, 1)
poly1d_fn = np.poly1d(coef)

# Add the linear fit line to the plot
plt.plot(max_t_plot_values, poly1d_fn(np.log(max_t_plot_values)), '--k', label=f'Fit: y = {coef[0]:.4f}log(x) + {coef[1]:.4f}')

plt.xscale('log')
plt.xlabel('max_t (log scale)')
plt.ylabel('lambda_2')
plt.title('lambda_2 vs. max_t')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('lambda_2_vs_max_t.png')
plt.show()

# Plot lambda_2 vs. asif for each min_t
for min_t in set(min_t_values):  # Ensure unique min_t values
    lambda_2_for_min_t = [lambda_2 for asif, mt, lambda_2, max_t in results if mt == min_t]
    if len(lambda_2_for_min_t) == len(asif_values):  # Ensure data matches in length
        plt.plot(asif_values, lambda_2_for_min_t, label=f"min_t={min_t}")

plt.xlabel('asif')
plt.ylabel('lambda_2')
plt.legend()
plt.title('lambda_2 vs. asif for different min_t values')
plt.grid(True, which="both", ls="--")
plt.savefig('lambda_2_vs_asif_per_min_t.png')
plt.show()
