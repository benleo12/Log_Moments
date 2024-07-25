import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the values of asif and evs to iterate over
asif_values = [0.02, 0.0175, 0.015, 0.0125, 0.01, 0.0075, 0.005, 0.0025]
evs = 10000000

# Define a function to run the LL script with specific asif values
def run_ll_script(asif):
    command = [
        'python3', 'shower_chi2_run_thrust.py',
        '--asif', str(asif),
        '--e', str(evs),
        '--piece', 'll',
        '--nbins', '64'
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the LL script with asif={asif}: {e}")

# Define a function to run the NLLc script with specific asif and lambda_2 values
def run_nllc_script(asif, lambda_1):
    command = [
        'python3', 'shower_chi2_run_thrust.py',
        '--asif', str(asif),
        '--e', str(evs),
        '--lam1', str(lambda_1),
        '--piece', 'nllc',
        '--nbins', '64'
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the NLLc script with asif={asif}, lambda_1={lambda_1}: {e}")

# Run the LL script for each asif and read the resulting lambda_2 values
ll_results = []
for asif in asif_values:
    run_ll_script(asif)
    params_ll_filename = f'params_ll_{evs}.txt'
    if os.path.exists(params_ll_filename):
        with open(params_ll_filename, 'r') as f:
            lines = f.readlines()
            lambda_1_value = float(lines[0].split(":")[1].strip())
            ll_results.append((asif, lambda_1_value))

# Save LL results to a file
with open('ll_params.txt', 'w') as f:
    for asif, lambda_1_value in ll_results:
        f.write(f"asif: {asif}, lambda_1: {lambda_1_value}\n")

# Extract asif and lambda_2 values for plotting
asif_plot_values = [result[0] for result in ll_results]
lambda_1_plot_values = [result[1] for result in ll_results]

# Plot lambda_2 vs. asif
plt.plot(asif_plot_values, lambda_1_plot_values, marker='o', label='LL Data')

# Perform a linear fit
coef = np.polyfit(asif_plot_values, lambda_1_plot_values, 1)
poly1d_fn = np.poly1d(coef)

# Add the linear fit line to the plot
plt.plot(asif_plot_values, poly1d_fn(asif_plot_values), '--k', label=f'LL Fit: y = {coef[0]:.4f}x + {coef[1]:.4f}')

# Run the NLLc script for each asif using the lambda_2 values from LL
nllc_results = []
for asif, lambda_1_value in ll_results:
    run_nllc_script(asif, lambda_1_value)
    params_nllc_filename = f'params_nllc_{evs}.txt'
    if os.path.exists(params_nllc_filename):
        with open(params_nllc_filename, 'r') as f:
            lines = f.readlines()
            lambda_2_value_nllc = float(lines[1].split(":")[1].strip())
            nllc_results.append((asif, lambda_2_value_nllc))

# Save NLLc results to a file
with open('nllc_params.txt', 'w') as f:
    for asif, lambda_2_value_nllc in nllc_results:
        f.write(f"asif: {asif}, lambda_2: {lambda_2_value_nllc}\n")

# Extract asif and lambda_2 values for NLLc plotting
asif_plot_values_nllc = [result[0] for result in nllc_results]
lambda_2_plot_values_nllc = [result[1] for result in nllc_results]

# Plot lambda_2 vs. asif for NLLc
plt.plot(asif_plot_values_nllc, lambda_2_plot_values_nllc, marker='x', label='NLLc Data', color='red')

# Perform a linear fit for NLLc
coef_nllc = np.polyfit(asif_plot_values_nllc, lambda_2_plot_values_nllc, 1)
poly1d_fn_nllc = np.poly1d(coef_nllc)

# Add the linear fit line to the plot for NLLc
plt.plot(asif_plot_values_nllc, poly1d_fn_nllc(asif_plot_values_nllc), '--r', label=f'NLLc Fit: y = {coef_nllc[0]:.4f}x + {coef_nllc[1]:.4f}')

plt.xlabel('asif')
plt.ylabel('lambda_2')
plt.title('lambda_2 vs. asif')
plt.legend()
plt.savefig('lambda_2_vs_asif.png')
plt.show()
