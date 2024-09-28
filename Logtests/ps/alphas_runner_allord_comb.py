import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib with appropriate settings
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX to render text
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 6,  # Reduced font size for legend
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.figsize": (12, 8),  # Increased size for better visibility
    "figure.dpi": 300,
    "axes.linewidth": 0.5,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# Define the values of asif and evs to iterate over
asif_values = [0.08]  # You can expand this list as needed, e.g., [0.01, 0.0075, 0.005, 0.0025]
evs = 500000
N = 2    # Number of runs with different seeds
B = 500  # Number of bootstrap samples

# Define a fixed lambda_1 value since LL is not being run
fixed_lambda_1 = 0.0  # Adjust this value based on your requirements

# Define the list of parameters to collect
parameters = ['lambda_1', 'lambda_2', 'npm1', 'npm2', 'nps1', 'nps2', 'npn1', 'npn2']

# Define a function to run the NLLc script with specific asif and seed values
def run_nllc_script(asif, seed, lambda_1_fixed):
    command = [
        'python3', 'shower_chi2_run_thrust_nuiss.py',
        '--asif', str(asif),
        '--e', str(evs),
        '--piece', 'nllc',
        '--nbins', '1',
        '-K', '1',
        '-C', '1',
        '-B', '0.0',
        '--lam1', str(lambda_1_fixed),  # Use a fixed lambda_1
        '-s', str(seed)
    ]
    try:
        print(f"Executing command: {' '.join(command)}")
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the NLLc script with asif={asif}, seed={seed}: {e}")

# Function to perform bootstrap resampling and compute confidence intervals
def bootstrap_ci(data, n_bootstrap_samples=1000, ci=95):
    """Generate bootstrap confidence intervals for the mean."""
    bootstrapped_means = []
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    for _ in range(n_bootstrap_samples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower_bound, upper_bound

# Initialize a dictionary to store all parameters for each asif
nllc_results = {asif: {param: [] for param in parameters} for asif in asif_values}

# Function to delete existing CSV files to ensure fresh runs
def delete_existing_csv(asif, seed):
    csv_filenames = [
        f"thrust_e{evs}_A{asif}_nllc_K1.0_B0.0_seed{seed}.csv",
        f"weight_e{evs}_A{asif}_nllc_K1.0_B0.0_seed{seed}.csv"
    ]
    for csv_file in csv_filenames:
        if os.path.exists(csv_file):
            try:
                os.remove(csv_file)
                print(f"Deleted existing file '{csv_file}' to allow overwriting.")
            except Exception as e:
                print(f"Failed to delete '{csv_file}': {e}")

# Run the NLLc script for each asif with different seeds and collect all parameters
for asif in asif_values:
    for seed in range(1, N + 1):  # Starting seeds from 1 for clarity
        print(f"\nRunning NLLc script for asif={asif}, seed={seed}")
        
        # Delete existing CSV files before running to force generation of new files
        delete_existing_csv(asif, seed)
        
        # Run the NLLc script
        run_nllc_script(asif, seed, fixed_lambda_1)
        
        # Define original and new filenames for params
        original_params_filename = f'params_nllc_{evs}.txt'
        new_params_filename = f'params_nllc_{evs}_{seed}.txt'
        
        # Rename the params file to include the seed
        if os.path.exists(original_params_filename):
            try:
                os.rename(original_params_filename, new_params_filename)
                print(f"Renamed '{original_params_filename}' to '{new_params_filename}'")
            except Exception as e:
                print(f"Failed to rename '{original_params_filename}' to '{new_params_filename}': {e}")
                continue  # Skip to the next iteration
            
            # Now, read the renamed params file
            if os.path.exists(new_params_filename):
                with open(new_params_filename, 'r') as f:
                    lines = f.readlines()
                    try:
                        # Extract parameters from the file
                        param_dict = {}
                        for line in lines:
                            key, value = line.strip().split(':')
                            if key in parameters:
                                param_dict[key] = float(value)
                        # Append each parameter to the respective list
                        for param in parameters:
                            if param in param_dict:
                                nllc_results[asif][param].append(param_dict[param])
                                print(f"Extracted {param}={param_dict[param]} from '{new_params_filename}'")
                            else:
                                print(f"Parameter '{param}' not found in '{new_params_filename}'")
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing '{new_params_filename}': {e}")
            else:
                print(f"Renamed file '{new_params_filename}' does not exist.")
        else:
            print(f"File '{original_params_filename}' not found after running NLLc script for asif={asif}, seed={seed}")

# Calculate bootstrap mean and confidence intervals for all parameters from NLLc
bootstrap_results_nllc = {asif: {} for asif in asif_values}

for asif in asif_values:
    print(f"\nProcessing bootstrap for asif={asif}")
    for param in parameters:
        data = nllc_results[asif][param]
        if len(data) > 0:
            mean, lower, upper = bootstrap_ci(data, n_bootstrap_samples=B, ci=95)
            bootstrap_results_nllc[asif][param] = {'mean': mean, 'lower': lower, 'upper': upper}
            print(f"Asif={asif}, {param}: Mean={mean}, 95% CI=({lower}, {upper})")
        else:
            bootstrap_results_nllc[asif][param] = {'mean': np.nan, 'lower': np.nan, 'upper': np.nan}
            print(f"Asif={asif}, {param}: No data available for bootstrapping.")

# Function to plot parameters with bootstrap confidence intervals
def plot_parameters(asif_plot_values, bootstrap_results, param_list, color_map):
    """
    Plots the given parameters against asif values with confidence intervals.

    Parameters:
    - asif_plot_values: List of asif values.
    - bootstrap_results: Dictionary containing bootstrap results.
    - param_list: List of parameters to plot.
    - color_map: Dictionary mapping parameters to colors.
    """
    plt.figure(figsize=(12, 8))
    
    for param in param_list:
        means = []
        lowers = []
        uppers = []
        for asif in asif_plot_values:
            result = bootstrap_results[asif][param]
            means.append(result['mean'])
            lowers.append(result['lower'])
            uppers.append(result['upper'])
        plt.fill_between(asif_plot_values, lowers, uppers, color=color_map[param], alpha=0.2)
        plt.plot(asif_plot_values, means, marker='o', color=color_map[param], label=f'{param} Mean')
    
    # Perform linear fits where applicable
    for param in param_list:
        means = []
        for asif in asif_plot_values:
            means.append(bootstrap_results[asif][param]['mean'])
        if len(asif_plot_values) >= 2 and not np.isnan(means).any():
            coef = np.polyfit(asif_plot_values, means, 1)
            poly_fn = np.poly1d(coef)
            plt.plot(asif_plot_values, poly_fn(asif_plot_values), '--', color=color_map[param],
                     label=f'{param} Fit: $y={coef[0]:.4f}x+{coef[1]:.4f}$')
            print(f"\n{param} Fit Coefficients: Slope={coef[0]:.4f}, Intercept={coef[1]:.4f}")
    
    # Customize plot
    plt.xlabel(r'$\alpha_{\mathrm{S}}$')
    plt.ylabel(r'Parameter Values')
    plt.title(r'Parameters vs. $\alpha_{\mathrm{S}}$ with Bootstrap Confidence Intervals')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig('parameters_vs_asif_bootstrap.png', dpi=300, bbox_inches='tight')
    plt.show()

# Define colors for each parameter for consistency in plotting
color_map = {
    'lambda_1': 'blue',
    'lambda_2': 'red',
    'npm1': 'green',
    'npm2': 'orange',
    'nps1': 'purple',
    'nps2': 'brown',
    'npn1': 'pink',
    'npn2': 'gray'
}

# Define the list of parameters to plot
plot_params = ['lambda_1', 'lambda_2', 'npm1', 'npm2', 'nps1', 'nps2', 'npn1', 'npn2']

# Extract asif values that have valid bootstrap results
valid_asif_values = [asif for asif in asif_values if not all(np.isnan([bootstrap_results_nllc[asif][param]['mean'] for param in plot_params]))]

if not valid_asif_values:
    print("\nNo valid bootstrap results available for plotting.")
else:
    plot_parameters(valid_asif_values, bootstrap_results_nllc, plot_params, color_map)
