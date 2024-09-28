import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX to render text
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 4,
    "font.size": 4,
    "legend.fontsize": 4,  # Reduced font size for legend
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    "figure.figsize": (3.375, 2.5),
    "figure.dpi": 300,
    "axes.linewidth": 0.5,
    "text.latex.preamble": r"\usepackage{amsmath}",
})


# Define the values of asif and evs to iterate over
asif_values = [0.02, 0.01, 0.005, 0.0025]
evs = 100000
N = 100    # Number of runs with different seeds
B = 500  # Number of bootstrap samples

# Function to run the LL script with specific asif and seed values
def run_ll_script(asif, seed):
    command = [
        'python3', 'shower_chi2_run_thrust.py',
        '--asif', str(asif),
        '--e', str(evs),
        '--piece', 'll',
        '--nbins', '1',
        '-s', str(seed)
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the LL script with asif={asif}, seed={seed}: {e}")

# Function to run the NLLc script with specific asif, lambda_1, and seed values
def run_nllc_script(asif, lambda_1, seed):
    command = [
        'python3', 'shower_chi2_run_thrust.py',
        '--asif', str(asif),
        '--e', str(evs),
        '--lam1', str(lambda_1),
        '--piece', 'nllc',
        '--nbins', '1',
        '-K', '100',
        '-B', '0.0',
        '-s', str(seed)
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the NLLc script with asif={asif}, lambda_1={lambda_1}, seed={seed}: {e}")

# Function to run the NLL1 script with specific asif, lambda_1, and seed values
def run_nll1_script(asif, lambda_1, seed):
    command = [
        'python3', 'shower_chi2_run_thrust.py',
        '--asif', str(asif),
        '--e', str(evs),
        '--lam1', str(lambda_1),
        '--piece', 'nll1',
        '--nbins', '1',
        '-K', '0',
        '-B', '4',
        '-s', str(seed)
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the NLL1 script with asif={asif}, lambda_1={lambda_1}, seed={seed}: {e}")

# Function to perform bootstrap resampling and compute confidence intervals
def bootstrap_ci(data, n_bootstrap_samples=1000, ci=95):
    """Generate bootstrap confidence intervals for the mean."""
    bootstrapped_means = []
    n = len(data)
    for _ in range(n_bootstrap_samples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower_bound, upper_bound

# Run the LL script for each asif with different seeds and collect lambda_1 values
ll_results = {asif: [] for asif in asif_values}

for asif in asif_values:
    for seed in range(N):
        print("seed", seed)
        seed*=10000
        run_ll_script(asif, seed)
        params_ll_filename = f'params_ll_{evs}_{seed}.txt'
        if os.path.exists(params_ll_filename):
            with open(params_ll_filename, 'r') as f:
                lines = f.readlines()
                lambda_1_value = float(lines[0].split(":")[1].strip())
                ll_results[asif].append(lambda_1_value)
        else:
            print(f"File {params_ll_filename} not found for asif={asif}, seed={seed}")

# Calculate bootstrap mean and confidence intervals for lambda_1 values from LL
bootstrap_results_ll = {asif: {} for asif in asif_values}

for asif in asif_values:
    if len(ll_results[asif]) > 0:
        mean, lower, upper = bootstrap_ci(ll_results[asif], n_bootstrap_samples=B, ci=95)
        bootstrap_results_ll[asif]['mean'] = mean
        bootstrap_results_ll[asif]['lower'] = lower
        bootstrap_results_ll[asif]['upper'] = upper
    else:
        print(f"No LL results for asif={asif}")
        bootstrap_results_ll[asif]['mean'] = None
        bootstrap_results_ll[asif]['lower'] = None
        bootstrap_results_ll[asif]['upper'] = None

# Extract values for plotting LL
asif_plot_values = []
mean_values_ll = []
lower_bounds_ll = []
upper_bounds_ll = []

for asif in asif_values:
    if bootstrap_results_ll[asif]['mean'] is not None:
        asif_plot_values.append(asif)
        mean_values_ll.append(bootstrap_results_ll[asif]['mean'])
        lower_bounds_ll.append(bootstrap_results_ll[asif]['lower'])
        upper_bounds_ll.append(bootstrap_results_ll[asif]['upper'])

# Plot mean values with bootstrap confidence intervals for LL
plt.fill_between(asif_plot_values, lower_bounds_ll, upper_bounds_ll, color='blue', alpha=0.2)
plt.plot(asif_plot_values, mean_values_ll, marker='o', color='blue', label='LL Mean')

# Perform a linear fit for LL mean values
coef_ll = np.polyfit(asif_plot_values, mean_values_ll, 1)
poly1d_fn_ll = np.poly1d(coef_ll)

# Run the NLLc script for each asif using the mean lambda_1 values from LL
nllc_results = {asif: [] for asif in asif_values}

for asif in asif_values:
    lambda_1_value = bootstrap_results_ll[asif]['mean']
    if lambda_1_value is not None:
        for seed in range(N):
            run_nllc_script(asif, lambda_1_value, seed)
            params_nllc_filename = f'params_nllc_{evs}_{seed}.txt'
            if os.path.exists(params_nllc_filename):
                with open(params_nllc_filename, 'r') as f:
                    lines = f.readlines()
                    lambda_2_value_nllc = float(lines[1].split(":")[1].strip())
                    nllc_results[asif].append(lambda_2_value_nllc)
            else:
                print(f"File {params_nllc_filename} not found for asif={asif}, seed={seed}")
    else:
        print(f"No mean lambda_1 for asif={asif}, skipping NLLc runs.")

# Run the NLL1 script for each asif using the mean lambda_1 values from LL
nll1_results = {asif: [] for asif in asif_values}

for asif in asif_values:
    lambda_1_value = bootstrap_results_ll[asif]['mean']
    if lambda_1_value is not None:
        for seed in range(N):
            run_nll1_script(asif, lambda_1_value, seed)
            params_nll1_filename = f'params_nll1_{evs}_{seed}.txt'
            if os.path.exists(params_nll1_filename):
                with open(params_nll1_filename, 'r') as f:
                    lines = f.readlines()
                    lambda_2_value_nll1 = float(lines[1].split(":")[1].strip())
                    nll1_results[asif].append(lambda_2_value_nll1)
            else:
                print(f"File {params_nll1_filename} not found for asif={asif}, seed={seed}")
    else:
        print(f"No mean lambda_1 for asif={asif}, skipping NLL1 runs.")

# Calculate bootstrap mean and confidence intervals for lambda_2 values from NLLc
bootstrap_results_nllc = {asif: {} for asif in asif_values}

for asif in asif_values:
    if len(nllc_results[asif]) > 0:
        mean, lower, upper = bootstrap_ci(nllc_results[asif], n_bootstrap_samples=B, ci=95)
        bootstrap_results_nllc[asif]['mean'] = mean
        bootstrap_results_nllc[asif]['lower'] = lower
        bootstrap_results_nllc[asif]['upper'] = upper
    else:
        print(f"No NLLc results for asif={asif}")
        bootstrap_results_nllc[asif]['mean'] = None
        bootstrap_results_nllc[asif]['lower'] = None
        bootstrap_results_nllc[asif]['upper'] = None

# Calculate bootstrap mean and confidence intervals for lambda_2 values from NLL1
bootstrap_results_nll1 = {asif: {} for asif in asif_values}

for asif in asif_values:
    if len(nll1_results[asif]) > 0:
        mean, lower, upper = bootstrap_ci(nll1_results[asif], n_bootstrap_samples=B, ci=95)
        bootstrap_results_nll1[asif]['mean'] = mean
        bootstrap_results_nll1[asif]['lower'] = lower
        bootstrap_results_nll1[asif]['upper'] = upper
    else:
        print(f"No NLL1 results for asif={asif}")
        bootstrap_results_nll1[asif]['mean'] = None
        bootstrap_results_nll1[asif]['lower'] = None
        bootstrap_results_nll1[asif]['upper'] = None



# Convert asif_plot_values to a NumPy array for boolean indexing
asif_plot_values = np.array(asif_plot_values)

# Extract values for plotting NLLc and NLL1
mean_values_nllc = []
lower_bounds_nllc = []
upper_bounds_nllc = []

mean_values_nll1 = []
lower_bounds_nll1 = []
upper_bounds_nll1 = []

for asif in asif_plot_values:
    # Optional: Round asif to match dictionary keys if necessary
    rounded_asif = round(asif, 5)  # Adjust decimal places as needed
    
    # NLLc
    if bootstrap_results_nllc.get(rounded_asif) and bootstrap_results_nllc[rounded_asif]['mean'] is not None:
        mean_values_nllc.append(bootstrap_results_nllc[rounded_asif]['mean'])
        lower_bounds_nllc.append(bootstrap_results_nllc[rounded_asif]['lower'])
        upper_bounds_nllc.append(bootstrap_results_nllc[rounded_asif]['upper'])
    else:
        mean_values_nllc.append(np.nan)
        lower_bounds_nllc.append(np.nan)
        upper_bounds_nllc.append(np.nan)
    
    # NLL1
    if bootstrap_results_nll1.get(rounded_asif) and bootstrap_results_nll1[rounded_asif]['mean'] is not None:
        mean_values_nll1.append(bootstrap_results_nll1[rounded_asif]['mean'])
        lower_bounds_nll1.append(bootstrap_results_nll1[rounded_asif]['lower'])
        upper_bounds_nll1.append(bootstrap_results_nll1[rounded_asif]['upper'])
    else:
        mean_values_nll1.append(np.nan)
        lower_bounds_nll1.append(np.nan)
        upper_bounds_nll1.append(np.nan)

# Convert lists to NumPy arrays for easier manipulation
mean_values_nllc = np.array(mean_values_nllc)
lower_bounds_nllc = np.array(lower_bounds_nllc)
upper_bounds_nllc = np.array(upper_bounds_nllc)

mean_values_nll1 = np.array(mean_values_nll1)
lower_bounds_nll1 = np.array(lower_bounds_nll1)
upper_bounds_nll1 = np.array(upper_bounds_nll1)

# Plot mean values with bootstrap confidence intervals for NLLc
plt.fill_between(asif_plot_values, lower_bounds_nllc, upper_bounds_nllc, color='red', alpha=0.2)
plt.plot(asif_plot_values, mean_values_nllc, marker='s', color='red', label='NLLc Mean')

# Plot mean values with bootstrap confidence intervals for NLL1
plt.fill_between(asif_plot_values, lower_bounds_nll1, upper_bounds_nll1, color='green', alpha=0.2)
plt.plot(asif_plot_values, mean_values_nll1, marker='^', color='green', label='NLL1 Mean')

# Perform a linear fit for NLLc mean values
valid_indices_nllc = ~np.isnan(mean_values_nllc)
if np.sum(valid_indices_nllc) > 1:
    coef_nllc = np.polyfit(asif_plot_values[valid_indices_nllc], mean_values_nllc[valid_indices_nllc], 1)
    poly1d_fn_nllc = np.poly1d(coef_nllc)
    print(f"NLLc Fit Coefficients: Slope = {coef_nllc[0]:.4f}, Intercept = {coef_nllc[1]:.4f}")
else:
    coef_nllc = [np.nan, np.nan]
    poly1d_fn_nllc = lambda x: np.nan * x
    print("Not enough valid data points for NLLc fit.")

# Perform a linear fit for NLL1 mean values
valid_indices_nll1 = ~np.isnan(mean_values_nll1)
if np.sum(valid_indices_nll1) > 1:
    coef_nll1 = np.polyfit(asif_plot_values[valid_indices_nll1], mean_values_nll1[valid_indices_nll1], 1)
    poly1d_fn_nll1 = np.poly1d(coef_nll1)
    print(f"NLL1 Fit Coefficients: Slope = {coef_nll1[0]:.4f}, Intercept = {coef_nll1[1]:.4f}")
else:
    coef_nll1 = [np.nan, np.nan]
    poly1d_fn_nll1 = lambda x: np.nan * x
    print("Not enough valid data points for NLL1 fit.")

# Create new x-values for the fit lines, starting from 0
fit_x = np.linspace(0, max(asif_plot_values), 100)

# Add the linear fit line to the plot for NLLc
plt.plot(fit_x, poly1d_fn_nllc(fit_x), '--', color='red', 
         label=fr'NLLc Fit: $y = {coef_nllc[0]:.4f}x + {coef_nllc[1]:.4f}$')

# Add the linear fit line to the plot for NLL1
plt.plot(fit_x, poly1d_fn_nll1(fit_x), '--', color='green', 
         label=fr'NLL1 Fit: $y = {coef_nll1[0]:.4f}x + {coef_nll1[1]:.4f}$')

# Add the linear fit line to the plot for LL
plt.plot(fit_x, poly1d_fn_ll(fit_x), '--', color='blue', 
         label=fr'LL Fit: $y = {coef_ll[0]:.4f}x + {coef_ll[1]:.4f}$')

# Set the x-axis limit to start from 0 and extend slightly beyond max asif_plot_values for better visualization
plt.xlim(0, max(asif_plot_values) * 1.05)

# Optionally, adjust the y-axis to ensure fit lines are visible
# Find the minimum and maximum y-values including fit lines
all_fit_y = np.concatenate([poly1d_fn_nllc(fit_x), poly1d_fn_nll1(fit_x)])
y_min = min(all_fit_y.min(), 0)
y_max = max(all_fit_y.max(), np.nanmax(mean_values_nllc), np.nanmax(mean_values_nll1))
plt.ylim(y_min * 1.05, y_max * 1.05)

# Labeling with LaTeX syntax
plt.xlabel(r'$\alpha_{\text{S}}$')
plt.ylabel(r'$\lambda$')
plt.title(r'$\lambda$ vs. $\alpha_{\text{S}}$')
plt.legend(loc='best', frameon=False)

# Adjust layout to make room for labels
plt.tight_layout()

# Save the figure in high resolution
plt.savefig('lambda_vs_asif_bootstrap_prl.png', dpi=300)
plt.show()

