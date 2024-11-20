# main_runner.py

import argparse
import data_generator
import analytics_NLL
import optimizer
import torch

def compute_weights(tau_values, learned_lambdas):
    # tau_values: torch tensor of tau values
    # learned_lambdas: dictionary containing 'lambda_0', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4'

    log_tau = torch.log(tau_values)
    lambda_0 = torch.tensor(learned_lambdas['lambda_0'], dtype=torch.float64)
    lambda_1 = torch.tensor(learned_lambdas['lambda_1'], dtype=torch.float64)
    lambda_2 = torch.tensor(learned_lambdas['lambda_2'], dtype=torch.float64)
    lambda_3 = torch.tensor(learned_lambdas['lambda_3'], dtype=torch.float64)
    lambda_4 = torch.tensor(learned_lambdas['lambda_4'], dtype=torch.float64)


    # Compute the exponent
    exponent = -lambda_0 - lambda_1 * log_tau - lambda_2 * log_tau**2- lambda_3 * log_tau**3- lambda_4 * log_tau**4

    weights = torch.exp(exponent)

    return weights

def plot_distributions(args, tau_i, analytic_moments, learned_lambdas, min_tau, max_tau):
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    # Unpack the analytics object
    analytics = analytic_moments['analytics']

    # Set desired log10(tau) range
    min_log_tau = -3.0
    max_log_tau = 0.99999  # Adjust to desired maximum

    # Create a range of log10(tau) values
    tau_log_range = torch.linspace(min_log_tau, max_log_tau, steps=1000)

    # Convert log10(tau) back to tau
    tau_range = 10 ** tau_log_range

    # Compute the analytic distribution
    def wLL(tau):
        partC = (analytics.R_LLp(tau) + analytics.R_NLLp(tau) + analytics.FpF(tau)) / tau
        exponC = torch.exp(-analytics.R_LL(tau) - analytics.R_NLL(tau) - analytics.logF(tau))
        return partC * exponC

    # Calculate the analytic distribution values
    analytic_vals = wLL(tau_range).detach().numpy()

    # Convert analytic_vals to probability density in log space
    # dP/dlog10(tau) = dP/dtau * dtau/dlog10(tau) = dP/dtau * tau * ln(10)
    analytic_vals_log = analytic_vals * tau_range.detach().numpy() * np.log(10)

    # Get the thrust data and original weights from tau_i
    tau_data = tau_i[:, 0]
    original_weights = tau_i[:, 1]

    # Compute new weights using learned lambdas
    new_weights = compute_weights(tau_data, learned_lambdas)

    # Convert tau_data to log10(tau) to avoid log(0)
    tau_data_np = tau_data.detach().numpy()
    tau_data_log = np.log10(np.maximum(tau_data_np, 1e-10))

    # Filter data within the desired range
    data_mask = (tau_data_log >= min_log_tau) & (tau_data_log <= max_log_tau)
    tau_data_log = tau_data_log[data_mask]
    new_weights_np = new_weights.detach().numpy()[data_mask]

    # Define histogram bins
    nbins = args.nbins
    bins = np.linspace(min_log_tau, max_log_tau, nbins + 1)

    # Plot the histogram of the unweighted thrust data
    plt.figure(figsize=(10, 6))
    counts_unweighted, bins_unweighted, _ = plt.hist(
        tau_data_log,
        bins=bins,
        density=True,
        alpha=0.5,
        label='Unweighted Thrust Data',
        histtype='stepfilled',
        color='blue'
    )

    # Plot the histogram of the weighted thrust data (using new weights)
    counts_weighted, bins_weighted, _ = plt.hist(
        tau_data_log,
        bins=bins,
        weights=new_weights_np,
        density=True,
        alpha=0.5,
        label='Weighted Thrust Data',
        histtype='step',
        linewidth=2,
        color='green'
    )

    # Plot the analytic distribution
    plt.plot(
        tau_log_range.detach().numpy(),
        analytic_vals_log,
        'r-',
        label='Analytic Distribution',
        linewidth=2
    )

    # Set plot labels and title
    plt.xlabel('$\\log_{10}(\\tau)$', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('Comparison of Thrust Data and Analytic Distribution', fontsize=16)
    plt.legend()
    plt.grid(True)

    # Set x-axis limits
    plt.xlim(min_log_tau, max_log_tau)

    # Optionally, adjust tick labels for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show the plot
    plt.show()

def main():
    import torch
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run data generation, analytics, and optimization.')

    # Existing arguments
    parser.add_argument('--e', type=int, default=20000, help='Number of samples to process')
    parser.add_argument('--asif', type=float, default=0.118, help='alphas limit')
    parser.add_argument('--piece', default='all', help='piece to fit')
    parser.add_argument('--min', type=float, default=0)
    parser.add_argument('--max', type=float, default=1)
    parser.add_argument('-K', type=float, default=1)
    parser.add_argument('-B', type=float, default=1)
    parser.add_argument('-C', type=float, default=0.01)
    parser.add_argument('-s', type=int, default=1234)
    parser.add_argument('-F', type=float, default=1)
    parser.add_argument('--nbins', type=int, default=50)  # Adjusted for better resolution
    parser.add_argument("-n", "--nem", default=2, dest="nem")
    # ... other arguments as needed ...

    # Add missing lambda arguments with default values
    parser.add_argument('--lam1', type=float, default=1e-3, help='Initial value for lambda_1')
    parser.add_argument('--lam2', type=float, default=1e-4, help='Initial value for lambda_2')
    parser.add_argument('--lam3', type=float, default=1e-5, help='Initial value for lambda_3')
    parser.add_argument('--lam4', type=float, default=1e-6, help='Initial value for lambda_4')

    args = parser.parse_args()

    # Step 1: Data Generation
    tau_i, min_tau, max_tau = data_generator.generate_data(args)

    # Step 2: Analytics and Calculating the Analytic Moments
    analytic_moments = analytics_NLL.calculate_analytic_moments(args, min_tau, max_tau)

    # Step 3: Optimization
    learned_lambdas = optimizer.run_optimization(args, tau_i, analytic_moments)

    # Plotting the analytic and thrust distributions
    plot_distributions(args, tau_i, analytic_moments, learned_lambdas, min_tau, max_tau)

if __name__ == "__main__":
    main()
