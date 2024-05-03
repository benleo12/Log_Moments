import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Assuming alpha_s is a known constant
alpha_s = 0.118
n_samp=100
threshold = 0.0033  # LQCD/MZ
max_iterations = 1000  # Limit on the number of iterations
# Dataset of {tau_i, q_tau_i} pairs
kap_1 = -6*alpha_s/(3*np.pi)/2
kap_2 = 4*alpha_s/(3*np.pi)

##### USE alphaS(...) in QCD.py for RUNNNIIINNNNGGGGG use nf=5

def corrected_update_tau(r_i, tau_i, kap_1, kap_2):
    return torch.exp(-np.sqrt(-(torch.log(r_i) - kap_2 * torch.log(tau_i) ** 2 - kap_1 * torch.log(tau_i)) / kap_2 + kap_1 ** 2 / (4 * kap_2 ** 2)) - kap_1 / (2 * kap_2))



# Manual integration function using Riemann sum
def mc_integration(integrand, tau_values, lambda_2):
    integral = torch.mean(integrand)
    return integral

# Define the integral equations using the dataset directly
def integral_equation_1_direct(lambda_0, lambda_1, lambda_2,sum_tau_i):
    part = (2*(kap_2+lambda_2)*torch.log(sum_tau_i)+(kap_1+lambda_1))/(2*kap_2*torch.log(sum_tau_i)+kap_1)
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(sum_tau_i) - lambda_2 * torch.log(sum_tau_i)**2)
    integral = mc_integration(expon*part, sum_tau_i, lambda_2)
    return integral - 1

def integral_equation_2_direct(lambda_0, lambda_1, lambda_2,sum_tau_i):
    part = (2*(kap_2+lambda_2)*torch.log(sum_tau_i)+(kap_1+lambda_1))/(2*kap_2*torch.log(sum_tau_i)+kap_1)
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(sum_tau_i) - lambda_2 * torch.log(sum_tau_i)**2) * torch.log(sum_tau_i)
    integral = mc_integration(expon*part, sum_tau_i,lambda_2)
    return integral + (np.sqrt(3)*np.pi*np.exp(3*alpha_s/(4*np.pi))*erfc(np.sqrt(alpha_s)/2*np.sqrt(3/np.pi))/(4*np.sqrt(alpha_s)))

def integral_equation_3_direct(lambda_0, lambda_1, lambda_2,sum_tau_i):
    part = (2*(kap_2+lambda_2)*torch.log(sum_tau_i)+(kap_1+lambda_1))/(2*kap_2*torch.log(sum_tau_i)+kap_1)
    expon = torch.exp(-1 - lambda_0 - lambda_1 * torch.log(sum_tau_i) - lambda_2 * torch.log(sum_tau_i)**2) * torch.log(sum_tau_i)**2
    integral = mc_integration(expon*part, sum_tau_i, lambda_2)
    return integral + 3*np.pi/(8*alpha_s)*(-2 + np.sqrt(3)*np.sqrt(alpha_s)*np.exp(3*alpha_s/(4*np.pi))*erfc(np.sqrt(alpha_s)/2*np.sqrt(3/np.pi)) )

# Initialize the Lagrange multipliers
lambda_0 = torch.tensor([-1.0], requires_grad=True)
lambda_1 = torch.tensor([-6 * alpha_s / (3 * np.pi)*0], requires_grad=True)
lambda_2 = torch.tensor([4 * alpha_s / (3 * np.pi)*0], requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([lambda_1], lr=0.01)
print("kap_1 = ", kap_1)
print("kap_2 = ", kap_2)
# Optimization loop
nstep = 2000
# Lists to collect data
lambda_1_values = []
lambda_2_values = []
loss_values = []
n_samp_values = []
for step in range(1000000):
    r_i = torch.rand(n_samp,max_iterations)  # Reinitialize r_i for clarity
    tau_i = torch.ones(n_samp)


    # placeholders
    sum_tau_i = torch.zeros(n_samp)
    iterations_count = torch.zeros(n_samp, dtype=torch.int)

    for i in range(n_samp):
     tau_temp = tau_i[i]  # Work with a temporary variable for updates
     sum_tau = 0.0  # Initialize sum for the current sample
     iterations = 0  # Reset iteration count for the current sample

    # Ensure that j iterates from 0 up to max_iterations-1
     for j in range(max_iterations):
        if tau_temp < threshold:
            break  # Exit the loop if tau_temp is below the threshold
        tau_temp = corrected_update_tau(r_i[i][j], tau_temp, kap_1, kap_2)
        sum_tau += tau_temp  # Accumulate sum of tau values
        iterations += 1  # Increment iteration count

    # At this point, you can use sum_tau or iterations as needed


     if iterations == max_iterations:
        # If the process exceeded max_iterations, discard the result for this sample
        sum_tau_i[i] = -1  # Sentinel value indicating discard
     else:
        # Otherwise, compute the average and store it
        sum_tau_i[i] = sum_tau   # Compute average based on actual iterations
     iterations_count[i] = iterations

    # Create a mask to identify values that should not be discarded
    valid_mask = sum_tau_i != -1

    # Apply the mask to sum_tau_i to keep only valid terms
    filtered_sum_tau_i = sum_tau_i[valid_mask]

    #if step % (10) == 0:
        #print("iterations = ",iterations_count)

    optimizer.zero_grad()
    loss_1 = integral_equation_1_direct(lambda_0, lambda_1, lambda_2, filtered_sum_tau_i)**2
    loss_2 = integral_equation_2_direct(lambda_0, lambda_1, lambda_2, filtered_sum_tau_i)**2
    loss_3 = integral_equation_3_direct(lambda_0, lambda_1, lambda_2, filtered_sum_tau_i)**2
    loss = loss_1 + loss_2 + loss_3  # Total loss


    if torch.isnan(loss):
     #print("sad, tau=", torch.sum(torch.isnan(tau_i)))
     #print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
     continue
    loss.backward()
    optimizer.step()

   # print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")

    if step >0 and step % (nstep+5) == 0:
        print(f"Step {step}, Loss: {loss.item()}, Lambda 0: {lambda_0.item()}, Lambda 1: {lambda_1.item()}, Lambda 2: {lambda_2.item()}")
        n_samp = int(n_samp*5)
        nstep = int(nstep/2.5)
        print("nstep = ", nstep)
        lambda_1_values.append(lambda_1.item())
        lambda_2_values.append(lambda_2.item())
        loss_values.append(loss.item())
        n_samp_values.append(n_samp)
    if nstep <= 5:
        break



# Plotting outside the loop

# Assuming n_samp_values and lambda_1_values are defined

plt.figure(figsize=(10, 6))
plt.plot(n_samp_values, lambda_1_values, 'o-', color='tab:red')  # Plotting with markers and lines
plt.xscale('log')  # Applying log scale to the x-axis
plt.xlabel(r'$n_{samp}$', fontsize=14)
plt.ylabel(r'$\lambda_1$', color='tab:red', fontsize=14)
plt.title('Log moment progression', fontsize=16)
plt.grid(True, which="both", ls="--")  # Adding grid lines for readability
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_samp_values, lambda_2_values, 'o-', color='tab:red')  # Plotting with markers and lines
plt.xscale('log')  # Applying log scale to the x-axis
plt.xlabel(r'$n_{samp}$', fontsize=14)
plt.ylabel(r'$\lambda_2$', color='tab:red', fontsize=14)
plt.title('Log moment progression', fontsize=16)
plt.grid(True, which="both", ls="--")  # Adding grid lines for readability
plt.show()

# Log-log plot for loss
plt.figure(figsize=(10, 6))
plt.loglog(n_samp_values, loss_values, color='tab:blue')
plt.xlabel(r'$n_{samp}$', fontsize=14)
plt.ylabel('Loss', color='tab:blue', fontsize=14)
plt.title('Loss progression', fontsize=16)
plt.tick_params(axis='y', labelcolor='tab:blue')
plt.grid(False, which="both", ls="--")  # Adding grid lines for readability
plt.tight_layout()
plt.show()


# Print final values of the Lagrange multipliers

print(f"Final Lambda 0: {lambda_0.item()}, Final Lambda 1: {lambda_1.item()}, Final Lambda 2: {lambda_2.item()}")
