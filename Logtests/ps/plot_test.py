import matplotlib.pyplot as plt
import numpy as np

# Data provided
asif_values = [0.06, 0.04, 0.0025]
lambda_2_values = [0.5798464417457581, 0.5644577741622925,0.5463507771492004]

# Perform a quadratic fit
coef = np.polyfit(asif_values, lambda_2_values, 1)
poly2d_fn = np.poly1d(coef)

# Generate values for plotting the fit line
asif_fit = np.linspace(min(asif_values), max(asif_values), 100)
lambda_2_fit = poly2d_fn(asif_fit)

# Plot lambda_2 vs. asif
plt.plot(asif_values, lambda_2_values, 'o', label='Data')
plt.plot(asif_fit, lambda_2_fit, '--k', label=f'Fit: y = {coef[0]:.4f}x^2 + {coef[1]:.4f}x')

plt.xlabel('asif')
plt.ylabel('lambda_2')
plt.title('lambda_2 vs. asif')
plt.legend()
plt.savefig('lambda_2_vs_asif.png')
plt.show()

# Print fit coefficients
print(f"Fit coefficients: y = {coef[0]:.4f}x^2 + {coef[1]:.4f}x ")

