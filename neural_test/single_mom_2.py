import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

# Define the test distribution q(x) - here as a normal distribution with mean 0 and standard deviation 1
q_mean = 0.0
q_stddev = 1.0
fixed_distribution = tfp.distributions.Normal(loc=q_mean, scale=q_stddev)


# Assume some initial distributions for p(x) and q(x)
# For example, both p and q can be normal distributions, 
# but q is fixed (prior) and p is trainable (to be optimized)
#fixed_distribution = tfp.distributions.Normal(loc=0.0, scale=1.0)
# Assume some initial distributions for p(x) and q(x)
trainable_loc = tf.Variable(0.0, name='trainable_loc')
trainable_scale = tf.Variable(1.0, name='trainable_scale')
trainable_distribution = tfp.distributions.Normal(loc=trainable_loc, scale=trainable_scale)

# Number of samples for Monte Carlo approximation
num_samples = 1000

# Lagrange multipliers (initialized to some value, e.g., 1.0)
lambda_0 = tf.Variable(1.0, dtype=tf.float32)
lambda_1 = tf.Variable(1.0, dtype=tf.float32)
lambda_2 = tf.Variable(1.0, dtype=tf.float32)

# Constants c1, c2, ..., for the constraints
c1 = 1.0  # example value
c2 = 2.0  # example value

# Loss function with Monte Carlo approximation of integrals
def loss_function():
    # Sample from p(x) for Monte Carlo integration
    samples = trainable_distribution.sample(num_samples)

    # Compute the Monte Carlo approximation of the integrals
    integral_1 = tf.reduce_mean(tf.math.log(trainable_distribution.prob(samples) / fixed_distribution.prob(samples)))
    integral_2 = tf.reduce_mean(trainable_distribution.prob(samples))
    integral_3 = tf.reduce_mean(trainable_distribution.prob(samples) * samples)  # g1(x) = x for example
    integral_4 = tf.reduce_mean(trainable_distribution.prob(samples) * samples**2)  # g2(x) = x^2 for example

    # Loss with Lagrange multipliers for constraints
    loss = -integral_1 - lambda_0 * (integral_2 - 1) - lambda_1 * (integral_3 - c1) - lambda_2 * (integral_4 - c2)
    return loss

# Optimizer
optimizer = tf.optimizers.Adam()

# Training step
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        # Make sure to watch the variables if they are not automatically being tracked
        tape.watch([trainable_loc, trainable_scale, lambda_0, lambda_1, lambda_2])
        loss = loss_function()

    # Compute the gradients with respect to the variables
    gradients = tape.gradient(loss, [trainable_loc, trainable_scale, lambda_0, lambda_1, lambda_2])
    
    # Apply the gradients to the variables
    optimizer.apply_gradients(zip(gradients, [trainable_loc, trainable_scale, lambda_0, lambda_1, lambda_2]))

# Training loop
for step in range(1000):  # Adjust the number of steps as necessary
    train_step()
    if step % 100 == 0:
        print("Step:", step, "Loss:", loss_function().numpy())
        print(f"Lambda_0: {lambda_0.numpy()}, Lambda_1: {lambda_1.numpy()}, Lambda_2: {lambda_2.numpy()}")

import matplotlib.pyplot as plt
import numpy as np

# Plot p(x) vs q(x) at each training step

# Define a range of x values for plotting the PDFs
x_values = np.linspace(-5, 5, 100)

# Convert x_values to a TensorFlow constant for computation
x_values_tf = tf.constant(x_values, dtype=tf.float32)

# Training loop with plotting
for step in range(1000):  # Adjust the number of steps as necessary
    train_step()
    if step % 100 == 0:  # Adjust as needed
        print(f"Step: {step}, Loss: {loss_function().numpy()}")
        print(f"Lambda_0: {lambda_0.numpy()}, Lambda_1: {lambda_1.numpy()}, Lambda_2: {lambda_2.numpy()}")

        # Compute the PDF of the trainable distribution (p(x)) and the fixed distribution (q(x))
        p_pdf = tfp.distributions.Normal(loc=trainable_loc, scale=trainable_scale).prob(x_values_tf)
        q_pdf = fixed_distribution.prob(x_values_tf)

        # Convert PDFs to numpy arrays for plotting
        p_pdf_np = p_pdf.numpy()
        q_pdf_np = q_pdf.numpy()

        # Plot the PDFs
        plt.figure(figsize=(10, 5))
        plt.plot(x_values, p_pdf_np, label='p(x)', color='blue')
        plt.plot(x_values, q_pdf_np, label='q(x)', color='orange')
        plt.title(f'PDFs of p(x) and q(x) at step {step}')
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        plt.show()

