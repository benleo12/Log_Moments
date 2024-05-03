import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Given constants, explicitly cast to tf.float32
alpha_s = tf.constant(0.2, dtype=tf.float32)  # Assuming alpha_s is given as 0.2 for example purposes
pi = tf.constant(np.pi, dtype=tf.float32)     # Cast pi to tf.float32

# Define the prior distribution q(τ)
def q_tau(tau):
    tau = tf.cast(tau, dtype=tf.float32)  # Ensure tau is also cast to tf.float32
    return (2 * alpha_s / (3 * pi * tau)) * (-4 * tf.math.log(tau))

# Initialize variables for p(τ) parameters and Lagrange multipliers
p_params = tf.Variable([0.0, 1.0], name='p_params', dtype=tf.float32)  # Initialize as float32
lambda_0 = tf.Variable(-1, name='lambda_0', dtype=tf.float32)        # Initialize as float32
lambda_1 = tf.Variable(4*alpha_s/3/pi, name='lambda_1', dtype=tf.float32)  # Initialize as float32

# Training constants
learning_rate = 0.00001
num_steps = 10000
num_samples = 10000  # Number of samples to approximate the integrals

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# Define the trainable distribution p(τ)
def p_distribution(tau, params):
    loc, scale = params[0], params[1]
    normal_dist = tfp.distributions.Normal(loc=loc, scale=scale)
    return normal_dist.prob(tau)

# Loss function
def loss_function(params, lambda_0, lambda_1):
    tau_samples = tf.random.uniform(shape=(num_samples,), minval=0.01, maxval=1.0)
    q_tau_values = q_tau(tau_samples)
    p_tau_values = p_distribution(tau_samples, params)

    # Calculate the KL divergence part of the loss
    kl_divergence = tf.reduce_mean(p_tau_values * (tf.math.log(p_tau_values) - tf.math.log(q_tau_values)))

    # Constraints
    constraint_1 = tf.reduce_mean(p_tau_values) - 1  # Integral of p(τ) should equal 1
    constraint_2 = tf.reduce_mean(p_tau_values * tf.math.log(tau_samples)**2) - constraint_1  # Integral of p(τ) * ln^2(τ) should equal c1

    # Combine the KL divergence and constraints with Lagrange multipliers
    loss = -kl_divergence - lambda_0 * constraint_1 - lambda_1 * constraint_2

    return loss




# Training loop
for step in range(num_steps):
    with tf.GradientTape() as tape:
        tape.watch(p_params)  # Ensure the gradient tape is watching p_params
        loss = loss_function(p_params, lambda_0, lambda_1)
    gradients = tape.gradient(loss, [p_params, lambda_0, lambda_1])
    optimizer.apply_gradients(zip(gradients, [p_params, lambda_0, lambda_1]))
    
    if step % 100 == 0 or step == num_steps - 1:
        
        print(f"Step: {step}, Loss: {loss.numpy()}, Lambda_0: {lambda_0.numpy()}, Lambda_1: {lambda_1.numpy()}")

        # Plotting the distributions
        tau_values = np.linspace(0.01, 1.0, 500)
        q_values = q_tau(tau_values).numpy()
        p_values = p_distribution(tau_values, p_params.numpy()).numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(tau_values, p_values, label=f'p(τ) at step {step}', color='blue')
        plt.plot(tau_values, q_values, label='q(τ)', color='orange', linestyle='dashed')
        plt.title(f'p(τ) vs q(τ) at Step {step}')
        plt.xlabel('τ')
       

