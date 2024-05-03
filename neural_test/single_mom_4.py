import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Given constants, explicitly cast to tf.float32
alpha_s = tf.constant(0.1, dtype=tf.float32)  # Assuming alpha_s is given as 0.2 for example purposes
pi = tf.constant(np.pi, dtype=tf.float32)     # Cast pi to tf.float32

# Define the prior distribution q(τ)
def q_tau(tau):
    tau = tf.cast(tau, dtype=tf.float32)  # Ensure tau is also cast to tf.float32
    return (2 * alpha_s / (3 * pi * tau)) * (-4 * tf.math.log(tau))

def p_true_tau(tau):
    tau = tf.cast(tau, dtype=tf.float32)  # Ensure tau is also cast to tf.float32
    return (2 * alpha_s / (3 * pi * tau)) * (-4 * tf.math.log(tau)) * tf.math.exp(- 2 * alpha_s / (3 * pi) * (2 * tf.math.log(tau)**2))

class ProbabilityDensityNetwork(Model):
    def __init__(self):
        super(ProbabilityDensityNetwork,self).__init__()
        # Increased number of layers and neurons in each layer to capture more complex patterns
        self.dense_1 = layers.Dense(256, activation='relu')
        self.dense_2 = layers.Dense(256, activation='relu')
        self.dense_3 = layers.Dense(128, activation='relu')
        self.dense_4 = layers.Dense(128, activation='relu')
        self.dense_5 = layers.Dense(64, activation='relu')
        self.dense_6 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1, activation='softplus')  # Ensures positive output

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)
        return self.output_layer(x)



# Instantiate the neural network with potentially more complex architecture
p_network = ProbabilityDensityNetwork()

# Initiating Lagrange multipliers
lambda_0 = tf.Variable(-1*0.5, name='lambda_0', dtype=tf.float32)
lambda_1 = tf.Variable(0.5*4 * alpha_s / (3 * pi), name='lambda_1', dtype=tf.float32)

# Training constants
learning_rate = 0.01
num_steps = 2000
num_samples = 1000  # Number of samples to approximate the integrals

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# Loss function with added numerical stability
def loss_function(lambda_0, lambda_1):
    epsilon = 1e-7  # Small constant to prevent log(0)
    tau_samples = tf.random.uniform((num_samples, 1), minval=0, maxval=1.0)
    #print(tau_samples)
    q_tau_values = q_tau(tau_samples)
    p_tau_values = p_network(tau_samples)

    p_tau_true_values = p_true_tau(tau_samples)
    #print(tf.reduce_mean(p_tau_true_values))
    # Ensure p_tau_values are bounded away from zero
    p_tau_values = tf.clip_by_value(p_tau_values, epsilon, np.inf)

    # Calculate the KL divergence part of the loss
    kl_divergence = tf.reduce_mean(p_tau_values * (tf.math.log(p_tau_values + epsilon) - tf.math.log(q_tau_values + epsilon)))

    # Constraints
    constraint_1 = tf.reduce_mean(p_tau_values)
    constraint_2 = tf.reduce_mean(p_tau_values * tf.math.log(tau_samples + epsilon)**2)

    # Theory inputs
    cc_1 = 1
    cc_2 = (3 * pi)/(4 * alpha_s) 
    #print(tf.reduce_mean(p_tau_true_values)-cc_1)
    #print(tf.reduce_mean(p_tau_true_values * (tf.math.log(tau_samples))**2)-cc_2)
    # Combine the KL divergence and constraints with Lagrange multipliers
    
    loss = kl_divergence - lambda_0 * (constraint_1 - cc_1) - lambda_1 *( constraint_2 - cc_2)
    loss_2 = kl_divergence
    #lambda_1 = 3 * pi * alpha_s / 4
    
    return loss_2

# Define a range of τ values for plotting
tau_values = np.linspace(0.1, 1.0, 10000)


for step in range(num_steps):
    with tf.GradientTape() as tape:
        loss = loss_function(lambda_0, lambda_1)
    #Compute the gradients for the trainable variables and the Lagrange multipliers
        gradients = tape.gradient(loss, p_network.trainable_variables + [lambda_0, lambda_1])


    # Apply the clipped gradients
    optimizer.apply_gradients(zip(gradients, p_network.trainable_variables + [lambda_0, lambda_1]))

    if step % 100 == 0 or step == num_steps - 1:
     # Print debugging information
     print(f"Step: {step}, Loss: {loss.numpy()}")
     print(f"Lambda_0: {lambda_0.numpy()}, Lambda_1: {lambda_1.numpy()}")
     # Convert numpy array tau_values to a TensorFlow tensor
     tau_values_tf = tf.constant(tau_values, dtype=tf.float32)

     # Evaluate p(τ) using the neural network
     p_values = p_network(tau_values_tf[:, tf.newaxis]).numpy().flatten()

     # Evaluate q(τ) for the same tau values
     q_values = q_tau(tau_values_tf).numpy()

     # Evaluate p_true(τ) for the same tau values
     p_true_values = p_true_tau(tau_values_tf).numpy()

     # Plotting the distributions
     plt.figure(figsize=(10, 5))
     plt.plot(tau_values, p_values, label=f'p(τ) at step {step}', color='blue')
    # plt.plot(tau_values, p_true_values, label=f'p_true(τ) at step {step}', color='green')
    # plt.plot(tau_values, q_values, label='q(τ)', color='orange', linestyle='dashed')
     plt.title(f'Step {step}: p(τ) vs q(τ)')
     plt.xlabel('τ')
     plt.ylabel('Probability Density')
     plt.legend()
     plt.show()
