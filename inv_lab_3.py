import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(np.pi * x)

def forward_euler(u, dx, dt, alpha, n, m):
    r = alpha * dt / dx ** 2
    for j in range(0, m - 1):
        for i in range(1, n - 1):
            u[j + 1, i] = u[j, i] + r * (u[j, i + 1] - 2 * u[j, i] + u[j, i - 1])
    return u

def compute_gradient(f_x, u_noisy, A, lam):
    return 2 * A.T @ (A @ f_x - u_noisy) + 2 * lam * f_x

def visualize_results(x, u_true, u_noisy, u_recovered):
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_true, label='Original state $u(0,t)$', color='black')
    plt.plot(x, u_noisy, label='Measured noisy data $u_{noise}$', linestyle='dotted', color='red')
    plt.plot(x, u_recovered, label='Recovered state $u_{recovered}$', linestyle='--', color='blue')

    plt.xlabel('Position $x$')
    plt.ylabel('Temperature $u$')
    plt.title('Heat Equation: Comparison of Initial States')
    plt.legend()
    plt.grid(True)
    plt.show()


# Parameters
T = 0.1
L = 1
alpha = 0.01  
n = 45
m = 30
delta_x = L / (n - 1)
delta_t = T / (m - 1)

x = np.linspace(0, L, n)
t = np.linspace(0, T, m)

u_matrix = np.zeros((m, n))

u_matrix[:, 0] = 0
u_matrix[:, -1] = 0

u_matrix[0, 1:n-1] = f(x[1:n-1])

u_matrix = forward_euler(u_matrix, delta_x, delta_t, alpha, n, m)
print("Calculated solution using the modified Euler method:\n", u_matrix)

noise_lvl = 0.05
u_noisy = u_matrix[0, :] + noise_lvl

# Inverse problem setup
r = alpha * delta_t / delta_x ** 2
A_matrix = np.zeros((n, n))
for i in range(1, n - 1):
    A_matrix[i, i - 1] = r
    A_matrix[i, i] = 1 - 2 * r
    A_matrix[i, i + 1] = r

lambdas = [10**(-i) for i in range(15)]
learning_rate = 0.0001
epsilon_threshold = 0.05

f_x = u_matrix[0, :]
f_estimate = f_x.copy()
b = True

while b == True:
    grad = compute_gradient(f_estimate, u_noisy, A_matrix, lambdas[5])
    f_next = f_estimate - learning_rate * grad

    if np.linalg.norm(f_next - f_estimate) <= epsilon_threshold:
        b = False
    else:
        f_estimate = f_next

u_matrix[0, :] = f_estimate
u_noisy_new = u_matrix[0, :] + noise_lvl

# Regularization
for i in range(15):
    u_recovered = np.linalg.solve(A_matrix.T @ A_matrix + lambdas[i] * np.eye(n), A_matrix.T @ u_noisy_new)
    print(f"λ = {lambdas[i]}, Estimated initial state: {u_recovered}")

u_recovered = np.linalg.solve(A_matrix.T @ A_matrix + lambdas[5] * np.eye(n), A_matrix.T @ u_noisy_new)

visualize_results(x, u_matrix[0, :], u_noisy_new, u_recovered)
