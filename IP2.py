import numpy as np
import matplotlib.pyplot as plt

T = 0.1
L = 1
alpha = 0.01  # alpha = k / (c * p)
n = 50
m = 50
delta_x = L / (n - 1)
delta_t = T / (m - 1)

x = np.linspace(0, L, n)
t = np.linspace(0, T, m)

# f(x) = sin(pi * x)
def f(x):
    return np.sin(np.pi * x)

# u(t, x)
u = np.zeros((m, n))

# IC: u(0,t) = u(L,t) = 0
u[:, 0] = 0
u[:, -1] = 0

# BC: u(x, 0) = f(x)
u[0, 1:n-1] = f(x[1:n-1])

# Thomas method
def thomas_method(u, delta_x, delta_t, alpha, n, m):
    r = alpha * delta_t / delta_x ** 2
    for j in range(0, m - 1):
        for i in range(1, n - 1):
            u[j + 1, i] = u[j, i] + r * (u[j + 1, i + 1] - 2 * u[j + 1, i] + u[j + 1, i - 1])
    return u

u = thomas_method(u, delta_x, delta_t, alpha, n, m)
print("Result of Euler method:\n", u)

# Add noise
noise_level = 0.01
u_noise = u[0,:] + noise_level * np.random.randn(n)


# Inverse
r = alpha * delta_t / delta_x ** 2
A = np.zeros((n, n))
for i in range(1, n - 1):
    A[i, i - 1] = r
    A[i, i] = 1 - 2 * r
    A[i, i + 1] = r

# Tichonov regulization
lmbda = [10**(-i) for i in range(20)]
for i in range(20):
    u_reg = np.linalg.solve(A.T @ A + lmbda[i] * np.eye(n), A.T @ u_noise)
    print(f"λ = {lmbda[i]}, Recovered initial condition: {u_reg}")

u_reg = np.linalg.solve(A.T @ A + lmbda[5] * np.eye(n), A.T @ u_noise)

plt.figure(figsize=(14, 8))

plt.plot(x, u[0, :], label='True initial condition $u(0,t)$', color='black', linewidth=2)
plt.plot(x, u_noise, label='Noisy measured data $u_{noise}$', linestyle='dotted', color='red', linewidth=2)

plt.xlabel('Position $x$')
plt.ylabel('Temperature $u$')
plt.title('Effect of Regularization Parameter λ on Recovered Initial Condition')
plt.legend()
plt.grid(True)
plt.show()