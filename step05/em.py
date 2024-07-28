import os
import numpy as np

def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2*np.pi) **D * det)
    y = z * np.exp(-1/2 * (x-mu).T @ inv @ (x-mu))
    return y

def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        y += phis[k] * multivariate_normal(x, mus[k], covs[k])
    return y

def likelihood(xs, phis, mus, covs):
    eps = 1e-8
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y+eps)
    return L/N

xs = np.loadtxt("step05/old_faithful.txt")
N = xs.shape[0]
K = xs.shape[1]

# init
phis = np.ones(K) / K
mus = np.array([[0, 50],[0, 100]])
covs = np.array([np.eye(K), np.eye(K)])

MAX_ITERS = 100
THRESHOLD = 1e-4

current_likelihood = likelihood(xs, phis, mus, covs)
for iter in range(MAX_ITERS):
    # E: expectation
    qs = np.zeros((N, K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            qs[n, k] = phis[k] * multivariate_normal(x, mus[k], covs[k])
        qs[n] /= gmm(x, phis, mus, covs)

    # M: maximization
    qs_sum = qs.sum(axis=0)
    for k in range(K):
        # phis
        phis[k] = qs_sum[k] / N

        # mus
        c = 0
        for n in range(N):
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]

        # covs
        c = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[:, np.newaxis]
            c += qs[n, k] * z @ z.T
        covs[k] = c / qs_sum[k]
    
    print(f"{current_likelihood:.3f}")
    next_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(next_likelihood - current_likelihood)
    if diff < THRESHOLD:
        break
    current_likelihood = next_likelihood

# visualize
import matplotlib.pyplot as plt
def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z)

plt.scatter(xs[:,0], xs[:,1])
plot_contour(phis, mus, covs)
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()

N = 500
new_xs = np.zeros((N, 2))
for n in range(N):
    k = np.random.choice(2, p=phis)
    mu, cov = mus[k], covs[k]
    new_xs[n] = np.random.multivariate_normal(mu, cov)

# visualize
plt.scatter(xs[:,0], xs[:,1], alpha=0.7, label='original')
plt.scatter(new_xs[:,0], new_xs[:,1], alpha=0.7, label='generated')
plt.legend()
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()