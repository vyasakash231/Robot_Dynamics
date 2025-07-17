import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import gamma, kv

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

from utils import *

class MIMOGaussianProcess:
    """
    Multi-Input Multi-Output Gaussian Process implementation using the Linear Model of Coregionalization (LMC) approach.
    
    This implementation extends the single-output GP to handle multiple outputs by modeling the correlations between
    different output dimensions. The key idea is to construct a valid covariance function that captures both:
    1. The spatial correlation between input points (as in single-output GP)
    2. The correlation between different outputs
    
    The covariance function takes the form:
    k((x,i), (x',j)) = B[i,j] * k_spatial(x, x')
    
    where:
    - x, x' are input vectors
    - i, j are output indices
    - B is the coregionalization matrix (captures output correlations)
    - k_spatial is the base kernel function for spatial correlation
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X_train = None
        self.Y_train = None
        self.B = np.eye(output_dim)  # Initialize coregionalization matrix

        # Default hyperparameters
        self.hyperparams = {
            'l': 1.0,           # Length scale
            'sigma_f': 1.0,     # Signal variance
            'sigma_n': 1e-6    # Noise standard deviation / Noise level
        }
        
    def set_training_data(self, X_train, Y_train):
        """
            X_train: Training inputs of shape (N, input_dim)
            Y_train: Training outputs of shape (N, output_dim)
        """
        assert X_train.shape[1] == self.input_dim, f"Expected {self.input_dim} input dimensions, got {X_train.shape[1]}"
        assert Y_train.shape[1] == self.output_dim, f"Expected {self.output_dim} output dimensions, got {Y_train.shape[1]}"
        
        self.X_train = X_train
        self.Y_train = Y_train

    def whitenoise_kernel(self, xi, xj, noise_level=1.0):
        """
        Compute the white noise kernel between two sets of points.
        
        Parameters:
        - xi: Array of shape (n, d), where n is the number of points and d is the dimensionality.
        - xj: Array of shape (m, d), where m is the number of points and d is the dimensionality.
        
        Returns:
        - Kernel matrix of shape (n, m).
        """
        xi = np.atleast_2d(xi)
        xj = np.atleast_2d(xj)

        n = xi.shape[0]
        m = xj.shape[0]
        kernel = np.zeros((n, m))  # Initialize an empty matrix
        
        # Compute the kernel matrix
        for i in range(n):
            for j in range(m):
                # Compare all elements of the arrays for equality
                # Use np.array_equal or np.allclose for robust comparison
                if np.allclose(xi[i], xj[j], rtol=1e-5, atol=1e-8):
                    kernel[i, j] = noise_level
        
        return kernel
        
        
    def gaussian_kernel(self, xi, xj, l=1.0, sigma=1.0):
        """
        Compute the Matern kernel between two sets of points.
        
        Parameters:
        - xi: Array of shape (n, d), where n is the number of points and d is the dimensionality.
        - xj: Array of shape (m, d), where m is the number of points and d is the dimensionality.
        
        Returns:
        - Kernel matrix of shape (n, m).
        """
        xi = np.atleast_2d(xi)
        xj = np.atleast_2d(xj)
        
        # Compute squared Euclidean distances efficiently
        euclid_dist = np.sum((xi[:, np.newaxis, :] - xj[np.newaxis, :, :]) ** 2, axis=2)
        return sigma**2 * np.exp(-(0.5 * euclid_dist) / l**2)
    
    '''https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/gaussian_process/kernels.py#L1598'''
    def matern_kernel(self, xi, xj, nu=1.5, l=1.0, sigma=1.0):
        """
        Compute the Matern kernel between two sets of points.
        
        Parameters:
        - xi: Array of shape (n, d), where n is the number of points and d is the dimensionality.
        - xj: Array of shape (m, d), where m is the number of points and d is the dimensionality.
        
        Returns:
        - Kernel matrix of shape (n, m).
        """
        xi = np.atleast_2d(xi)
        xj = np.atleast_2d(xj)
        
        # Compute pairwise Euclidean distances
        dist = np.sqrt(np.sum((xi[:, np.newaxis, :] - xj[np.newaxis, :, :]) ** 2, axis=2))
        
        # Avoid division by zero for zero distances
        dist[dist == 0] = 1e-8
        
        # Precompute constants
        sqrt_2nu = np.sqrt(2 * nu)
        scaled_dist = sqrt_2nu * dist / l
        
        # Compute the Matern kernel
        if nu == 0.5:
            # Matern 1/2 kernel (equivalent to exponential kernel)
            K = sigma**2 * np.exp(-scaled_dist)
        elif nu == 1.5:
            # Matern 3/2 kernel
            K = sigma**2 * (1 + scaled_dist) * np.exp(-scaled_dist)
        elif nu == 2.5:
            # Matern 5/2 kernel
            K = sigma**2 * (1 + scaled_dist + scaled_dist**2 / 3) * np.exp(-scaled_dist)
        else:
            # General Matern kernel for any nu > 0
            term1 = (2 ** (1 - self.nu)) / gamma(self.nu)
            term2 = (sqrt_2nu * dist / self.l) ** self.nu
            term3 = kv(self.nu, scaled_dist)  # Kv() -> Modified Bessel function of the second kind
            K = (self.sigma**2) * term1 * term2 * term3
        return K
    
    def mimo_kernel(self, X1, X2, l=None, sigma_f=None):
        """
        Compute the full MIMO kernel matrix incorporating both spatial and output correlations.
        
        Args:
            X1: First set of inputs (N, input_dim)
            X2: Second set of inputs (M, input_dim)
            
        Returns:
            Full kernel matrix of shape (N*output_dim, M*output_dim)
        """
        if l is None:
            l = self.hyperparams['l']
        if sigma_f is None:
            sigma_f = self.hyperparams['sigma_f']

        # K_spatial = self.gaussian_kernel(X1, X2, l, sigma_f)
        K_spatial = self.matern_kernel(X1, X2, nu=2.5, l=l, sigma=sigma_f) + self.whitenoise_kernel(X1, X2, noise_level=1e-2)
        
        # Construct the full MIMO kernel using Kronecker product
        K_full = np.kron(K_spatial, self.B)
        return K_full
    
    def posterior(self, X_test, sigma_n=None):
        """
        Compute the posterior distribution for test points.
        """
        if sigma_n is None:
            sigma_n = self.hyperparams['sigma_n']

        N_train = self.X_train.shape[0]
        N_test = X_test.shape[0]
        
        # Compute kernel matrices
        K_train = self.mimo_kernel(self.X_train, self.X_train)
        K_test = self.mimo_kernel(X_test, X_test)
        K_cross = self.mimo_kernel(self.X_train, X_test)
        
        # Add noise to training kernel
        K_train = K_train + sigma_n**2 * np.eye(N_train * self.output_dim)  # Ky = K11 + σ^{2} * I, where I is the identity matrix
        
        # Reshape Y_train for MIMO calculations
        Y_train_flat = self.Y_train.reshape(-1)
        
        try:
            # Compute posterior using Cholesky decomposition for stability
            L = np.linalg.cholesky(K_train)
            m = np.linalg.solve(L, Y_train_flat)
            alpha = np.linalg.solve(L.T, m)
            
            # Compute posterior mean and covariance
            mu_post = K_cross.T @ alpha
            v = np.linalg.solve(L, K_cross)
            cov_post = K_test - v.T @ v
            
            # Add a small jitter to ensure positive definiteness
            cov_post = cov_post + 1e-8 * np.eye(N_test * self.output_dim)
            
            # Reshape posterior mean back to (M, output_dim)
            mu_post = mu_post.reshape(N_test, self.output_dim)
            return mu_post, cov_post
        
        except np.linalg.LinAlgError:
            # Fall back to more stable but slower method if Cholesky fails
            print("Warning: Cholesky decomposition failed in posterior calculation.")
            
            # Use pseudoinverse instead
            K_inv = np.linalg.pinv(K_train)
            mu_post = (K_cross.T @ K_inv @ Y_train_flat).reshape(N_test, self.output_dim)
            cov_post = K_test - K_cross.T @ K_inv @ K_cross + 1e-8 * np.eye(N_test * self.output_dim)
            return mu_post, cov_post
    
    def predict(self, X_test, return_std=True, n_std=2.0):
        mu, cov = self.posterior(X_test, sigma_n=1e-6)
        if return_std:
            std = np.sqrt(np.diag(cov)).reshape(-1, self.output_dim)
            return mu, mu - n_std * std, mu + n_std * std
        return mu
    
    def fit_coregionalization_matrix(self, method: str = 'empirical'):
        """
        From "Kernels for Vector-Valued Functions: a Review"
        Fit the coregionalization matrix B using training data.
        method: Method to use for fitting ('empirical' or 'mle')     
        """
        if method == 'empirical':
            # Simple empirical estimation using output correlations
            self.B = np.corrcoef(self.Y_train.T)
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        
    def log_maximum_likelihood(self, params):  # params: Hyperparameters [log(l), log(sigma_f), log(sigma_n)]
        """
        Compute the log marginal likelihood of the training data.
        log(p(y|X,θ)) = - 0.5 * y^T * Ky^{-1} * y - 0.5 * log|K11| - N/2 * log(2*pi)  
        Ky = K11 + σ^{2} * I, where I is the identity matrix
        """
        # Extract and transform parameters (work with log values for optimization stability)
        l = np.exp(params[0])
        sigma_f = np.exp(params[1])
        sigma_n = np.exp(params[2])

        # Ensure minimum noise to prevent singular matrices
        sigma_n = max(sigma_n, 1e-6)

        # Reshape Y_train for MIMO calculations
        Y_train_flat = self.Y_train.reshape(-1)

        try:
            # Compute kernel matrix
            K = self.mimo_kernel(self.X_train, self.X_train, l, sigma_f)
            n = K.shape[0]
            
            # Use Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add jitter and try again
                K = K + 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
            
            # Compute log determinant using Cholesky (more stable)
            log_det_K = 2 * np.sum(np.log(np.diag(L)))
            
            # Solve linear system using Cholesky
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_flat))
            
            # Compute log likelihood
            log_likelihood = - 0.5 * Y_train_flat.T @ alpha - 0.5 * log_det_K - 0.5 * n * np.log(2 * np.pi)
            
            # Return negative log likelihood (for minimization)
            return -log_likelihood
        
        except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
            # Handle any other numerical errors
            print(f"Warning in log likelihood calculation: {e}")
            return 1e10

    def optimize_hyperparameters(self, n_restarts: int = 5, verbose: bool = True):
        """
        Optimize hyperparameters by minimizing the negative log marginal likelihood.
        n_restarts: Number of random restarts for optimization
        verbose: Whether to print optimization progress
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError("Training data must be set before optimizing hyperparameters.")
        
        # Initial hyperparameter guess (in log space)
        init_params = np.log(np.array([self.hyperparams['l'], self.hyperparams['sigma_f'], self.hyperparams['sigma_n']]))
        
        # Define bounds for hyperparameters (in log space)
        bounds = [
            (-5, 5),   # log(l)
            (-5, 5),   # log(sigma_f)
            (-7, 0)    # log(sigma_n)
        ]
        
        # Optimize from different starting points
        best_nlml = np.inf
        best_params = None
        
        if verbose:
            print("Starting hyperparameter optimization...")
        
        # First try optimization from current hyperparameters
        options = {'maxiter': 100, 'disp': False}
        try:
            result = minimize(self.log_maximum_likelihood, init_params, method='L-BFGS-B', bounds=bounds, options=options)
            best_nlml = result.fun
            best_params = result.x
        except Exception as e:
            if verbose:
                print(f"Optimization from initial point failed: {e}")
        
        # Try random restarts
        for i in range(n_restarts):
            try:
                # Random starting point (in log space)
                random_params = np.random.uniform(low=np.array([b[0] for b in bounds]), high=np.array([b[1] for b in bounds]))
                result = minimize(self.log_maximum_likelihood, random_params, method='L-BFGS-B', bounds=bounds, options=options)
                if result.fun < best_nlml:
                    best_nlml = result.fun
                    best_params = result.x
                if verbose:
                    print(f"Restart {i+1}/{n_restarts}, nlml: {result.fun:.4f}")
            except Exception as e:
                if verbose:
                    print(f"Restart {i+1}/{n_restarts} failed: {e}")
        
        # Check if optimization succeeded
        if best_params is None:
            if verbose:
                print("Optimization failed. Using initial hyperparameters.")
            return self.hyperparams
        
        # Update hyperparameters with best values
        self.hyperparams['l'] = np.exp(best_params[0])
        self.hyperparams['sigma_f'] = np.exp(best_params[1])
        self.hyperparams['sigma_n'] = np.exp(best_params[2])
        
        if verbose:
            print(f"Optimized hyperparameters:")
            print(f"  Length scale (l): {self.hyperparams['l']:.4f}")
            print(f"  Signal variance (sigma_f): {self.hyperparams['sigma_f']:.4f}")
            print(f"  Noise std (sigma_n): {self.hyperparams['sigma_n']:.4f}")
            print(f"  Final NLML: {best_nlml:.4f}")
        return self.hyperparams

    def plot_2d_outputs(self, X_test, output_names=None):
        if self.output_dim != 2:
            raise ValueError("This plotting function is only for 2D outputs")
            
        mu, lower, upper = self.predict(X_test)
        
        if output_names is None:
            output_names = [f'Output {i+1}' for i in range(2)]
            
        plt.figure(figsize=(12, 5))
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.plot(X_test[:, 0], mu[:, i], 'g-', label='Mean prediction')
            plt.fill_between(X_test[:, 0], lower[:, i], upper[:, i], color='grey', alpha=0.5, label='95% confidence')
            if self.X_train is not None:
                plt.scatter(self.X_train[:, 0], self.Y_train[:, i], c='r', marker='o', label='Training data')
            plt.xlabel('Input')
            plt.ylabel(output_names[i])
            plt.legend()
        plt.tight_layout()
        plt.show()

#########################################################################################################################################################

if __name__ == "__main__":
    # # Generate some synthetic data for a 2D input, 2D output problem
    # """ Trajectory tracking """
    # # trajectory 
    # data = np.load('../data/'+'/'+str('example')+'.npz')
    # X_train = data['demo']  # (611, 2)
    # X_train = resample(X_train, 400)  # (300, 2)

    # X_dot = np.zeros((len(X_train), 2))
    # for i in range(len(X_train)-1):
    #     X_dot[i,:] = X_train[i+1,:] - X_train[i,:]
    
    # # Create and train the MIMO GP
    # gp = MIMOGaussianProcess(input_dim=2, output_dim=2)
    # gp.set_training_data(X_train, X_dot)
    
    # # Fit the coregionalization matrix
    # gp.fit_coregionalization_matrix()
    
    # # Optimize hyperparameters
    # # optimized_params = gp.optimize_hyperparameters(n_restarts=5)
    
    # # Generate test points
    # x_grid = np.linspace(np.min(X_train[:, 0]-10), np.max(X_train[:, 0]+10), 60)
    # y_grid = np.linspace(np.min(X_train[:, 1]-10), np.max(X_train[:, 1]+10), 60)
    # dataXX, dataYY = np.meshgrid(x_grid, y_grid)
    # X_test = np.column_stack((dataXX.ravel(), dataYY.ravel()))
    
    # Y_predict = gp.predict(X_test, return_std=False)
    # u = Y_predict[:, 0].reshape(dataXX.shape)
    # v = Y_predict[:, 1].reshape(dataYY.shape)
    
    # # Plot the results
    # plt.subplot(1, 2, 1)
    # plt.scatter(X_train[:, 0], X_train[:, 1], color=[1, 0, 0])
    # # ax.quiver(X_test[:, 0], X_test[:, 1], Y_predict[:, 0], Y_predict[:, 1], alpha=0.8, color=[0, 0, 0])
    # plt.streamplot(dataXX, dataYY, u, v, color=[0, 0, 0], density=2)
    # plt.xlim([-40, 30])
    # plt.ylim([-30, 40])


    # # kernel = C(constant_value=np.sqrt(0.1)) * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(noise_level=1e-2)
    # kernel = Matern(1*np.ones(2), nu=2.5) + WhiteKernel(noise_level=1e-2)
    # GP = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=5)
    
    # GP.fit(X_train, X_dot)
    # Y_predict = GP.predict(X_test, return_std=False)
    
    # u = Y_predict[:, 0].reshape(dataXX.shape)
    # v = Y_predict[:, 1].reshape(dataYY.shape)

    # plt.subplot(1, 2, 2)
    # plt.scatter(X_train[:, 0], X_train[:, 1], color=[1, 0, 0])
    # # # ax.quiver(X_test[:, 0], X_test[:, 1], Y_predict[:, 0], Y_predict[:, 1], alpha=0.8, color=[0, 0, 0])
    # plt.streamplot(dataXX, dataYY, u, v, color=[0, 0, 0], density=2)
    # plt.xlim([-40, 30])
    # plt.ylim([-30, 40])
 
    # plt.show()

    ######################################################################################################

    # Generate some synthetic data for a 2D input, 2D output problem
    """ Trajectory tracking """
    # trajectory 
    data = np.load('../data/'+'/'+str('example')+'.npz')
    X_train = data['demo']  # (611, 2)
    X_train = resample(X_train, 400)  # (300, 2)

    T = 4
    t = np.linspace(0, T, X_train.shape[0])  # demo trajectory timing
    dt = t[1] - t[0]

    X_incremental = X_train[1:] - X_train[:-1]
    X_states = X_train[:-1]

    # X_dot = np.zeros((len(X_train), 2))
    # for i in range(len(X_train)-1):
    #     X_dot[i,:] = X_train[i+1,:] - X_train[i,:]
    
    # Create and train the MIMO GP
    gp = MIMOGaussianProcess(input_dim=2, output_dim=2)
    gp.set_training_data(X_states, X_incremental)
    
    # Fit the coregionalization matrix
    gp.fit_coregionalization_matrix()
    
    # Optimize hyperparameters
    # optimized_params = gp.optimize_hyperparameters(n_restarts=5)
    
    # Generate test points
    x_grid = np.linspace(np.min(X_train[:, 0]-10), np.max(X_train[:, 0]+10), 50)
    y_grid = np.linspace(np.min(X_train[:, 1]-10), np.max(X_train[:, 1]+10), 50)
    dataXX, dataYY = np.meshgrid(x_grid, y_grid)
    X_test = np.column_stack((dataXX.ravel(), dataYY.ravel()))
    
    Y_predict = gp.predict(X_test, return_std=False)
    u = Y_predict[:, 0].reshape(dataXX.shape)
    v = Y_predict[:, 1].reshape(dataYY.shape)
    
    # Plot the results
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], color=[1, 0, 0])
    # ax.quiver(X_test[:, 0], X_test[:, 1], Y_predict[:, 0], Y_predict[:, 1], alpha=0.8, color=[0, 0, 0])
    plt.streamplot(dataXX, dataYY, u, v, color=[0, 0, 0], density=2)
    plt.xlim([-40, 30])
    plt.ylim([-30, 40])


    kernel = C(constant_value=np.sqrt(0.1)) * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(noise_level=1e-2)
    # kernel = Matern(1*np.ones(2), nu=2.5) + WhiteKernel(noise_level=1e-2)
    GP = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=5)
    
    GP.fit(X_states, X_incremental)
    Y_predict = GP.predict(X_test, return_std=False)
    
    u = Y_predict[:, 0].reshape(dataXX.shape)
    v = Y_predict[:, 1].reshape(dataYY.shape)

    plt.subplot(1, 2, 2)
    plt.scatter(X_train[:, 0], X_train[:, 1], color=[1, 0, 0])
    # # ax.quiver(X_test[:, 0], X_test[:, 1], Y_predict[:, 0], Y_predict[:, 1], alpha=0.8, color=[0, 0, 0])
    plt.streamplot(dataXX, dataYY, u, v, color=[0, 0, 0], density=2)
    plt.xlim([-40, 30])
    plt.ylim([-30, 40])
 
    plt.show()