import deepxde as dde
import numpy as np
import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('ground_ecscavation_data.csv')

# Assuming the data has columns: 'Time', 'X', 'Y', 'Z', 'dX', 'dY', 'dZ'
# Extract relevant columns
time = data['Time'].values
coordinates = data[['X', 'Y', 'Z']].values
deformation = data[['dX', 'dY', 'dZ']].values

# Define the Navier-Cauchy equations
def navier_cauchy(x, y):
    # x: input coordinates (time and space)
    # y: output displacement vector (u)
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]

    # Compute necessary derivatives
    u_t = dde.grad.jacobian(y, x, i=0, j=0)
    u_tt = dde.grad.hessian(y, x, i=0, j=0)
    u_x = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.hessian(y, x, i=0, j=1)
    u_y = dde.grad.jacobian(y, x, i=0, j=2)
    u_yy = dde.grad.hessian(y, x, i=0, j=2)
    u_z = dde.grad.jacobian(y, x, i=0, j=3)
    u_zz = dde.grad.hessian(y, x, i=0, j=3)

    # Placeholder for Lamé parameters and body force
    lambda_ = 1.0
    mu = 1.0
    rho = 1.0
    f = np.zeros_like(u)

    # Navier-Cauchy equations
    navier_cauchy_x = rho * u_tt - (lambda_ + mu) * (u_x + u_y + u_z) - mu * (u_xx + u_yy + u_zz) - f
    navier_cauchy_y = rho * u_tt - (lambda_ + mu) * (u_x + u_y + u_z) - mu * (u_xx + u_yy + u_zz) - f
    navier_cauchy_z = rho * u_tt - (lambda_ + mu) * (u_x + u_y + u_z) - mu * (u_xx + u_yy + u_zz) - f

    return [navier_cauchy_x, navier_cauchy_y, navier_cauchy_z]

# Define the geometry and time domain
geom = dde.geometry.Rectangle([-1, -1], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define boundary and initial conditions
def boundary(_, on_boundary):
    return on_boundary

def initial(_, on_initial):
    return on_initial

# Define the custom loss function with a priori estimates
def custom_loss(y_true, y_pred, x):
    # Calculate the original loss (e.g., MSE)
    original_loss = dde.losses.mse(y_true, y_pred)

    # Calculate the supremum and infimum of the predicted function over the cluster of points
    sup_f = np.max(y_pred)
    inf_f = np.min(y_pred)

    # Define the upper and lower bounds from a priori estimates
    U = 50  # Example upper bound
    L = 10  # Example lower bound

    # Calculate the penalty terms
    penalty_sup = dde.backend.maximum(0, sup_f - U)
    penalty_inf = dde.backend.maximum(0, L - inf_f)

    # Combine the original loss with the penalty terms
    lambda_ = 0.1  # Hyperparameter controlling the strength of the penalty
    total_loss = original_loss + lambda_ * (penalty_sup + penalty_inf)

    return total_loss

# Define the PINN model
data_pde = dde.data.TimePDE(
    geomtime,
    navier_cauchy,
    [],
    [],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    anchors=coordinates
)

net = dde.nn.FNN([4] + [20] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data_pde, net)

# Compile the model with the custom loss function
model.compile("adam", lr=0.001, loss=custom_loss)

# Train the model
losshistory, train_state = model.train(iterations=10000)

# Save the model
dde.saveplot(losshistory, train_state, issave=True, loss_fig_name="loss.png")
