### INITIALIZATION ###

# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt
import random
import sys
import os

# If GPU acceleration is available, not recommended to run without GPU acceleration
gpu=0    
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Parameters
w = h = 5                        # plate size, mm
dx = dy = 0.05                   # discretization in x/y dimension, mm
nx, ny = int(w/dx), int(h/dy)    # number of cells in x/y dimension

dx2, dy2 = dx*dx, dy*dy
dt = 0.001                       # discretization of time

# Parameters
D = 1e-3              # diffusion
epsilon = 0.2
q = 1e-3
f = 1.

### DATA GENERATION ###

# Initial conditions
low, high = 0., 0.1

u0 = low * np.ones((nx, ny))
v0 = low * np.ones((nx, ny))
u = u0.copy()
v = v0.copy()

for i in range(nx):
    for j in range(ny):
         if j > 0.5 * ny and i == j:
            u0[i,j] = high
          
# Number of timesteps
nsteps = 2501
noise_std = 0.0

# Propagate with forward-difference in time, central-difference in space
def do_timestep(u0, v0, u, v):
    du = D * ((u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2 + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2) + 1/epsilon * (u0[1:-1, 1:-1] * (1 - u0[1:-1, 1:-1]) - (u0[1:-1, 1:-1] - q) / (u0[1:-1, 1:-1] + q) * f * v0[1:-1, 1:-1])
    dv = D * ((v0[2:, 1:-1] - 2*v0[1:-1, 1:-1] + v0[:-2, 1:-1])/dx2 + (v0[1:-1, 2:] - 2*v0[1:-1, 1:-1] + v0[1:-1, :-2])/dy2) + u0[1:-1, 1:-1] - v0[1:-1, 1:-1]  
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + dt * du
    v[1:-1, 1:-1] = v0[1:-1, 1:-1] + dt * dv         

    u0 = u.copy()
    v0 = v.copy()
    return u0, v0, u, v, du, dv

print("Generating data.")
u_data = np.zeros((nsteps, nx, ny))
v_data = np.zeros((nsteps, nx, ny))
du_data = np.zeros((nsteps, nx-2, ny-2))
dv_data = np.zeros((nsteps, nx-2, ny-2))
u_data[0] = u
v_data[0] = v

for m in range(nsteps):
    for _ in range(10): # only sample every 10 time steps
      u0, v0, u, v, du, dv = do_timestep(u0, v0, u, v) 

    u_data[m] = u + np.random.normal(0, noise_std, size=(100,100))
    v_data[m] = v + np.random.normal(0, noise_std, size=(100,100))
    du_data[m] = du
    dv_data[m] = dv

print("Data generated.")
# Differentiation
tv_diff = False;    # True: differentiate noisy measurements with tvdiff, False: use calculated derivatives

if tv_diff:
    # Originally, this loop was parallelized. A naive serialized version is used here for cross-platform compatability
    print("Calculating derivatives with TV Diff.")
    for i in range(nx):
        for j in range(ny):
            du_data[:,i,j] = TVRegDiff(u_data[:,i,j],10,0.005,scale='large',dx=dt*10,plotflag=False,diagflag=False)[0:nsteps-500]
            dv_data[:,i,j] = TVRegDiff(v_data[:,i,j],10,0.005,scale='large',dx=dt*10,plotflag=False,diagflag=False)[0:nsteps-500]

### TRAINING ###

# Convert numpy arrays to tensors
u_data_tensor = torch.FloatTensor(u_data).to(device)
v_data_tensor = torch.FloatTensor(v_data).to(device)
uv_data_tensor = torch.stack([u_data_tensor, v_data_tensor], dim=-1)

du_data_tensor = torch.FloatTensor(du_data).to(device)
dv_data_tensor = torch.FloatTensor(dv_data).to(device)
duv_data_tensor = torch.stack([du_data_tensor, dv_data_tensor], dim=-1)

# First-principles model based on incomplete knowledge
class pureODE(nn.Module):
    
    def __init__(self, D):#, epsilon, q):
        super(hybridODE, self).__init__()

        self.D = D

    def forward(self, t, y):
        u = y[...,0]
        v = y[...,1]

        du = self.D * ((u[..., 2:, 1:-1] - 2*u[..., 1:-1, 1:-1] + u[..., :-2, 1:-1])/dx2 + (u[..., 1:-1, 2:] - 2*u[..., 1:-1, 1:-1] + u[..., 1:-1, :-2])/dy2)
        dv = self.D * ((v[..., 2:, 1:-1] - 2*v[..., 1:-1, 1:-1] + v[..., :-2, 1:-1])/dx2 + (v[..., 1:-1, 2:] - 2*v[..., 1:-1, 1:-1] + v[..., 1:-1, :-2])/dy2)

        return torch.stack([du, dv], dim=-1)

# Integrated model based on incomplete knowledge
class hybridODE(nn.Module):
    
    def __init__(self, D):#, epsilon, q):
        super(hybridODE, self).__init__()

        self.D = D #nn.Parameter(D)

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        u = y[...,0]
        v = y[...,1]

        du = self.D * ((u[..., 2:, 1:-1] - 2*u[..., 1:-1, 1:-1] + u[..., :-2, 1:-1])/dx2 + (u[..., 1:-1, 2:] - 2*u[..., 1:-1, 1:-1] + u[..., 1:-1, :-2])/dy2)
        dv = self.D * ((v[..., 2:, 1:-1] - 2*v[..., 1:-1, 1:-1] + v[..., :-2, 1:-1])/dx2 + (v[..., 1:-1, 2:] - 2*v[..., 1:-1, 1:-1] + v[..., 1:-1, :-2])/dy2)

        return torch.stack([du, dv], dim=-1) + self.net(y)[...,1:-1,1:-1,:]

# Training
niters = 5000           # number of iterations
D0 = torch.tensor(1e-2) # initial value of D

model = hybridODE(D0)   # choose model (pureODE or hybridODE)
optimizer = optim.Adam(model.parameters(), lr=1e-1)

print("Starting training.")
for it in range(1, niters + 1):
    optimizer.zero_grad()
    pred = model(0, uv_data_tensor)
    loss = torch.mean((pred - duv_data_tensor)**2)
    loss.backward()
    optimizer.step()

    if (it) % 250 == 0:
        print('Iteration: ', it, '/', niters)

print("Training completed.")

### VISUALISTATION ###

# Helper function
def do_timestep_hybridODE(u0_tensor, v0_tensor, u_tensor, v_tensor):
    # Propagate with forward-difference in time, central-difference in space

    duv = model(0, torch.stack([u0_tensor, v0_tensor], dim=-1))
    du_tensor = duv[...,0]
    dv_tensor = duv[...,1]
    
    u_tensor[1:-1, 1:-1] = u0_tensor[1:-1, 1:-1] + dt * du_tensor
    v_tensor[1:-1, 1:-1] = v0_tensor[1:-1, 1:-1] + dt * dv_tensor

    u0_tensor = u_tensor.clone()
    v0_tensor = v_tensor.clone()
    return u0_tensor, v0_tensor, u_tensor, v_tensor, du_tensor, dv_tensor 

# Solve PDE
with torch.no_grad(): # No gradients needed for visualisation
  u0_tensor = torch.FloatTensor(u0).to(device)
  v0_tensor = torch.FloatTensor(v0).to(device)
  u_tensor = u0_tensor.clone()
  v_tensor = v0_tensor.clone()

  # Number of timesteps
  nsteps = 201

  u_pred_data = torch.zeros((nsteps, nx, ny))
  v_pred_data = torch.zeros((nsteps, nx, ny))
  du_pred_data = torch.zeros((nsteps, nx-2, ny-2))
  dv_pred_data = torch.zeros((nsteps, nx-2, ny-2))
  u_pred_data[0] = u0_tensor
  v_pred_data[0] = v0_tensor

  for m in range(nsteps):
      for _ in range(10): # Sample every 10 timesteps
        u0_tensor, v0_tensor, u_tensor, v_tensor, du_tensor, dv_tensor = do_timestep_hybridODE(u0_tensor, v0_tensor, u_tensor, v_tensor) 

      u_pred_data[m] = u_tensor
      v_pred_data[m] = v_tensor
      du_pred_data[m] = du_tensor
      dv_pred_data[m] = dv_tensor
      if m%100 == 0:
        plt.figure(figsize=(20,20))
        plt.imshow(u_tensor.detach().cpu().numpy())