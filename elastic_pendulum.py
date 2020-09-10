### INITIALIZATION ###

# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import matplotlib.pyplot as plt
import random

# Install torchdiffeq from git
# !pip install git+https://github.com/rtqichen/torchdiffeq.git

# Import odeint with automatic differentiation or adjoint method
adjoint=False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# If GPU acceleration is available
gpu=0    
global device
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Training parameters
niters=2000        # training iterations
data_size=1000      # samples in dataset
batch_time = 16    # steps in batch
batch_size = 128   # samples per batch

### DATA GENERATION ###

# Ideal pendulum model
class ideal_pendulum(nn.Module):
    
    def __init__(self):
        super(ideal_pendulum, self).__init__()
        self.g = 9.81                  # Acceleration due to gravity
        self.l = 3.52                  # Length of the pendulum

    def forward(self, t, y):
        x = y.view(-1,4)[:,0]
        dx = y.view(-1,4)[:,1]
        theta = y.view(-1,4)[:,2]
        dtheta = y.view(-1,4)[:,3]

        ddx = 0. * dx
        ddtheta  = -self.g/self.l * torch.sin(theta)     

        return torch.stack([dx, ddx, dtheta, ddtheta], dim=1).to(device)

# Elastic pendulum model
class elastic_pendulum(nn.Module):
    
    def __init__(self):
        super(elastic_pendulum, self).__init__()
        self.g = 9.81                  # Acceleration due to gravity
        self.l0 = 2.25                 # Length of the Pendulum
        self.k_m = 366                 # Spring Constant / Mass of the pendulum
        self.s0 = 1.125

    def forward(self, t, y):
        x = y.view(-1,4)[:,0]
        dx = y.view(-1,4)[:,1]
        theta = y.view(-1,4)[:,2]
        dtheta = y.view(-1,4)[:,3]

        ddx = (self.l0 + x) * dtheta**2 - self.k_m * (x + self.s0) + self.g * torch.cos(theta)
        ddtheta = - self.g/(self.l0 + x) * torch.sin(theta) - 2 * dx / (self.l0 + x) * dtheta

        return torch.stack([dx, ddx, dtheta, ddtheta], dim=1).to(device)

# Initial condition & time span
true_y0 = torch.tensor([[-.75, 0., 1.25, 0.]]).to(device)
t = torch.linspace(0., 10., data_size).to(device)

# Disable backprop, solve system of ODEs
print("Generating data.")
with torch.no_grad():
    true_y = odeint(elastic_pendulum(), true_y0, t, method='dopri5') # Select here the system generating data (ideal_pendulum, elastic_pendulum)
print("Data generated.")

# Add noise (mean = 0, std = 0.1)
true_y += torch.randn(data_size,1,4)/10.

# Batch function
def get_batch(batch_time, batch_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True)).to(device)
    batch_y0 = true_y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0).to(device)
    return batch_y0, batch_t, batch_y

### MODELS ###

# Purely first-principles model based on incomplete knowledge
class pureODE(nn.Module):
  
    def __init__(self, p0):
        super(pureODE, self).__init__()
        
        self.paramsODE = nn.Parameter(p0)
        self.g = 9.81                  # Acceleration due to gravity
        self.l = self.paramsODE[0]     # Length of the Pendulum
        
    def forward(self, t, y):
        x = y.view(-1,4)[:,0]
        dx = y.view(-1,4)[:,1]
        theta = y.view(-1,4)[:,2]
        dtheta = y.view(-1,4)[:,3]

        ddx = 0. * dx
        ddtheta = -self.g/self.l * torch.sin(theta)       

        return torch.stack([dx, ddx, dtheta, ddtheta], dim=1).view(-1,1,4).to(device)

# Data-driven model
class neuralODE(nn.Module):

    def __init__(self):
        super(neuralODE, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 40),
            nn.Tanh(),
            nn.Linear(40, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.5)
                
    def forward(self, t, y):
        dx = y.view(-1,4)[:,1].view(-1,1,1)
        ddx = self.net(y)[:,0,0].view(-1,1,1)
        dtheta = y.view(-1,4)[:,3].view(-1,1,1)
        ddtheta = self.net(y)[:,0,1].view(-1,1,1)

        return torch.cat((dx, ddx, dtheta, ddtheta), dim=-1).to(device)

# Integrated first-principles/data-driven model based on incomplete knowledge  
class hybridODE(nn.Module):

    def __init__(self, p0):
        super(hybridODE, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 40),
            nn.Tanh(),
            nn.Linear(40, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.5)
                nn.init.constant_(m.bias, val=0)
                
        self.paramsODE = nn.Parameter(p0)
        self.g = 9.81                 # Acceleration due to gravity
        self.l = self.paramsODE[0]    # Length of the Pendulum
        

    def forward(self, t, y):
        x = y.view(-1,4)[:,0]
        dx = y.view(-1,4)[:,1]
        theta = y.view(-1,4)[:,2]
        dtheta = y.view(-1,4)[:,3]

        ddx = self.net(y)[:,0,0]
        ddtheta = -self.g/self.l * torch.sin(theta) + self.net(y)[:,0,1]

        return (torch.stack([dx, ddx, dtheta, ddtheta], dim=1).view(-1,1,4)).to(device)


### TRAINING ###

# Initialization
lr = 1e-2                                       # learning rate
p0 = torch.tensor([2., 360., 1.]).to(device)    # initial conditions parameters
model = hybridODE(p0).to(device)                # choose type of model to train (pureODE(p0), neuralODE(), hybridODE(p0))
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #optional learning rate scheduler


print("Starting training.")
for it in range(1, niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch(batch_time, batch_size)
    pred_y = odeint(model, batch_y0, batch_t, method='rk4')    #It is advised to use a fixed-step solver during training to avoid underflow of dt
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (it) % 250 == 0:
        print('Iteration: ', it, '/', niters)

### VISUALIZATION ###

pred_y = odeint(model, true_y0.view(1,1,4), t, method='rk4').view(-1,1,4)

plt.figure(figsize=(20, 10))
plt.plot(t.detach().cpu().numpy(), pred_y[:,0,2].detach().cpu().numpy())
plt.plot(t.detach().cpu().numpy(), true_y[:,0,2].cpu().numpy(), 'o')
plt.xlabel('t')
plt.ylabel('theta')
plt.show()  

plt.figure(figsize=(20, 10))
plt.plot(pred_y[:,0,2].detach().cpu().numpy(), pred_y[:,0,3].detach().cpu().numpy())
plt.plot(true_y[:,0,2].detach().cpu().numpy(), true_y[:,0,3].cpu().numpy(), 'o')
plt.xlabel('theta')
plt.ylabel('dtheta')
plt.show()