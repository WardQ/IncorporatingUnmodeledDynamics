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
niters=1000        # training iterations
data_size=1000      # samples in dataset
batch_time = 16    # steps in batch
batch_size = 256   # samples per batch

### DATA GENERATION ###

class glycolysis(nn.Module):
    
    def __init__(self):
        super(glycolysis, self).__init__()
        self.J0 = 2.5    #mM min-1
        self.k1 = 100.   #mM-1 min-1
        self.k2 = 6.     #mM min-1
        self.k3 = 16.    #mM min-1
        self.k4 = 100.   #mM min-1
        self.k5 = 1.28   #mM min-1
        self.k6 = 12.    #mM min-1
        self.k = 1.8     #min-1
        self.kappa = 13. #min-1
        self.q = 4.
        self.K1 = 0.52   #mM
        self.psi = 0.1
        self.N = 1.      #mM
        self.A = 4.      #mM

    def forward(self, t, y):
        S1 = y.view(-1,7)[:,0]
        S2 = y.view(-1,7)[:,1]
        S3 = y.view(-1,7)[:,2]
        S4 = y.view(-1,7)[:,3]
        S5 = y.view(-1,7)[:,4]
        S6 = y.view(-1,7)[:,5]
        S7 = y.view(-1,7)[:,6]

        dS1 = self.J0 - (self.k1*S1*S6)/(1 + (S6/self.K1)**self.q)
        dS2 = 2. * (self.k1*S1*S6)/(1 + (S6/self.K1)**self.q) - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * (self.A - S6)
        dS4 = self.k3 * S3 * (self.A - S6) - self.k4 * S4 * S5 - self.kappa *(S4 - S7)
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        dS6 = -2. * (self.k1 * S1 * S6) / (1 + (S6 / self.K1)**self.q) + 2. * self.k3 * S3 * (self.A - S6) - self.k5 * S6
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        return torch.stack([dS1, dS2, dS3, dS4, dS5, dS6, dS7], dim=1).to(device)

# Initial condition, time span & parameters
true_y0 = torch.tensor([[1.6, 1.5, 0.2, 0.35, 0.3, 2.67, 0.1]]).to(device)
t = torch.linspace(0., 4., data_size).to(device)
p = torch.tensor([2.5, 100., 6., 16., 100., 1.28, 12., 1.8, 13., 4., 0.52, 0.1, 1., 4.]).to(device)


# Disable backprop, solve system of ODEs
print("Generating data.")
with torch.no_grad():
    true_y = odeint(glycolysis(), true_y0, t, method='dopri5')
print("Data generated.")

# Add noise (mean = 0, std = 0.1)
true_y *= (1 + torch.randn(data_size,1,7)/20.)

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
        self.J0 = self.paramsODE[0]     #mM min-1
        self.k2 = self.paramsODE[2]     #mM min-1
        self.k3 = self.paramsODE[3]     #mM min-1
        self.k4 = self.paramsODE[4]     #mM min-1
        self.k5 = self.paramsODE[5]     #mM min-1
        self.k6 = self.paramsODE[6]     #mM min-1
        self.k = self.paramsODE[7]      #min-1
        self.kappa = self.paramsODE[8]  #min-1
        self.psi = self.paramsODE[11]
        self.N = self.paramsODE[12]     #mM
        self.A = self.paramsODE[13]     #mM

    def forward(self, t, y):
        S1 = y.view(-1,7)[:,0]
        S2 = y.view(-1,7)[:,1]
        S3 = y.view(-1,7)[:,2]
        S4 = y.view(-1,7)[:,3]
        S5 = y.view(-1,7)[:,4]
        S6 = y.view(-1,7)[:,5]
        S7 = y.view(-1,7)[:,6]

        dS1 = self.J0
        dS2 = - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * (self.A - S6)
        dS4 = self.k3 * S3 * (self.A - S6) - self.k4 * S4 * S5 - self.kappa *(S4 - S7)
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        dS6 = 2. * self.k3 * S3 * (self.A - S6) - self.k5 * S6
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        return torch.stack([dS1, dS2, dS3, dS4, dS5, dS6, dS7], dim=1).view(-1,1,7).to(device)

# Data-driven model
class neuralODE(nn.Module):

    def __init__(self):
        super(neuralODE, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(7, 50),
            nn.Tanh(),
            nn.Linear(50, 20),
            nn.Tanh(),
            nn.Linear(20, 7),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.5)
                
    def forward(self, t, y):

        return self.net(y)

# Integrated first-principles/data-driven model      
class hybridODE(nn.Module):
    
    def __init__(self, p0):
        super(hybridODE, self).__init__()

        self.paramsODE = p0#nn.Parameter(p0)
        self.J0 = self.paramsODE[0]     #mM min-1
        #self.k1 = self.paramsODE[1]     #mM-1 min-1
        self.k2 = self.paramsODE[2]     #mM min-1
        self.k3 = self.paramsODE[3]     #mM min-1
        self.k4 = self.paramsODE[4]     #mM min-1
        self.k5 = self.paramsODE[5]     #mM min-1
        self.k6 = self.paramsODE[6]     #mM min-1
        self.k = self.paramsODE[7]      #min-1
        self.kappa = self.paramsODE[8]  #min-1
        #self.q = self.paramsODE[9]
        #self.K1 = self.paramsODE[10]    #mM
        self.psi = self.paramsODE[11]
        self.N = self.paramsODE[12]     #mM
        self.A = self.paramsODE[13]     #mM

        self.net = nn.Sequential(
            nn.Linear(7, 50),
            nn.Tanh(),
            nn.Linear(50, 20),
            nn.Tanh(),
            nn.Linear(20, 7),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        S1 = y.view(-1,7)[:,0]
        S2 = y.view(-1,7)[:,1]
        S3 = y.view(-1,7)[:,2]
        S4 = y.view(-1,7)[:,3]
        S5 = y.view(-1,7)[:,4]
        S6 = y.view(-1,7)[:,5]
        S7 = y.view(-1,7)[:,6]

        dS1 = self.J0 + 0. * S1 #for dimensions
        dS2 = - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * (self.A - S6)
        dS4 = self.k3 * S3 * (self.A - S6) - self.k4 * S4 * S5 - self.kappa *(S4 - S7)
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        dS6 = 2. * self.k3 * S3 * (self.A - S6) - self.k5 * S6
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        return (torch.stack([dS1, dS2, dS3, dS4, dS5, dS6, dS7], dim=1).view(-1,1,7) + self.net(y)).to(device)


### TRAINING ###

# Initialization
lr = 1e-2                                       # learning rate
#p0 = torch.tensor([2., 120., 5., 14., 90., 1., 15., 2., 15., 4., 0.4, 0.2, 2., 3.]).to(device)
model = hybridODE(p).to(device)                # choose type of model to train (pureODE(p0), neuralODE(), hybridODE(p0))
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1) #optional learning rate scheduler


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
plt.show()  

plt.figure(figsize=(20, 10))
plt.plot(pred_y[:,0,2].detach().cpu().numpy(), pred_y[:,0,3].detach().cpu().numpy())
plt.plot(true_y[:,0,2].detach().cpu().numpy(), true_y[:,0,3].cpu().numpy(), 'o')
plt.show()