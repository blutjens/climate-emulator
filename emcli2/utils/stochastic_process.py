import numpy as np
from tqdm import tqdm
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

def generate_energy_balance_model(time_steps=250, num_samples=3, dt=1.0, theta=0.1, forcing=None, sigma=0.1):
    """
    time_steps int: number of time steps after warmup; set equal to runtime for dt=1, in yr
    num_samples int: number of samples
    dt float: Length of time increment, in yr
    theta float: Speed of mean reversion
    forcing np.array(time_steps): perturbation radiative forcing over time
    sigma float: Volatility
    warmup_time: time steps for warmup that will be cutoff
    """
    # Parameters for the OU process
    mu = 0.0         # Long-term mean
    X0 = 0.0         # Initial value

    runtime = 250 # time after warmup
    time, dt = np.linspace(0, runtime-1, time_steps, retstep=True)
    warmup_time = 500 # Need warmup for variance to reach equilibrium
    warmup_steps = int(warmup_time / dt)
    warmup = np.linspace(0, warmup_time-1, warmup_steps)
    T_vec = np.concatenate((warmup, time+warmup_time))
    total_steps = time_steps + warmup_steps
    total_time = warmup_time + runtime

    if forcing is None:
        forcing = np.zeros((time_steps))
    forcing = np.concatenate((np.zeros((warmup_time)), forcing))

    # Pre-allocate array for efficiency
    ou_samples = np.zeros((num_samples, total_steps))
    ou_samples[:,0] = X0

    dW = np.sqrt(dt) * np.random.normal(0, 1, size=(num_samples,total_steps))

    # Generate the OU process
    for t in range(0, total_steps - 1):
        ou_samples[:,t+1] = ou_samples[:,t] + forcing[t] * dt + theta * (mu - ou_samples[:,t]) * dt + sigma * dW[:,t]

    # Remove warmup
    ou_samples = ou_samples[:,warmup_steps:]
    T_vec = time

    return ou_samples, T_vec

def generate_ornstein_uhlenbeck_process(time_steps=250, num_samples=3, dt=1.0):
    """
    time_steps int: number of time steps after warmup; set equal to runtime for dt=1.
    num_samples int: number of samples
    """
    # Parameters for the OU process
    theta = 0.1      # Speed of mean reversion
    mu = 0.0         # Long-term mean
    sigma = 0.1      # Volatility
    X0 = 1.0         # Initial value

    runtime = 250 # time after warmup
    time, dt = np.linspace(0, runtime-1, time_steps, retstep=True)
    warmup_time = 500 # time for warmup; will be cutoff
    warmup_steps = int(warmup_time / dt)
    warmup = np.linspace(0, warmup_time-1, warmup_steps)
    T_vec = np.concatenate((warmup, time+warmup_time))
    total_steps = time_steps + warmup_steps
    total_time = warmup_time + runtime

    # Pre-allocate array for efficiency
    ou_samples = np.zeros((num_samples, total_steps))
    ou_samples[:,0] = X0

    dW = np.sqrt(dt) * np.random.normal(0, 1, size=(num_samples,total_steps))

    # Generate the OU process
    for t in range(0, total_steps - 1):
        ou_samples[:,t+1] = ou_samples[:,t] + theta * (mu - ou_samples[:,t]) * dt + sigma * dW[:,t]

    # Remove warmup
    ou_samples = ou_samples[:,warmup_steps:]
    T_vec = time

    return ou_samples, T_vec

def generate_wiener_process(time_steps=250, num_samples=3, dt=1.0):
    """
    Generate multiple samples of a Wiener process (Brownian motion).
    src: claude.ai

    Parameters:
    -----------
    time_steps : int, optional
        Number of time steps in the simulation
    num_samples : int, optional
        Number of independent Wiener process samples to generate
    seed : int, optional
        Random seed for reproducibility 
    dt : float, optional
        Time step size
    Returns:
    --------
    wiener_samples : numpy.ndarray
        Array of Wiener process samples with shape (num_samples, time_steps)
    """
    
    # Generate Wiener process samples
    # Wiener process is the integral of Gaussian white noise
    # Each step is an independent normal distribution with mean 0 and variance dt
    wiener_samples = np.zeros((num_samples, time_steps))
    
    for i in range(num_samples):
        # Generate increments from standard normal distribution
        increments = np.random.normal(0, np.sqrt(dt), time_steps)
        
        # First step is always 0
        wiener_samples[i, 0] = 0
        
        # Cumulative sum of increments creates the Wiener process
        wiener_samples[i, 1:] = np.cumsum(increments[1:])
    
    return wiener_samples

class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series data
    
    Args:
        X (numpy.ndarray): Input time series data
        y (numpy.ndarray, optional): Target values 
    """
    def __init__(self, inputs, targets):
        # Convert to torch tensors
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

        # Flatten across sample and time dimension    
        self.inputs = self.inputs.flatten()
        self.targets = self.targets.flatten()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.ones(1)*self.inputs[idx], torch.ones(1)*self.targets[idx]

class TimeSeriesFCN(nn.Module):
    def __init__(self, input_dim=1, hidden_layers=[64, 32], output_dim=1):
        """
        Fully Connected Neural Network for Time Series Data
        
        Args:
            input_dim (int): Number of time steps in the input
            hidden_layers (list): List of hidden layer sizes
            output_dim (int, optional): Dimension of the output 
        """
        super(TimeSeriesFCN, self).__init__()
                
        # Time step embedding (optional, can be removed or modified)
        self.time_embedding = nn.Linear(input_dim, input_dim)
        
        # Construct hidden layers
        layers = []
        prev_layer_size = input_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_layer_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(0.2))  # Add dropout for regularization
            prev_layer_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(prev_layer_size, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the neural network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps)
        
        Returns:
            torch.Tensor: Output prediction
        """
        # Optional time step embedding
        x_embedded = self.time_embedding(x)
        
        # Pass through the fully connected layers
        return self.model(x_embedded)

def train_time_series_model(X_train, Y_train, 
                            X_val=None, Y_val=None,
                            X_test=None, Y_test=None,
                             batch_size=32, 
                             epochs=100, 
                             learning_rate=0.001,
                             weight_decay=0.0,
                             return_best_val_ckpt=False,
                             verbose=True):
    """
    Train the time series fully connected neural network using DataLoader
    
    Args:
        X_train (numpy.ndarray): Training input data
        Y_train (numpy.ndarray, optional): Training target data
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Optimization learning rate
        weight_decay (float): Optimization weight decay
        return_best_val_ckpt (bool): If true, returns model checkpoint
            that achieved the best validation score    
    Returns:
        tuple: Trained model, training losses, validation losses
    """    
    # Create train dataloader 
    train_set = TimeSeriesDataset(X_train, Y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Create val dataloader
    if X_val is not None and Y_val is not None:
        val_set = TimeSeriesDataset(X_val, Y_val)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Create test dataloader
    if X_test is not None and Y_test is not None:
        test_set = TimeSeriesDataset(X_test, Y_test)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = TimeSeriesFCN(input_dim=1, output_dim=1)
    if return_best_val_ckpt:
        incumbent = copy.deepcopy(model)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    train_losses = []
    val_losses = []
    test_losses = []
    avg_val_loss = None
    avg_test_loss = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Test phase
        model.eval()
        with torch.no_grad():
            # Compute val loss
            if X_val is not None and Y_val is not None:
                epoch_val_loss = 0.0
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    val_loss = criterion(outputs, batch_y)
                    epoch_val_loss += val_loss.item()
                avg_val_loss = epoch_val_loss / len(val_loader)
                if return_best_val_ckpt:
                    # update incumbent
                    if epoch == 0:
                        incumbent = copy.deepcopy(model)
                    elif avg_val_loss < np.min(val_losses):
                        incumbent = copy.deepcopy(model)
                val_losses.append(avg_val_loss)
            # Compute test loss
            if X_test is not None and Y_test is not None:
                epoch_test_loss = 0.0
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x)
                    test_loss = criterion(outputs, batch_y)
                    epoch_test_loss += test_loss.item()
                avg_test_loss = epoch_test_loss / len(test_loader)
                test_losses.append(avg_test_loss)    
        
        # Learning rate scheduling
        # scheduler.step(avg_val_loss)
        
        # Print progress
        if verbose and epoch % 25 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            loss_print = f'Train Loss: {avg_train_loss:.4f} | '
            if avg_val_loss is not None:
                loss_print += f'Val Loss: {avg_val_loss:.4f} | '
            if avg_test_loss is not None:
                loss_print += f'Test Loss: {avg_test_loss:.4f}' 
            print(loss_print)
    
    if return_best_val_ckpt:
        model = incumbent
    return model, train_losses, val_losses, test_losses

def plot_stochastic_process_mse(mse_simple, mse_fcn, num_train_samples_set, filepath_to_save, logaxis=False, label_simple_model='best constant', fill_between=True):
    # Plots
    plt.figure(figsize=(10, 6))
    plt.plot(num_train_samples_set, mse_simple.mean(axis=1), label=label_simple_model, color='tab:orange')
    if fill_between:
        plt.fill_between(num_train_samples_set, mse_simple.mean(axis=1)+mse_simple.std(axis=1), mse_simple.mean(axis=1)-mse_simple.std(axis=1), color='tab:orange', alpha=0.3)
    plt.plot(num_train_samples_set, mse_fcn.mean(axis=1), label='fcn', color='tab:blue')
    if fill_between:
        plt.fill_between(num_train_samples_set, mse_fcn.mean(axis=1)+mse_fcn.std(axis=1), mse_fcn.mean(axis=1)-mse_fcn.std(axis=1), color='tab:blue', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    if logaxis:
        plt.xscale('log')
    plt.xlabel('# realizations in train set')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    if filepath_to_save is not None:
        Path(filepath_to_save + '.png').parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)
    else:
        plt.show()
    plt.close()

def plot_bias_variance(mse_simple, sbias_simple, var_simple, mse_fcn, sbias_fcn, var_fcn, num_train_samples_set=10, add_fcn=False, filepath_to_save=None, logaxis=False, label_simple_model='best simple', fill_between=True):
    plt.figure(figsize=(10, 6))
    params = {'legend.fontsize': 'x-large',
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    plt.plot(num_train_samples_set, mse_simple.mean(axis=1), label=f'MSE {label_simple_model}', color='tab:orange', linestyle='-', linewidth='6', marker='o', markersize=10,alpha=0.6)
    if fill_between:
        plt.fill_between(num_train_samples_set, mse_simple.mean(axis=1)+mse_simple.std(axis=1), mse_simple.mean(axis=1)-mse_simple.std(axis=1), color='tab:orange', alpha=0.2)
    plt.plot(num_train_samples_set, var_simple, label=f'Var {label_simple_model}', color='darkgoldenrod', linestyle='--', marker='*')
    plt.plot(num_train_samples_set, sbias_simple, label=rf'Bias$^2$ {label_simple_model}', color='darkgoldenrod', linestyle='dotted', marker='D',markersize=5,linewidth=2)

    if add_fcn:
        plt.plot(num_train_samples_set, mse_fcn.mean(axis=1), label='MSE neural net', color='tab:blue', linestyle='-', linewidth='6', marker='o', markersize=10,alpha=0.6)
        if fill_between:
            plt.fill_between(num_train_samples_set, mse_fcn.mean(axis=1)+mse_fcn.std(axis=1), mse_fcn.mean(axis=1)-mse_fcn.std(axis=1), color='tab:blue', alpha=0.2)
        plt.plot(num_train_samples_set, var_fcn, label='Var neural net', color='darkblue', linestyle='--', marker='*')
        plt.plot(num_train_samples_set, sbias_fcn, label=r'Bias$^2$ neural net', color='darkblue', linestyle='dotted', marker='D',markersize=5,linewidth=2)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    if logaxis:
        from matplotlib.ticker import ScalarFormatter
        plt.xscale('log')
        xticks = [1,3,5,10,15,20,30,40,50,100]
        # Remove values that are too high
        iter=0
        for _ in np.arange(len(xticks)):
            if xticks[iter] > num_train_samples_set.max():
                xticks.remove(xticks[iter])
            else:
                iter += 1
        plt.xticks(xticks)
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
    plt.xlabel('# realizations in train set')
    plt.ylabel(r'MSE, Bias$^2$, and Variance')
    plt.legend()            
    plt.tight_layout()
    if filepath_to_save is not None:
        Path(filepath_to_save + '.png').parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath_to_save)
    else:
        plt.show()
    plt.close()
    plt.rcParams.clear()
    plt.rcParams.update(plt.rcParamsDefault)

def run_stochastic_process_experiment_spatial_MSE(targets, num_train_samples_set=np.array([1,2,4,7,10,30]),
                                      num_draws=5, 
                                      process_name='ornstein_uhlenbeck',
                                      add_fcn=True,
                                      dir_figures='docs/figures/ornstein_uhlenbeck_process/temp/',
                                      dir_ckpts=None,
                                      use_best_val_ckpt=False,
                                      nonlinearity='exponential'):
    """
    Same as run_stochastic_process_experiment, but computing the equivalent of the spatial MSE by
      averaging over time before computing the MSE.
    
    Args:
        use_best_val_ckpt bool: see train_time_series_model()
        nonlinearity
    """
    time_steps = targets.shape[0]
    t_eval = 21 # Number of years at the end of the vector that should be used to compute the n-yr average scores.

    # Initialize arrays to track statistics
    simple_mean = np.zeros((len(num_train_samples_set),time_steps)) # Mean predictions
    simple_sq = np.zeros((len(num_train_samples_set),time_steps)) # Mean square predictions
    simple_tmean_sq = np.zeros((len(num_train_samples_set))) # Mean squared 21-average predictions
    mse_simple = np.zeros((len(num_train_samples_set), num_draws)) # Mean squared error of each 21-yr avg prediction wrt. the test data
    mse_simple_global = np.zeros((len(num_train_samples_set), num_draws)) # Mean squared error of each predictions wrt. the test data
    rmse_simple = np.zeros((len(num_train_samples_set), num_draws)) # Root mean squared error wrt. the test data
    sbiast_simple = np.zeros((len(num_train_samples_set))) # Average squared bias of 21-yr avg predictions
    sbiast_simple_global = np.zeros((len(num_train_samples_set))) # Average squared bias per time-step
    simple_var = np.zeros((len(num_train_samples_set),time_steps)) # Variance across model predictions, per time step
    var_simple = np.zeros((len(num_train_samples_set))) # Variance of 21-yr averaged model predictions across the ensemble
    var_simple_global = np.zeros((len(num_train_samples_set))) # Variance of model predictions across the ensemble, averaged across time
    
    fcn_mean = np.zeros((len(num_train_samples_set),time_steps))
    fcn_sq = np.zeros((len(num_train_samples_set),time_steps))
    fcn_tmean_sq = np.zeros((len(num_train_samples_set)))
    mse_fcn = np.zeros((len(num_train_samples_set), num_draws))
    mse_fcn_global = np.zeros((len(num_train_samples_set), num_draws))
    rmse_fcn = np.zeros((len(num_train_samples_set), num_draws))
    sbiast_fcn = np.zeros((len(num_train_samples_set)))
    sbiast_fcn_global = np.zeros((len(num_train_samples_set)))
    fcn_var = np.zeros((len(num_train_samples_set),time_steps))
    var_fcn = np.zeros((len(num_train_samples_set)))
    var_fcn_global = np.zeros((len(num_train_samples_set)))

    for m, num_realizations_in_train in tqdm(enumerate(num_train_samples_set)):
        for n in tqdm(np.arange(num_draws)):
            if 'wiener' in process_name or 'ornstein_uhlenbeck' in process_name:
                if process_name == 'wiener':
                    noise_samples_train = generate_wiener_process(time_steps=time_steps,num_samples=num_realizations_in_train,dt=0.001)
                    T_vec = np.arange(time_steps)
                    Y_train = targets + noise_samples_train
                elif process_name == 'ornstein_uhlenbeck':
                    noise_samples_train, T_vec = generate_ornstein_uhlenbeck_process(time_steps=time_steps,num_samples=num_realizations_in_train)
                    Y_train = targets + noise_samples_train
                    if use_best_val_ckpt:
                        samples_val, _ = generate_ornstein_uhlenbeck_process(time_steps=time_steps,num_samples=num_realizations_in_train)
                        Y_val = targets + samples_val
                        Y_val = Y_val.mean(axis=0)[None,:]
                    else:
                        Y_val = None
                # Use time as input feature
                x_vals = T_vec
                X_train = torch.arange(time_steps)[None,:]/float(time_steps)
            elif 'energy_balance' in process_name:
                r_cumul_emissions = 4./5000
                cumul_emissions = 5000 * np.exp(- (np.arange(time_steps,dtype=np.float32)-250.)**2. / (2. * 50.**2))
                pert_radiative_forcing = r_cumul_emissions * cumul_emissions

                climate_feedback = -2. # Local feedback parameter, lambda, in W m^-2 K^-1
                water_density = 997.                # Density of water, rho_w, in kg m^-3
                water_heat_capacity = 4184.         # Specific heat capacity of water, in W s kg^-1 K^-1
                water_depth = 150. # Effective water depth, h(r), in m
                                #  Choosing value for high-latitude ocean instead of land, because
                                #  internal variability over land regresses to the mean so quickly that
                                #  it doesn't introduce multi-decadal oscillations
                heat_capacity = water_density * water_heat_capacity * water_depth # Heat capacity, in W s m^-2 K^-1
                secs_per_yr = 60.*60.*24.*365.25 # Seconds per year in s yr^-1
                theta = -climate_feedback * secs_per_yr / heat_capacity # Speed of mean reversion, in yr^-1
                sigma = 0.1 # strength of year-to-year variability
                sigma_nonlinear = 3. # multiplicative factor of year-to-year variability amplitude of nonlinear var.

                if 'quadratic' in nonlinearity:
                    def nonlinear_fn(x):
                        return 0.03*(4.*x)**2
                else:
                    def nonlinear_fn(x):
                        return 0.04*(np.exp(2*x)-1.)
                
                # Get emission-forced signal without any noise
                targets_linear, T_vec = generate_energy_balance_model(
                    time_steps=time_steps,
                    num_samples=1,
                    forcing=pert_radiative_forcing * secs_per_yr / heat_capacity,
                    theta=theta,
                    sigma=0.)
                targets_linear = targets_linear.squeeze()

                samples_train_linear, _ = generate_energy_balance_model(
                    time_steps=time_steps,
                    num_samples=num_realizations_in_train,
                    forcing=pert_radiative_forcing * secs_per_yr / heat_capacity,
                    theta=theta,
                    sigma=sigma)

                Y_train = nonlinear_fn(targets_linear) + sigma_nonlinear * (samples_train_linear - targets_linear)
                targets = nonlinear_fn(targets_linear)

                if use_best_val_ckpt:
                    samples_val_linear, _ = generate_energy_balance_model(
                        time_steps=time_steps,
                        num_samples=num_realizations_in_train,
                        forcing=pert_radiative_forcing * secs_per_yr / heat_capacity,
                        theta=theta,
                        sigma=sigma)
                    Y_val = nonlinear_fn(targets_linear) + sigma_nonlinear * (samples_val_linear - targets_linear)
                    Y_val = Y_val.mean(axis=0)[None,:]
                else:
                    Y_val = None

                # Use cumulative emissions as input feature
                x_vals = cumul_emissions / np.max(cumul_emissions)
                X_train = torch.from_numpy(cumul_emissions)[None,:] / np.max(cumul_emissions)
            else:
                raise NotImplementedError('process_name variable needs to be defined.')

            Y_train = Y_train.mean(axis=0)[None,:] # average across ensemble dimension

            # Get prediction from the best linear fit
            coef = np.polyfit(x_vals,Y_train.mean(axis=0),1)
            simple_preds = np.poly1d(coef)(x_vals)
            label_simple_model = 'Linear Fit'

            # Compute statistics of simple model predictions
            simple_mean[m,:] += simple_preds
            simple_sq[m,:] += simple_preds**2
            simple_tmean_sq[m] += np.mean(simple_preds[-t_eval:])**2
            # global_MSE: 
            mse_simple_global[m,n] = np.mean((targets - simple_preds)**2)
            # spatial MSE:
            mse_simple[m,n] = (np.mean(targets[-t_eval:]) - np.mean(simple_preds[-t_eval:]))**2
            
            if add_fcn:
                # Compute RMSE of best neural network
                if use_best_val_ckpt:
                    X_val = X_train
                else:
                    X_val = None

                # Train the model
                trained_model, train_losses, val_losses, test_losses = train_time_series_model(X_train=X_train, Y_train=Y_train, 
                                            X_val=X_val, Y_val=Y_val,
                                            X_test=None, Y_test=None,
                                            batch_size=500, 
                                            epochs=150, 
                                            learning_rate=0.005, 
                                            weight_decay=0.00001,
                                            return_best_val_ckpt=use_best_val_ckpt,
                                            verbose=False)
            
                fcn_preds = trained_model(X_train.squeeze()[:,None]).detach().numpy().squeeze()

                # Compute statistics of neural network prediction
                fcn_mean[m,:] += fcn_preds
                fcn_sq[m,:] += fcn_preds**2
                fcn_tmean_sq[m] += np.mean(fcn_preds[-t_eval:])**2
                # global_MSE: 
                mse_fcn_global[m,n] = np.mean((targets - fcn_preds)**2)
                # spatial MSE
                mse_fcn[m,n] = (np.mean(targets[-t_eval:]) - np.mean(fcn_preds[-t_eval:]))**2

        filepath_to_save = dir_figures + '/spatial_MSE/Y_mse'
        plot_stochastic_process_mse(mse_simple, mse_fcn, num_train_samples_set, filepath_to_save=filepath_to_save, label_simple_model=label_simple_model, fill_between=False)
        filepath_to_save = filepath_to_save + '_log'
        plot_stochastic_process_mse(mse_simple, mse_fcn, num_train_samples_set, filepath_to_save=filepath_to_save, logaxis=True, label_simple_model=label_simple_model, fill_between=False)
        filepath_to_save = dir_figures + '/global_MSE/Y_mse'
        plot_stochastic_process_mse(mse_simple_global, mse_fcn_global, num_train_samples_set, filepath_to_save=filepath_to_save, label_simple_model=label_simple_model, fill_between=False)
        filepath_to_save = filepath_to_save + '_log'
        plot_stochastic_process_mse(mse_simple_global, mse_fcn_global, num_train_samples_set, filepath_to_save=filepath_to_save, logaxis=True, label_simple_model=label_simple_model, fill_between=False)
            
        # Compute statistics every batch, to plot intermediate results (even though it sacrifices runtime)
        simple_mean[m,:] = simple_mean[m,:] / float(num_draws)
        simple_sq[m,:] = simple_sq[m,:] / float(num_draws)
        simple_tmean_sq[m] = simple_tmean_sq[m] / float(num_draws)
        # global_var: 
        simple_var[m,:] = (simple_sq[m,:] - simple_mean[m,:]**2) * float(num_draws) / (float(num_draws - 1))
        var_simple_global[m] = simple_var[m,:].mean() # Reduce along time dimension
        # spatial_var:
        var_simple[m] = (simple_tmean_sq[m] - np.mean(simple_mean[m,-t_eval:])**2) * float(num_draws) / (float(num_draws - 1))
        # global_bias: 
        sbiast_simple_global[m] = ((simple_mean[m,:] - targets)**2).mean() # Compute ensemble-mean prediction, then bias, then square, then average across time
        # spatial_bias:
        sbiast_simple[m] = (np.mean(simple_mean[m,-t_eval:]) - np.mean(targets[-t_eval:]))**2 # Compute ensemble-mean 21-yr avg prediction, then bias, then square
        rmse_simple[m,:] = np.sqrt(mse_simple[m,:])

        fcn_mean[m,:] = fcn_mean[m,:] / float(num_draws)
        fcn_sq[m,:] = fcn_sq[m,:] / float(num_draws)
        fcn_tmean_sq[m] = fcn_tmean_sq[m] / float(num_draws)
        # global_var: 
        fcn_var[m,:] = (fcn_sq[m,:] - fcn_mean[m,:]**2) * float(num_draws) / (float(num_draws - 1))
        var_fcn_global[m] = fcn_var[m,:].mean()
        # spatial_var:
        var_fcn[m] = (fcn_tmean_sq[m] - np.mean(fcn_mean[m,-t_eval:])**2) * float(num_draws) / (float(num_draws - 1))
        # global_bias: 
        sbiast_fcn_global[m] = ((fcn_mean[m,:] - targets)**2).mean()
        # spatial_bias
        sbiast_fcn[m] = (np.mean(fcn_mean[m,-t_eval:]) - np.mean(targets[-t_eval:]))**2
        rmse_fcn[m,:] = np.sqrt(mse_fcn[m,:])

        # Plot bias variance
        filepath_to_save = dir_figures + '/spatial_MSE/bias_var'
        plot_bias_variance(mse_simple, sbiast_simple, var_simple, mse_fcn, sbiast_fcn, var_fcn, num_train_samples_set=num_train_samples_set, add_fcn=add_fcn, filepath_to_save=filepath_to_save, label_simple_model=label_simple_model, fill_between=False)
        filepath_to_save = filepath_to_save + '_log'
        plot_bias_variance(mse_simple, sbiast_simple, var_simple, mse_fcn, sbiast_fcn, var_fcn, num_train_samples_set=num_train_samples_set, add_fcn=add_fcn, filepath_to_save=filepath_to_save, logaxis=True, label_simple_model=label_simple_model, fill_between=False)
        # Plot bias variance for global
        filepath_to_save = dir_figures + '/global_MSE/bias_var'
        plot_bias_variance(mse_simple_global, sbiast_simple_global, var_simple_global, mse_fcn_global, sbiast_fcn_global, var_fcn_global, num_train_samples_set=num_train_samples_set, add_fcn=add_fcn, filepath_to_save=filepath_to_save, label_simple_model=label_simple_model, fill_between=False)
        filepath_to_save = filepath_to_save + '_log'
        plot_bias_variance(mse_simple_global, sbiast_simple_global, var_simple_global, mse_fcn_global, sbiast_fcn_global, var_fcn_global, num_train_samples_set=num_train_samples_set, add_fcn=add_fcn, filepath_to_save=filepath_to_save, logaxis=True, label_simple_model=label_simple_model, fill_between=False)

    if dir_ckpts is not None:
        Path(dir_ckpts).mkdir(parents=True, exist_ok=True)
        np.save(dir_ckpts + f'tmp_rmse_{simple_model_name}.npy', rmse_simple)
        np.save(dir_ckpts + f'tmp_mse_{simple_model_name}.npy', mse_simple)
        np.save(dir_ckpts + f'tmp_var_{simple_model_name}.npy', var_simple)

        np.save(dir_ckpts + 'tmp_rmse_fcn.npy', rmse_fcn)
        np.save(dir_ckpts + 'tmp_mse_fcn.npy', mse_fcn)
        np.save(dir_ckpts + 'tmp_var_fcn.npy', var_fcn)

        #rmse_simple = np.load(dir_ckpts + 'tmp_rmse_simple.npy')
        #rmse_fcn = np.load(dir_ckpts + 'tmp_rmse_fcn.npy')

def run_stochastic_process_experiment(targets, num_train_samples_set=np.array([1,2,4,7,10,30]),
                                      num_draws=5, 
                                      process_name='ornstein_uhlenbeck',
                                      simple_model_name='linear',
                                      add_fcn=True,
                                      dir_figures='docs/figures/ornstein_uhlenbeck_process/temp/',
                                      dir_ckpts=None,
                                      ):
    """
    Args:
        targets np.array(time_steps)
        num_train_samples_set np.array(num_sets): contains the number of realizations in 
            the training set for which the experiment should be conducted. For each value in the 
            array there will be num_draws random draws of n_realizations.
        num_draws int: Number of draws per value in 'num_train_samples_set'
        process_name str: Name of stochastic process that should be used for sampling noise
        simple_model_name str: Name of simple model that should be used for comparison with the neural net
        add_fcn bool: If True, fits neural net to each draw of n realizations. Set to False for debugging.
        dir_figures str: Relative or absolute path to the directory for figures from this fn
        dir_ckpts str: Relative or absolute path to the directory for saving the data to recreate 
            the plots. E.g., 'runs/ornstein_uhlenbeck_process/temp/'. If None, data will not be saved.
    """
    time_steps = targets.shape[0]

    # Initialize arrays to track statistics
    simple_mean = np.zeros((len(num_train_samples_set),time_steps)) # Mean predictions
    simple_sq = np.zeros((len(num_train_samples_set),time_steps)) # Mean square predictions
    mse_simple = np.zeros((len(num_train_samples_set), num_draws)) # Mean squared error wrt. the test data
    rmse_simple = np.zeros((len(num_train_samples_set), num_draws)) # Root mean squared error wrt. the test data
    sbiast_simple = np.zeros((len(num_train_samples_set))) # Average squared bias per time-step
    simple_var = np.zeros((len(num_train_samples_set),time_steps)) # Variance across model predictions, per time step
    var_simple = np.zeros((len(num_train_samples_set))) # Variance across the model predictions, averaged across time
    
    fcn_mean = np.zeros((len(num_train_samples_set),time_steps))
    fcn_sq = np.zeros((len(num_train_samples_set),time_steps))
    mse_fcn = np.zeros((len(num_train_samples_set), num_draws))
    rmse_fcn = np.zeros((len(num_train_samples_set), num_draws))
    sbiast_fcn = np.zeros((len(num_train_samples_set)))
    fcn_var = np.zeros((len(num_train_samples_set),time_steps))
    var_fcn = np.zeros((len(num_train_samples_set)))

    for m, num_realizations_in_train in tqdm(enumerate(num_train_samples_set)):
        for n in tqdm(np.arange(num_draws)):
            if process_name == 'wiener':
                samples_train = generate_wiener_process(time_steps=time_steps,num_samples=num_realizations_in_train,dt=0.001)
                T_vec = np.arange(time_steps)
            elif process_name == 'ornstein_uhlenbeck':
                samples_train, T_vec = generate_ornstein_uhlenbeck_process(time_steps=time_steps,num_samples=num_realizations_in_train)
            else:
                raise NotImplementedError('process_name variable needs to be defined.')

            Y_train = targets + samples_train
            Y_train = Y_train.mean(axis=0)[None,:] # average across ensemble dimension

            # Get prediction from the simple model
            if simple_model_name == 'constant':
                # Predict the best constant
                simple_preds = Y_train.mean() * np.ones(time_steps)
                label_simple_model = 'Best Constant'
            elif simple_model_name == 'linear':
                # Predict the best linear fit
                coef = np.polyfit(T_vec,Y_train.mean(axis=0),1)
                simple_preds = np.poly1d(coef)(T_vec)
                label_simple_model = 'Linear Fit'
            else:
                raise NotImplementedError(f'simple_model_name is set to {simple_model_name} which is invalid.')

            # Compute statistics of simple model predictions
            simple_mean[m,:] += simple_preds
            simple_sq[m,:] += simple_preds**2
            mse_simple[m,n] = np.mean((targets - simple_preds)**2)

            if add_fcn:
                # Compute RMSE of best neural network
                X_train = torch.repeat_interleave(torch.arange(time_steps)[None,:]/float(time_steps), repeats=Y_train.shape[0], dim=0)

                # Train the model
                trained_model, train_losses, val_losses, test_losses = train_time_series_model(X_train=X_train, Y_train=Y_train, 
                                            X_test=None, Y_test=None,
                                            batch_size=500, 
                                            epochs=150, 
                                            learning_rate=0.005,
                                            weight_decay=0.00001,
                                            return_best_val_ckpt=True,
                                            verbose=False)
            
                fcn_preds = trained_model(X_train.squeeze()[:,None]).detach().numpy().squeeze()

                # Compute statistcs of neural network prediction
                fcn_mean[m,:] += fcn_preds
                fcn_sq[m,:] += fcn_preds**2
                mse_fcn[m,n] = np.mean((targets - fcn_preds)**2)


        filepath_to_save = dir_figures + '/Y_mse'
        plot_stochastic_process_mse(mse_simple, mse_fcn, num_train_samples_set, filepath_to_save=filepath_to_save, label_simple_model=label_simple_model)
        filepath_to_save = filepath_to_save + '_log'
        plot_stochastic_process_mse(mse_simple, mse_fcn, num_train_samples_set, filepath_to_save=filepath_to_save, logaxis=True, label_simple_model=label_simple_model)
            
        # Compute statistics every batch, to plot intermediate results (even though it sacrifices runtime)
        simple_mean[m,:] = simple_mean[m,:] / float(num_draws)
        simple_sq[m,:] = simple_sq[m,:] / float(num_draws)
        simple_var[m,:] = (simple_sq[m,:] - simple_mean[m,:]**2) * float(num_draws) / (float(num_draws - 1))
        sbiast_simple[m] = ((simple_mean[m,:] - targets)**2).mean() # Compute average prediction, then bias, then square, then average across time
        var_simple[m] = simple_var[m,:].mean() # Reduce along time dimension
        rmse_simple[m,:] = np.sqrt(mse_simple[m,:])

        fcn_mean[m,:] = fcn_mean[m,:] / float(num_draws)
        fcn_sq[m,:] = fcn_sq[m,:] / float(num_draws)
        fcn_var[m,:] = (fcn_sq[m,:] - fcn_mean[m,:]**2) * float(num_draws) / (float(num_draws - 1))
        sbiast_fcn[m] = ((fcn_mean[m,:] - targets)**2).mean()
        var_fcn[m] = fcn_var[m,:].mean()
        rmse_fcn[m,:] = np.sqrt(mse_fcn[m,:])

        # Plot bias variance
        filepath_to_save = dir_figures + '/bias_var'
        plot_bias_variance(mse_simple, sbiast_simple, var_simple, mse_fcn, sbiast_fcn, var_fcn, num_train_samples_set=num_train_samples_set, add_fcn=add_fcn, filepath_to_save=filepath_to_save, label_simple_model=label_simple_model)
        filepath_to_save = filepath_to_save + '_log'
        plot_bias_variance(mse_simple, sbiast_simple, var_simple, mse_fcn, sbiast_fcn, var_fcn, num_train_samples_set=num_train_samples_set, add_fcn=add_fcn, filepath_to_save=filepath_to_save, logaxis=True, label_simple_model=label_simple_model)

    if dir_ckpts is not None:
        Path(dir_ckpts).mkdir(parents=True, exist_ok=True)
        np.save(dir_ckpts + f'tmp_rmse_{simple_model_name}.npy', rmse_simple)
        np.save(dir_ckpts + f'tmp_mse_{simple_model_name}.npy', mse_simple)
        np.save(dir_ckpts + f'tmp_var_{simple_model_name}.npy', var_simple)

        np.save(dir_ckpts + 'tmp_rmse_fcn.npy', rmse_fcn)
        np.save(dir_ckpts + 'tmp_mse_fcn.npy', mse_fcn)
        np.save(dir_ckpts + 'tmp_var_fcn.npy', var_fcn)

        #rmse_simple = np.load(dir_ckpts + 'tmp_rmse_simple.npy')
        #rmse_fcn = np.load(dir_ckpts + 'tmp_rmse_fcn.npy')

def get_args():
    parser = argparse.ArgumentParser(description='Plotting routines')
    parser.add_argument('--repo_root', type=str, default='', help='Root path to repository')
    parser.add_argument('--process_name', type=str, default='ornstein_uhlenbeck', choices=['ornstein_uhlenbeck', 'wiener', 'energy_balance_nonlinear'],
                        help='Name of the stochastic process')
    parser.add_argument('--simple_model_name', type=str, default='linear', choices=['linear', 'constant'],
                        help='Name of the simple model')
    parser.add_argument('--cos_amplitude', type=float, default=0.1, help='amplitude of cosine. Only used if process_name==ornstein-uhlenbeck')
    parser.add_argument('--nonlinearity', type=str, default='exponential', help='fn of nonlinearity. Only used if energy_balance in process_name.')
    parser.add_argument('--plot_only',  action='store_true', default=False, help='If true, create plots from file')
    parser.add_argument('--debug',  action='store_true', default=False, help='If true, set small numbers to run code quickly')
    return parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    # Set random seed for reproducibility
    np.random.seed(42)

    time_steps = 250
    simple_model_name = args.simple_model_name
    process_name = args.process_name
    add_fcn = True
    logging.info(f'process name: {process_name}')

    if 'ornstein_uhlenbeck' in process_name:
        logging.info(f'cosine amplitude: {args.cos_amplitude}')
        cosine = args.cos_amplitude*np.cos(np.arange(time_steps)/float(time_steps)*2.*np.pi)
        linear = 1./(2.*time_steps) * np.arange(time_steps)
        targets = cosine + linear
    elif 'energy_balance' in process_name:
        # Pass dummy targets as they'll be constructed within the function
        targets = np.zeros(time_steps)
    
    num_train_samples_set = np.array([1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,50])
    num_draws = 2000 # number of draws to estimate the population statistics
    if args.debug:
        num_train_samples_set = np.array([50]) # np.array([1,2,3,4,5,10,20,30,50])
        num_draws = 2   
    
    #dir_figures = args.repo_root + f'docs/figures/{process_name}_process/cos_linear_w_ou_noise_reg/spatial_MSE_{num_draws}draws/cos_{args.cos_amplitude:.4f}/'.replace('.','-')
    #dir_ckpts = args.repo_root + f'runs/{process_name}_process/spatial_MSE_{num_draws}/'
    dir_figures = args.repo_root + f'docs/figures/{process_name}_{args.nonlinearity}/{num_draws}draws/'.replace('.','-')
    dir_ckpts = args.repo_root + f'runs/{process_name}_{args.nonlinearity}/{num_draws}draws/'
    run_stochastic_process_experiment_spatial_MSE(targets=targets,
                                    num_train_samples_set=num_train_samples_set,
                                    process_name=process_name,
                                    num_draws=num_draws, 
                                    dir_figures=dir_figures, 
                                    dir_ckpts=dir_ckpts,
                                    nonlinearity=args.nonlinearity,
                                    use_best_val_ckpt=True)