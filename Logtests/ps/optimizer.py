# optimizer.py

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# 1. Define TauDataset at the top level
class TauDataset(Dataset):
    def __init__(self, tau_data):
        self.tau_data = tau_data

    def __len__(self):
        return self.tau_data.shape[0]

    def __getitem__(self, idx):
        return self.tau_data[idx]

def run_optimization(args, tau_i, analytic_moments):
    # Unpack analytic moments
    C0 = analytic_moments['C0']
    C1 = analytic_moments['C1']
    C2 = analytic_moments['C2']
    C3 = analytic_moments['C3']
    C4 = analytic_moments['C4']
    analytics = analytic_moments['analytics']

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Lagrange multipliers as Parameters for optimization
    lambda_0 = torch.nn.Parameter(torch.tensor([1e-5], device=device, dtype=torch.float32))
    lambda_1 = torch.nn.Parameter(torch.tensor([max(args.lam1, 0.0)], device=device, dtype=torch.float32))
    lambda_2 = torch.nn.Parameter(torch.tensor([max(args.lam2, 0.0)], device=device, dtype=torch.float32))
    lambda_3 = torch.nn.Parameter(torch.tensor([max(args.lam3, 0.0)], device=device, dtype=torch.float32))
    lambda_4 = torch.nn.Parameter(torch.tensor([max(args.lam4, 0.0)], device=device, dtype=torch.float32))

    # Define the optimizer
    defrate = 0.001
    optimizer = optim.Adam(
        [lambda_0, lambda_1, lambda_2, lambda_3, lambda_4],
        lr=defrate
    )

    # Convert tau_i to tensor and move to device
    # Replace torch.tensor with clone().detach()
    tau_tensor = tau_i.clone().detach().to(dtype=torch.float32, device=device)

    # Create the Dataset and DataLoader
    dataset = TauDataset(tau_tensor)

    # Decide on batch size
    # Adjust based on your system's memory and performance
    batch_size = len(tau_i)  # Example batch size; tweak as needed

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Start with 0; increase if stable and beneficial
        pin_memory=True if device.type == 'cuda' else False
    )

    # Define the loss function
    def integral_equation_2_direct(lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, tau_batch, wgt_batch):
        # Compute exponent: -lambda_0 - lambda_1 * log_tau - lambda_2 * log_tau**2 - lambda_3 * log_tau**3 - lambda_4 * log_tau**4
        log_tau = torch.log(tau_batch)
        exponent = -lambda_0 - lambda_1 * log_tau - lambda_2 * log_tau**2 - lambda_3 * log_tau**3 - lambda_4 * log_tau**4
        expon = torch.exp(exponent)

        # Compute weighted values
        vals = expon * wgt_batch

        # Correct normalization: sum of modified weights
        total_weight = torch.sum(vals)

        # Compute moments
        #total_weight = torch.sum(wgt_batch)
        data_R_0 = torch.sum(vals) / total_weight
        data_R_1 = torch.sum(vals * log_tau) / total_weight
        data_R_2 = torch.sum(vals * log_tau**2) / total_weight
        data_R_3 = torch.sum(vals * log_tau**3) / total_weight
        data_R_4 = torch.sum(vals * log_tau**4) / total_weight

        # Compute loss based on the selected 'piece'
        if args.piece == 'll':
            loss = (data_R_0 - C0) ** 2
        elif args.piece in ['nllc', 'nll1']:
            loss = (data_R_0 - C0) ** 2 + (data_R_1 - C1) ** 2
        elif args.piece == 'all':
            loss = (
                (data_R_0 - C0) ** 2 +
                (data_R_1 - C1) ** 2 +
                (data_R_2 - C2) ** 2 +
                (data_R_3 - C3) ** 2 +
                (data_R_4 - C4) ** 2 
            )
        else:
            loss = torch.tensor(0.0, device=device, dtype=torch.float32)  # Default to zero if piece is not recognized

        return loss

    # Optimization loop parameters
    num_epochs = 10000  # Adjust as needed
    print_interval = 50  # Print every 10 epochs

    # Optional: Mixed Precision (if using GPU)
    # Uncomment the following lines if you decide to use mixed precision
    # from torch.cuda.amp import autocast, GradScaler
    # scaler = GradScaler()

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        epoch_loss = 0.0
        for batch in dataloader:
            tau_batch = batch[:, 0]
            wgt_batch = batch[:, 1]
            
            optimizer.zero_grad()
            
            # Optional: Mixed Precision
            # with autocast():
            loss = integral_equation_2_direct(lambda_0, lambda_1, lambda_2, lambda_3, lambda_4, tau_batch, wgt_batch)
            
            # Optional: Mixed Precision Backpropagation
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # If not using mixed precision
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered at epoch {epoch}. Skipping this batch.")
                continue

            loss.backward()
            optimizer.step()

            # Enforce parameter constraints if necessary
            with torch.no_grad():
                lambda_1.clamp_(min=0)
                lambda_2.clamp_(min=0)
                lambda_3.clamp_(min=0)
                lambda_4.clamp_(min=0)

            epoch_loss += loss.item()

        # Compute average epoch loss
        avg_loss = epoch_loss / len(dataloader)

        # Print loss and parameters at intervals
        if epoch % print_interval == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f} - Time: {elapsed:.2f}s")
            print(f"Parameters: lambda_0={lambda_0.item():.6f}, lambda_1={lambda_1.item():.6f}, "
                  f"lambda_2={lambda_2.item():.6f}, lambda_3={lambda_3.item():.6f}, lambda_4={lambda_4.item():.6f}")

    # Final output
    print(f"Final Loss: {avg_loss:.6f}")
    print(f"Final Parameters: lambda_0={lambda_0.item():.6f}, lambda_1={lambda_1.item():.6f}, "
          f"lambda_2={lambda_2.item():.6f}, lambda_3={lambda_3.item():.6f}, lambda_4={lambda_4.item():.6f}")

    # Return the learned lambdas
    learned_lambdas = {
        'lambda_0': lambda_0.item(),
        'lambda_1': lambda_1.item(),
        'lambda_2': lambda_2.item(),
        'lambda_3': lambda_3.item(),
        'lambda_4': lambda_4.item()
    }

    return learned_lambdas
