# data_generator.py

import os
import subprocess
import torch
import csv

def generate_data(args):
    import config
    config.use_torch = 1
    config.Kfac = args.K
    config.Blfac = args.B
    config.Ffac = args.F
    config.seed = args.s

    n_samp = args.e
    asif = args.asif

    # Function to run the dire.py script and generate data
    def run_pypy_script(pypy_script_path, asif, n_samp, piece):
        flags = [
            '-e', str(n_samp),
            '-a', str(asif),
            '-b', '1',
            '-C', '0.01',
            '-x', str(piece),
            '-K', str(config.Kfac),
            '-B', str(config.Blfac),
            '-n', '10',
            '-s', str(args.s),
            '-F', str(config.Ffac)
        ]

        # Filenames based on parameters
        filename = [
            f"thrust_e{n_samp}_A{asif}_{args.piece}_K{args.K}_B{args.B}_seed{args.s}.csv",
            f"weight_e{n_samp}_A{asif}_{args.piece}_K{args.K}_B{args.B}_seed{args.s}.csv"
        ]

        # Check if the file already exists
        if os.path.exists(filename[0]):
            print(f"File {filename} already exists. Not overwriting.")
            return filename

        try:
            # Run the dire.py script with specified flags
            subprocess.check_call(['pypy', pypy_script_path] + flags)

            # Rename output files
            if os.path.exists("thrust_values.csv"):
                os.rename("thrust_values.csv", filename[0])
                os.rename("weight_values.csv", filename[1])
            else:
                print("Expected output file 'thrust_values.csv' not found.")
                return None

        except subprocess.CalledProcessError as e:
            print("An error occurred while running the PyPy script:", e)
            return None

        return filename

    # Function to read CSV data into a PyTorch tensor
    def read_csv_to_torch(csv_file_paths):
        with open(csv_file_paths[0], 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            tau_values = [float(row[0]) for row in csv_reader]
        with open(csv_file_paths[1], 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            wgt_values = [float(row[0]) for row in csv_reader]
        return torch.stack([torch.tensor(tau_values), torch.tensor(wgt_values)], dim=1)

    # Generate data using the dire.py script
    pypy_script_path = os.path.expanduser('dire.py')
    tau_i_files = run_pypy_script(pypy_script_path, asif, n_samp, args.piece)
    if tau_i_files is None:
        print("Data generation failed.")
        sys.exit(1)

    # Read the generated data
    tau_i = read_csv_to_torch(tau_i_files)

    # Process data to get min_tau and max_tau
    min_tau = tau_i[:, 0].min()
    max_tau = tau_i[:, 0].max()

    min_fudge = 1
    min_tau = torch.tensor(max(args.min, min_tau.item())) * min_fudge
    max_tau = torch.tensor(min(args.max, max_tau.item()))

    print("Tau min/max:", min_tau.item(), max_tau.item())

    # Filter data within the specified range
    tau_i = tau_i[tau_i[:, 0] > min_tau]
    tau_i = tau_i[tau_i[:, 0] < max_tau]

    print("max_tau", torch.max(tau_i[:, 0]))
    print("Tau min/max:", min_tau.item(), max_tau.item())

    return tau_i, min_tau, max_tau
