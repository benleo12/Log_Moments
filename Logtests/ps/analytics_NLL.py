# analytics_NLL.py

import torch

# Set the configuration before importing nll_torch
import config
config.use_torch = 1  # Ensure that nll.py uses torch

import nll as nll_torch
from qcd import AlphaS

def calculate_analytic_moments(args, min_tau, max_tau):
    torch.set_default_dtype(torch.float64)
    # Remove deprecated function
    # torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

    # Initialize alphas and analytics objects
    asif = args.asif
    alphas = [AlphaS(91.1876, asif, 0), AlphaS(91.1876, asif, 1)]
    analytics = nll_torch.NLL(alphas, a=1, b=1, t=91.1876**2, piece=args.piece)

    # Define wLL function
    def wLL(tau):
        partC = (analytics.R_LLp(tau) + analytics.R_NLLp(tau) + analytics.FpF(tau)) / tau
        exponC = torch.exp(-analytics.R_LL(tau) - analytics.R_NLL(tau) - analytics.logF(tau))
        return partC * exponC

    # Define numerical integration function
    def torch_quad(func, a, b, func_mul=None, func_mul2=None, func_mul3=None, func_mul4=None, num_steps=1000000):
        x = torch.logspace(torch.log10(a), torch.log10(b), steps=num_steps, dtype=torch.float64)
        dx = (x[1:] - x[:-1])
        y = (func(x[1:]) + func(x[:-1])) / 2.
        if func_mul is not None:
            y *= (func_mul(x[1:]) + func_mul(x[:-1])) / 2.
        if func_mul2 is not None:
            y *= (func_mul2(x[1:]) + func_mul2(x[:-1])) / 2.
        if func_mul3 is not None:
            y *= (func_mul3(x[1:]) + func_mul3(x[:-1])) / 2.
        if func_mul4 is not None:
            y *= (func_mul4(x[1:]) + func_mul4(x[:-1])) / 2.
        y_dx = y * dx
        integral = torch.sum(y_dx)
        return integral

    # Compute analytic moments
    C0 = torch_quad(wLL, min_tau, max_tau)
    C1 = torch_quad(wLL, min_tau, max_tau, func_mul=torch.log)
    C2 = torch_quad(wLL, min_tau, max_tau, func_mul=torch.log, func_mul2=torch.log)
    C3 = torch_quad(wLL, min_tau, max_tau, func_mul=torch.log, func_mul2=torch.log, func_mul3=torch.log)
    C4 = torch_quad(wLL, min_tau, max_tau, func_mul=torch.log, func_mul2=torch.log, func_mul3=torch.log, func_mul4=torch.log)

    print("Analytic moments:", C0.item(), C1.item(), C2.item(), C3.item(), C4.item())

    analytic_moments = {
        'C0': C0,
        'C1': C1,
        'C2': C2,
        'C3': C3,
        'C4': C4,
        'analytics': analytics
    }

    return analytic_moments

# Ensure that there is no code at the module level that uses 'asif' or 'args'
# All computations should be within the 'calculate_analytic_moments' function
