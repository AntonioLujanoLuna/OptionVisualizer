"""
pricing_models.py

Contains all pricing logic for European options:
1. Black-Scholes (calls & puts) + Greeks
2. Binomial (Cox-Ross-Rubinstein)
3. Monte Carlo (basic GBM simulation)
"""

import math
import numpy as np
from scipy.stats import norm

# --------------------------------------------------
# Black-Scholes
# --------------------------------------------------
def black_scholes_call_price(S, K, r, sigma, T):
    """
    Computes the Black-Scholes call option price.
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S, K, r, sigma, T):
    """
    Computes the Black-Scholes put option price.
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_greeks_call(S, K, r, sigma, T):
    """
    Returns a dict of Greeks (Delta, Gamma, Vega, Theta, Rho) for a call option.
    """
    if T <= 0:
        return {"Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}

    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * norm.pdf(d1) * math.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm.cdf(d2))
    rho   = K * T * math.exp(-r * T) * norm.cdf(d2)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega":  vega,
        "Theta": theta,
        "Rho":   rho
    }

def black_scholes_greeks_put(S, K, r, sigma, T):
    """
    Returns a dict of Greeks (Delta, Gamma, Vega, Theta, Rho) for a put option.
    """
    # For a put, we can either compute directly or use put-call parity to adjust the sign of Delta, etc.
    # Here, we do it directly for clarity.
    if T <= 0:
        return {"Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}

    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    delta = norm.cdf(d1) - 1.0  # Another expression is norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * norm.pdf(d1) * math.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
             + r * K * math.exp(-r * T) * norm.cdf(-d2))
    rho   = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega":  vega,
        "Theta": theta,
        "Rho":   rho
    }

# --------------------------------------------------
# Binomial Model (Cox-Ross-Rubinstein)
# --------------------------------------------------
def binomial_call_price(S, K, r, sigma, T, steps=100):
    """
    European call option via a simple Cox-Ross-Rubinstein binomial tree.
    """
    if T <= 0:
        return max(S - K, 0.0)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Terminal payoffs
    call_vals = []
    for i in range(steps+1):
        ST = S * (u**(steps - i)) * (d**i)
        call_vals.append(max(ST - K, 0.0))

    # Step back
    for _ in range(steps):
        new_vals = []
        for i in range(len(call_vals) - 1):
            new_vals.append(math.exp(-r*dt)*(p*call_vals[i] + (1.0 - p)*call_vals[i+1]))
        call_vals = new_vals
    return call_vals[0]

def binomial_put_price(S, K, r, sigma, T, steps=100):
    """
    European put option via Cox-Ross-Rubinstein binomial tree.
    """
    if T <= 0:
        return max(K - S, 0.0)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)

    put_vals = []
    for i in range(steps+1):
        ST = S * (u**(steps - i)) * (d**i)
        put_vals.append(max(K - ST, 0.0))

    for _ in range(steps):
        new_vals = []
        for i in range(len(put_vals) - 1):
            new_vals.append(math.exp(-r*dt)*(p*put_vals[i] + (1.0 - p)*put_vals[i+1]))
        put_vals = new_vals
    return put_vals[0]

# --------------------------------------------------
# Monte Carlo Model
# --------------------------------------------------
def monte_carlo_call_price(S, K, r, sigma, T, sims=10000):
    """
    Basic Monte Carlo pricing of a European call under GBM assumptions.
    """
    if T <= 0:
        return max(S - K, 0.0)
    # Terminal price S(T) = S * exp( (r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z )
    Z = np.random.normal(0, 1, sims)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0.0)
    return math.exp(-r*T) * np.mean(payoff)

def monte_carlo_put_price(S, K, r, sigma, T, sims=10000):
    """
    Basic Monte Carlo pricing of a European put under GBM assumptions.
    """
    if T <= 0:
        return max(K - S, 0.0)
    Z = np.random.normal(0, 1, sims)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(K - ST, 0.0)
    return math.exp(-r*T) * np.mean(payoff)

# --------------------------------------------------
# Unified Option Pricing Function
# --------------------------------------------------
def option_price(S, K, r, sigma, T, opt_type="call", model="Black-Scholes",
                 steps=100, sims=10000):
    """
    Generic interface that calls the appropriate function based on `model` and `opt_type`.
    - steps: used by Binomial
    - sims : used by Monte Carlo
    """
    opt_type = opt_type.lower().strip()
    model = model.lower().strip()

    # Validate
    if opt_type not in ["call", "put"]:
        raise ValueError("opt_type must be 'call' or 'put'.")
    if model not in ["black-scholes", "binomial", "monte carlo"]:
        raise ValueError("model must be 'Black-Scholes', 'Binomial', or 'Monte Carlo'.")

    # Black-Scholes
    if model == "black-scholes":
        if opt_type == "call":
            return black_scholes_call_price(S, K, r, sigma, T)
        else:
            return black_scholes_put_price(S, K, r, sigma, T)

    # Binomial
    elif model == "binomial":
        if opt_type == "call":
            return binomial_call_price(S, K, r, sigma, T, steps)
        else:
            return binomial_put_price(S, K, r, sigma, T, steps)

    # Monte Carlo
    else:  # "monte carlo"
        if opt_type == "call":
            return monte_carlo_call_price(S, K, r, sigma, T, sims)
        else:
            return monte_carlo_put_price(S, K, r, sigma, T, sims)
