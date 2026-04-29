"""
Section 5.2 — Implémentation du routeur optimal (Réseau AMM)
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import time
import os
import warnings

warnings.filterwarnings("ignore")

TOKENS = ["WETH", "USDC", "DAI", "WBTC"]
TOKEN_IDX = {token: i for i, token in enumerate(TOKENS)}

POOLS = {
    "ETH/USDC": (TOKEN_IDX["WETH"], TOKEN_IDX["USDC"]),
    "ETH/DAI":  (TOKEN_IDX["WETH"], TOKEN_IDX["DAI"]),
    "USDC/DAI": (TOKEN_IDX["USDC"], TOKEN_IDX["DAI"]),
    "ETH/WBTC": (TOKEN_IDX["WETH"], TOKEN_IDX["WBTC"]),
}

GAMMA = 0.997  

def load_market_state(data_dir="data", row_idx=0):
    state = {}
    for p_name in POOLS.keys():
        file_path = os.path.join(data_dir, f"{p_name.replace('/', '_')}_hf.csv")
        df = pd.read_csv(file_path)
        state[p_name] = {
            "R0": float(df.iloc[row_idx]["reserve0"]),
            "R1": float(df.iloc[row_idx]["reserve1"])
        }
    return state

def compute_direct_swap(market_state, amount_in):
    R0 = market_state["ETH/DAI"]["R0"]
    R1 = market_state["ETH/DAI"]["R1"]
    return R1 * (GAMMA * amount_in) / (R0 + GAMMA * amount_in)

def route_optimal_swap(market_state, amount_in=50.0):
    n_tokens = len(TOKENS)
    deltas = {p: cp.Variable(2, nonneg=True) for p in POOLS.keys()}
    lambdas = {p: cp.Variable(2, nonneg=True) for p in POOLS.keys()}
    
    constraints = []
    
    for p_name, (idx0, idx1) in POOLS.items():
        R0 = market_state[p_name]["R0"]
        R1 = market_state[p_name]["R1"]
        D = deltas[p_name]
        L = lambdas[p_name]
        
        post_R0 = R0 + GAMMA * D[0] - L[0]
        post_R1 = R1 + GAMMA * D[1] - L[1]
        
        constraints += [post_R0 >= 0, post_R1 >= 0]
        constraints += [L[0] <= R0, L[1] <= R1]
        
        # Invariant Normalisé
        constraints += [cp.geo_mean(cp.hstack([post_R0 / R0, post_R1 / R1])) >= 1.0]

    psi = cp.Variable(n_tokens)
    psi_expr = [0] * n_tokens
    
    for p_name, (idx0, idx1) in POOLS.items():
        psi_expr[idx0] += lambdas[p_name][0] - deltas[p_name][0]
        psi_expr[idx1] += lambdas[p_name][1] - deltas[p_name][1]
        
    for i in range(n_tokens):
        constraints += [psi[i] == psi_expr[i]]
        
    idx_in = TOKEN_IDX["WETH"]
    idx_out = TOKEN_IDX["DAI"]
    
    for i in range(n_tokens):
        if i == idx_in:
            constraints += [psi[i] >= -amount_in]
        elif i == idx_out:
            pass
        else:
            constraints += [psi[i] >= 0]

    objective = cp.Maximize(psi[idx_out])
    prob = cp.Problem(objective, constraints)
    
    start_time = time.perf_counter()
    prob.solve(solver=cp.CLARABEL, verbose=False)
    exec_time_ms = (time.perf_counter() - start_time) * 1000
    
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return psi[idx_out].value, exec_time_ms
    return None, exec_time_ms

def run_backtest():
    print("=== Démarrage du Backtest (Section 5.2) ===")
    TRADE_SIZE = 50.0 
    results = []
    
    for i in range(100):
        try:
            state = load_market_state(row_idx=i)
        except Exception:
            break
            
        direct_out = compute_direct_swap(state, TRADE_SIZE)
        opt_out, latency = route_optimal_swap(state, TRADE_SIZE)
        
        if opt_out is not None:
            improvement_bps = ((opt_out - direct_out) / direct_out) * 10000 
            results.append({
                "block": i,
                "direct_out": direct_out,
                "optimal_out": opt_out,
                "improvement_bps": improvement_bps,
                "latency_ms": latency
            })
            
    if not results:
        print("Échec du backtest.")
        return
        
    df = pd.DataFrame(results)
    print(f"\nRésultats sur {len(df)} transactions (Taille : {TRADE_SIZE} WETH) :")
    print("-" * 50)
    print(f"Sortie Directe moyenne    : {df['direct_out'].mean():.2f} DAI")
    print(f"Sortie Optimale moyenne   : {df['optimal_out'].mean():.2f} DAI")
    print(f"Amélioration moyenne      : {df['improvement_bps'].mean():.2f} bps")
    print(f"Latence moyenne (Solveur) : {df['latency_ms'].mean():.2f} ms")
    
    best = df.loc[df['improvement_bps'].idxmax()]
    print("\nMeilleure opportunité détectée :")
    print(f"Bloc {int(best['block'])} -> Gain de {best['optimal_out'] - best['direct_out']:.2f} DAI (+{best['improvement_bps']:.2f} bps)")

if __name__ == "__main__":
    run_backtest()