"""
Section 5.2 — Algorithme de Routage Optimal (Smart Order Router)
Version : Production-Ready pour Données Réelles (Web3/Alchemy)
Objectif : Maximiser la sortie DAI pour une entrée WETH fixe.
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import time
import os
import warnings

# On ignore les warnings de convergence pour plus de clarté
warnings.filterwarnings("ignore")

# ── 1. CONFIGURATION DU RÉSEAU (MAPPING RÉEL) ────────────────────────────────

TOKENS = ["WETH", "USDC", "DAI", "WBTC"]
TOKEN_IDX = {token: i for i, token in enumerate(TOKENS)}

# Mapping basé sur l'extraction Uniswap V2 réelle :
# ETH/USDC : 0=USDC, 1=WETH | ETH/DAI : 0=DAI, 1=WETH 
# USDC/DAI : 0=DAI, 1=USDC | ETH/WBTC : 0=WETH, 1=WBTC
POOLS = {
    "ETH/USDC": (TOKEN_IDX["USDC"], TOKEN_IDX["WETH"]),
    "ETH/DAI":  (TOKEN_IDX["DAI"],  TOKEN_IDX["WETH"]),
    "USDC/DAI": (TOKEN_IDX["DAI"],  TOKEN_IDX["USDC"]),
    "ETH/WBTC": (TOKEN_IDX["WETH"], TOKEN_IDX["WBTC"]),
}

GAMMA = 0.997  # Frais de 0.3%

# ── 2. CHARGEMENT DES ÉTATS DE MARCHÉ ───────────────────────────────────────

def load_market_state(data_dir="data", row_idx=0):
    """Charge les réserves réelles à partir des fichiers CSV générés."""
    state = {}
    for p_name in POOLS.keys():
        file_path = os.path.join(data_dir, f"{p_name.replace('/', '_')}_hf.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Donnée manquante : {file_path}")
        df = pd.read_csv(file_path)
        state[p_name] = {
            "R0": float(df.iloc[row_idx]["reserve0"]),
            "R1": float(df.iloc[row_idx]["reserve1"])
        }
    return state

# ── 3. CŒUR DE L'OPTIMISATION (CONVEXE) ──────────────────────────────────────

def route_optimal_swap(market_state, amount_in_weth):
    n_tokens = len(TOKENS)
    # Variables de décision : Deltas (ce qu'on donne au pool) et Lambdas (ce qu'on reçoit)
    deltas = {p: cp.Variable(2, nonneg=True) for p in POOLS.keys()}
    lambdas = {p: cp.Variable(2, nonneg=True) for p in POOLS.keys()}
    
    constraints = []
    
    for p_name, (idx0, idx1) in POOLS.items():
        R0 = market_state[p_name]["R0"]
        R1 = market_state[p_name]["R1"]
        D, L = deltas[p_name], lambdas[p_name]
        
        # NORMALISATION NUMÉRIQUE : 
        # On divise les expressions par les réserves pour travailler autour de 1.0
        # Cela évite que le solveur Clarabel ne crash avec des grands nombres.
        post_R0 = (R0 + GAMMA * D[0] - L[0]) / R0
        post_R1 = (R1 + GAMMA * D[1] - L[1]) / R1
        
        # Invariant de produit constant (relaxation convexe)
        constraints += [cp.geo_mean(cp.hstack([post_R0, post_R1])) >= 1.0]
        
        # Sécurité : on ne peut pas sortir plus que ce qui est dans le pool
        constraints += [L[0] <= R0 * 0.95, L[1] <= R1 * 0.95]

    # Conservation des flux (Net Transaction Vector Psi)
    psi = [0] * n_tokens
    for p_name, (idx0, idx1) in POOLS.items():
        psi[idx0] += lambdas[p_name][0] - deltas[p_name][0]
        psi[idx1] += lambdas[p_name][1] - deltas[p_name][1]
        
    idx_eth = TOKEN_IDX["WETH"]
    idx_dai = TOKEN_IDX["DAI"]
    
    # On donne amount_in_weth au réseau
    constraints += [psi[idx_eth] == -amount_in_weth]
    
    # Pour les jetons intermédiaires (USDC, WBTC), le bilan final doit être >= 0
    for i in range(n_tokens):
        if i != idx_eth and i != idx_dai:
            constraints += [psi[i] >= 0]

    # Résolution
    start_time = time.time()
    objective = cp.Maximize(psi[idx_dai])
    prob = cp.Problem(objective, constraints)
    
    try:
        # On utilise Clarabel avec des paramètres de tolérance adaptés à la réalité
        prob.solve(solver=cp.CLARABEL, verbose=False, tol_gap_abs=1e-4)
        latency = (time.time() - start_time) * 1000
        
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return max(0, psi[idx_dai].value), latency
    except Exception:
        pass
    
    return 0.0, 0.0

# ── 4. BENCHMARK : ROUTE DIRECTE ─────────────────────────────────────────────

def get_direct_route_output(market_state, amount_in):
    """Calcule la sortie si on passait uniquement par le pool ETH/DAI."""
    # Dans ETH/DAI, DAI est Token0 (idx0) et WETH est Token1 (idx1)
    r_dai = market_state["ETH/DAI"]["R0"]
    r_eth = market_state["ETH/DAI"]["R1"]
    
    # Formule standard Uniswap V2 : out = (R_out * gamma * delta_in) / (R_in + gamma * delta_in)
    return r_dai * (GAMMA * amount_in) / (r_eth + GAMMA * amount_in)

# ── 5. RUN DU BACKTEST ───────────────────────────────────────────────────────

def run_backtest():
    print("=== Démarrage du Backtest (Section 5.2) ===")
    TRADE_SIZE = 50.0  # WETH
    
    results = []
    
    # On boucle sur les 100 snapshots de données réelles
    for i in range(100):
        try:
            state = load_market_state(row_idx=i)
        except Exception:
            break
            
        direct_out = get_direct_route_output(state, TRADE_SIZE)
        opt_out, latency = route_optimal_swap(state, TRADE_SIZE)
        
        if opt_out > 0:
            diff_bps = ((opt_out - direct_out) / direct_out) * 10000
            results.append({
                "direct": direct_out,
                "optimal": opt_out,
                "improvement_bps": diff_bps,
                "latency": latency
            })

    if not results:
        print("❌ Erreur : Le solveur n'a trouvé aucune solution. Vérifiez vos fichiers de données.")
        return

    df = pd.DataFrame(results)
    
    print("-" * 50)
    print(f"Résultats sur {len(df)} transactions (Taille : {TRADE_SIZE} WETH) :")
    print("-" * 50)
    print(f"Sortie Directe moyenne    : {df['direct'].mean():.2f} DAI")
    print(f"Sortie Optimale moyenne   : {df['optimal'].mean():.2f} DAI")
    print(f"Amélioration moyenne      : {df['improvement_bps'].mean():.2f} bps")
    print(f"Latence moyenne (Solveur) : {df['latency'].mean():.2f} ms")
    
    best = df.loc[df['improvement_bps'].idxmax()]
    print(f"\nMeilleure opportunité détectée :")
    print(f"Gain de {best['optimal'] - best['direct']:.2f} DAI (+{best['improvement_bps']:.2f} bps)")

if __name__ == "__main__":
    run_backtest()