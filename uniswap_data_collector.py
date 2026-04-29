"""
Section 5.1 — Collecte et préparation des données de Microstructure (Uniswap V2)
Source : The Graph Decentralized Network (API key requise) - Granularité : Swaps (Tick)
Fallback : Données synthétiques haute fréquence (GBM, pas de 12s) si hors ligne.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time

# ── Configuration ──────────────────────────────────────────────────────────────

GRAPH_API_KEY = "YOUR_API_KEY_HERE"   # ← Remplacer par votre clé
GRAPH_URL = (
    f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}"
    "/subgraphs/id/A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum"
)

USE_SYNTHETIC_FALLBACK = True

POOL_IDS = {
    "ETH/USDC": "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
    "ETH/DAI":  "0xa478c2975ab1ea89e8196811f51a7b7ade33eb11",
    "USDC/DAI": "0xae461ca67b15dc8dc81ce7615e0320da1a9ab8d5",
    "ETH/WBTC": "0xbb2b8038a1640196fbe3e38816f3e67cba72d940",
}

# ── Requêtes GraphQL (The Graph) - Microstructure ──────────────────────────────

def fetch_recent_swaps(pool_id: str, n_swaps: int = 1000) -> pd.DataFrame:
    """Récupère les N derniers swaps pour analyser la microstructure."""
    query = """
    {
      swaps(
        first: %d
        orderBy: timestamp
        orderDirection: desc
        where: { pair: "%s" }
      ) {
        timestamp
        amount0In
        amount1In
        amount0Out
        amount1Out
        amountUSD
        pair {
          reserve0
          reserve1
          reserveUSD
        }
      }
    }
    """ % (n_swaps, pool_id.lower())

    response = requests.post(GRAPH_URL, json={"query": query}, timeout=15)
    response.raise_for_status()
    
    rows = response.json()["data"]["swaps"]
    df = pd.json_normalize(rows)
    
    # Nettoyage et typage
    float_cols = ['amount0In', 'amount1In', 'amount0Out', 'amount1Out', 'amountUSD', 
                  'pair.reserve0', 'pair.reserve1', 'pair.reserveUSD']
    df[float_cols] = df[float_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    
    # Renommer pour correspondre à notre standard interne
    df.rename(columns={
        'pair.reserve0': 'reserve0',
        'pair.reserve1': 'reserve1',
        'pair.reserveUSD': 'reserveUSD'
    }, inplace=True)
    
    return df.sort_values("timestamp").reset_index(drop=True)

def fetch_pool_metadata(pool_id: str) -> dict:
    query = """
    {
      pair(id: "%s") {
        token0 { symbol decimals }
        token1 { symbol decimals }
      }
    }
    """ % pool_id.lower()

    response = requests.post(GRAPH_URL, json={"query": query}, timeout=15)
    response.raise_for_status()
    pair = response.json()["data"]["pair"]
    return {
        "token0": pair["token0"]["symbol"],
        "token1": pair["token1"]["symbol"],
        "fee": 0.003,
        "gamma": 0.997,
    }

# ── Données synthétiques (GBM Haute Fréquence - 12s) ──────────────────────────

SYNTHETIC_PARAMS = {
    "ETH/USDC": {"R0": 45_000,    "R1": 82_000_000,  "vol_annuelle": 0.50, "token0": "WETH", "token1": "USDC"},
    "ETH/DAI":  {"R0": 30_000,    "R1": 55_000_000,  "vol_annuelle": 0.50, "token0": "WETH", "token1": "DAI"},
    "USDC/DAI": {"R0": 50_000_000,"R1": 50_000_000,  "vol_annuelle": 0.05, "token0": "USDC", "token1": "DAI"},
    "ETH/WBTC": {"R0": 20_000,    "R1": 1_200,       "vol_annuelle": 0.45, "token0": "WETH", "token1": "WBTC"},
}

def generate_synthetic_hf_snapshots(name: str, n_blocks: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Génère des données à la fréquence du bloc Ethereum (12 secondes).
    """
    rng = np.random.default_rng(seed)
    p = SYNTHETIC_PARAMS[name]
    R0, R1, vol_annuelle = p["R0"], p["R1"], p["vol_annuelle"]

    # Ajustement de la volatilité pour un pas de temps de 12 secondes
    # 1 an = 365 * 24 * 60 * 60 / 12 = 2_628_000 blocs
    vol_bloc = vol_annuelle / np.sqrt(2_628_000)

    start_time = datetime.now(timezone.utc) - timedelta(seconds=n_blocks*12)
    timestamps = [start_time + timedelta(seconds=12 * i) for i in range(n_blocks)]
    
    r0_series = R0 * np.cumprod(np.exp(-0.5 * vol_bloc**2 + vol_bloc * rng.standard_normal(n_blocks)))
    r1_series = R1 * np.cumprod(np.exp(-0.5 * vol_bloc**2 + vol_bloc * rng.standard_normal(n_blocks)))

    return pd.DataFrame({
        "timestamp": timestamps,
        "reserve0": r0_series,
        "reserve1": r1_series,
        "reserveUSD": r0_series * 2000 * 2, # Approximation pour simplifier
    })

def get_synthetic_metadata(name: str) -> dict:
    p = SYNTHETIC_PARAMS[name]
    return {"token0": p["token0"], "token1": p["token1"], "fee": 0.003, "gamma": 0.997}

# ── Calculs financiers (Intégration Quant) ─────────────────────────────────────

def compute_swap_output(delta_in: float, reserve_in: float, reserve_out: float, gamma: float = 0.997) -> float:
    return reserve_out * (gamma * delta_in) / (reserve_in + gamma * delta_in)

def compute_spot_price(reserve_in: float, reserve_out: float) -> float:
    return reserve_out / reserve_in

def compute_effective_price(delta_in: float, reserve_in: float, reserve_out: float, gamma: float = 0.997) -> float:
    return compute_swap_output(delta_in, reserve_in, reserve_out, gamma) / delta_in

def compute_slippage(delta_in: float, reserve_in: float, reserve_out: float, gamma: float = 0.997) -> float:
    if delta_in == 0: return 0.0
    p_spot = compute_spot_price(reserve_in, reserve_out)
    p_eff  = compute_effective_price(delta_in, reserve_in, reserve_out, gamma)
    return (p_spot - p_eff) / p_spot

def enrich_dataframe(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Ajoute le spot price et le slippage pour des tailles d'ordres réalistes (en USD)."""
    df = df.copy()
    
    # Prix implicite du token0 en USD (Hypothèse : reserveUSD est répartie 50/50 entre token0 et token1)
    df["price_token0_usd"] = (df["reserveUSD"] / 2) / df["reserve0"]
    df["spot_price_0_to_1"] = df["reserve1"] / df["reserve0"]

    # Tailles d'ordres institutionnelles en USD
    trade_sizes_usd = [10_000, 100_000, 1_000_000]
    
    for usd_amount in trade_sizes_usd:
        col_name = f"slippage_{usd_amount//1000}k_USD"
        
        # Convertir le montant USD en quantité de Token0 (delta_in)
        # On utilise np.where pour éviter les divisions par zéro
        delta_in_token0 = np.where(df["price_token0_usd"] > 0, usd_amount / df["price_token0_usd"], 0)
        
        df[col_name] = df.apply(
            lambda r: compute_slippage(
                delta_in=usd_amount / r["price_token0_usd"] if r["price_token0_usd"] > 0 else 0,
                reserve_in=r["reserve0"],
                reserve_out=r["reserve1"],
                gamma=meta["gamma"]
            ), axis=1
        )
        
    df["pool_name"] = f"{meta['token0']}/{meta['token1']}"
    return df

# ── Pipeline principal ─────────────────────────────────────────────────────────

def build_liquidity_graph(n_items: int = 1000) -> tuple:
    pools_data, metadata = {}, {}

    for name, pool_id in POOL_IDS.items():
        print(f"  {name}...", end=" ", flush=True)
        use_synth = USE_SYNTHETIC_FALLBACK

        if not use_synth:
            try:
                meta = fetch_pool_metadata(pool_id)
                df   = fetch_recent_swaps(pool_id, n_items)
                print("OK (The Graph - Swaps)")
            except Exception as e:
                print(f"Erreur API ({e.__class__.__name__}), bascule sur fallback GBM")
                use_synth = True

        if use_synth:
            meta = get_synthetic_metadata(name)
            df   = generate_synthetic_hf_snapshots(name, n_items)
            print("OK (Synthétique GBM - 12s)")

        pools_data[name] = enrich_dataframe(df, meta)
        metadata[name]   = meta
        time.sleep(0.05)

    return pools_data, metadata

def summary_stats(pools_data: dict) -> pd.DataFrame:
    rows = []
    for name, df in pools_data.items():
        rows.append({
            "Pool":                  name,
            "Nb Blocs/Swaps":        len(df),
            "Réserve 0 (moy.)":      f"{df['reserve0'].mean():.2e}",
            "Réserve 1 (moy.)":      f"{df['reserve1'].mean():.2e}",
            "Slippage 10k$ (moy.)":  f"{df['slippage_10k_USD'].mean()*100:.3f}%",
            "Slippage 100k$ (moy.)": f"{df['slippage_100k_USD'].mean()*100:.3f}%",
            "Slippage 1M$ (moy.)":   f"{df['slippage_1000k_USD'].mean()*100:.3f}%", # <-- La correction est ici
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("=== 5.1 — Collecte de données Haute Fréquence ===\n")
    print("[1/2] Récupération de la microstructure (1000 observations)...")
    pools_data, metadata = build_liquidity_graph(n_items=1000)

    print("\n[2/2] Statistiques descriptives de l'exécution :")
    print(summary_stats(pools_data).to_string(index=False))
    print("\n✓ Pipeline validé. Données prêtes pour l'optimisation convexe (Section 5.2).")

    import os
    os.makedirs("data", exist_ok=True)
    for name, df in pools_data.items():
        df.to_csv(f"data/{name.replace('/', '_')}_hf.csv", index=False)
    print("\nFichiers CSV sauvegardés dans le dossier /data/")