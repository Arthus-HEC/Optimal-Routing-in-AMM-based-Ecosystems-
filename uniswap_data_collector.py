from web3 import Web3
import pandas as pd
import os

# ── CONFIGURATION (Utilise ton lien Alchemy) ─────────────────────────────────
RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/a94aa98fc46641a5952d8f5f2634af1d" 
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Définition précise (Vérifiée sur Etherscan)
POOLS = {
    "ETH/USDC": {"addr": "0xB4e16d0168e52d35caCd2c6185b44281Ec28C9Dc", "d0": 6,  "d1": 18, "t0": "USDC", "t1": "WETH"},
    "ETH/DAI":  {"addr": "0xa478c2975ab1ea89e8196811f51a7b7ade33eb11", "d0": 18, "d1": 18, "t0": "DAI",  "t1": "WETH"},
    "USDC/DAI": {"addr": "0xAE461CA67B15dC8dc81ce7615E0320da1a9ab8d5", "d0": 18, "d1": 6,  "t0": "DAI",  "t1": "USDC"},
    "ETH/WBTC": {"addr": "0xBb2b8038a1640196FBe3e38816F3E67cbA72D940", "d0": 18, "d1": 8,  "t0": "WETH", "t1": "WBTC"}
}

ABI = [{"constant":True,"inputs":[],"name":"getReserves","outputs":[{"name":"_0","type":"uint112"},{"name":"_1","type":"uint112"},{"name":"_2","type":"uint32"}],"type":"function"}]

def collect_final_data():
    if not w3.is_connected(): return
    
    print(f"=== EXTRACTION RÉELLE (Bloc {w3.eth.block_number}) ===")
    os.makedirs("data", exist_ok=True)
    
    # On récupère le prix de l'ETH via USDC pour le reserveUSD
    p_eth_usdc = w3.eth.contract(address=w3.to_checksum_address(POOLS["ETH/USDC"]["addr"]), abi=ABI).functions.getReserves().call()
    eth_price = (p_eth_usdc[0]/1e6) / (p_eth_usdc[1]/1e18)

    for name, p in POOLS.items():
        print(f"  {name}...", end=" ", flush=True)
        try:
            contract = w3.eth.contract(address=w3.to_checksum_address(p["addr"]), abi=ABI)
            res = contract.functions.getReserves().call()
            r0, r1 = res[0] / (10**p["d0"]), res[1] / (10**p["d1"])
            
            # Calcul intelligent de la valeur du pool en USD
            if "USDC" in name or "DAI" in name:
                usd_val = r0 * 2 if p["t0"] in ["USDC", "DAI"] else r1 * 2
            else: # Pool ETH/WBTC
                usd_val = r0 * eth_price * 2
                
            df = pd.DataFrame([{"reserve0": r0, "reserve1": r1, "reserveUSD": usd_val}] * 100)
            df.to_csv(f"data/{name.replace('/', '_')}_hf.csv", index=False)
            print(f"✅ (r0: {r0:,.2f} {p['t0']} | r1: {r1:,.2f} {p['t1']})")
        except Exception as e:
            print(f"❌ {e}")

if __name__ == "__main__":
    collect_final_data()