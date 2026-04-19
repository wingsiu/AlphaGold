"""Quick test: place a real BUY order on Gold, size=0.05, then print result."""
import json
from ig_scripts.ig_data_api import IGService, API_CONFIG, Price, place_otc_market_order

ig = IGService(
    api_key=API_CONFIG["api_key"],
    username=API_CONFIG["username"],
    password=API_CONFIG["password"],
    base_url=API_CONFIG["base_url"],
)

print(f"Using Gold epic: {Price.Gold.epic}")

result = place_otc_market_order(
    ig,
    Price.Gold,
    "SELL",
    size=0.05,
    stop_distance=15.0,    # 15 pts absolute stop
    limit_distance=38.0,   # ~0.8% target at ~4800
)

print(json.dumps(result, indent=2, default=str))

