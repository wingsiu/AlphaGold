import joblib
import numpy as np

# Load the latest S1 cycle model
model = joblib.load('runtime/bot_assets/wf_models_v14/filter_v14_cycle_36_2026-05-08.joblib')

booster = model.get_booster()
importance_gain = booster.get_score(importance_type='gain')

sorted_gain = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)

print("=== Top 20 Features by GAIN (Latest S1 Filter Model) ===")
for i, (feat, score) in enumerate(sorted_gain[:20]):
    print(f"{i+1:2d}. {feat:<25} {score:.4f}")

print("\n")

# Load the latest S2 cycle model
model2 = joblib.load('runtime/bot_assets/wf_models_v14/directional_v14_cycle_36_2026-05-08.joblib')

booster2 = model2.get_booster()
importance_gain2 = booster2.get_score(importance_type='gain')

sorted_gain2 = sorted(importance_gain2.items(), key=lambda x: x[1], reverse=True)

print("=== Top 20 Features by GAIN (Latest S2 Directional Model) ===")
for i, (feat, score) in enumerate(sorted_gain2[:20]):
    print(f"{i+1:2d}. {feat:<25} {score:.4f}")

