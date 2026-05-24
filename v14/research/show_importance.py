import joblib
import pandas as pd
import glob
import os

def print_importance(model_path, title):
    print(f"\n=== {title} Feature Importance (Top 20) ===")
    try:
        model = joblib.load(model_path)
        
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            xgb_model = model.named_steps['classifier']
        else:
            xgb_model = model
            
        booster = xgb_model.get_booster()
        importance = booster.get_score(importance_type='gain')
        
        # If feature names are missing, they might just be f0, f1...
        # We need to map them if possible. Let's just print the raw dictionary first.
        
        # Sort by value
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, score) in enumerate(sorted_imp[:20]):
            print(f"{i+1:2d}. {feat:<30} {score:.4f}")
    except Exception as e:
        print(f"Error loading {model_path}: {e}")

if __name__ == "__main__":
    # Get the LAST cycle model (cycle 36) instead of cycle 9
    s1_models = sorted(glob.glob("runtime/bot_assets/wf_models_v14/filter_v14_cycle_*.joblib"), key=os.path.getmtime)
    if s1_models:
        print_importance(s1_models[-1], "Stage 1 (Filter) - Latest Cycle")
    
    s2_models = sorted(glob.glob("runtime/bot_assets/wf_models_v14/directional_v14_cycle_*.joblib"), key=os.path.getmtime)
    if s2_models:
        print_importance(s2_models[-1], "Stage 2 (Directional) - Latest Cycle")
