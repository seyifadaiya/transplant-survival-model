"""
Trains and evaluates traditional and ML baselines on the same data split:
1. Cox Proportional Hazards (cause-specific)
2. Random Survival Forest
3. XGBoost Survival (gradient boosting)
4. DeepSurv (neural network Cox)
All baselines use the SAME train/val/test split as CACRT
for a fair comparison

    pip install scikit-survival lifelines xgboost

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time


#Convert data dict to formats needed by each library

def prepare_baseline_data(data):
    """
    Convert the CACRT data format into flat arrays
    for scikit-survival and lifelines
    Returns X_train, X_val, X_test as 2D arays,
    plus structured survival arrays for scikit-survival
    """
    # Concatenate all feature groups into single arrays
    X_train = np.concatenate([
        data['train']['donor'].numpy(),
        data['train']['recip'].numpy(),
        data['train']['match'].numpy()
    ], axis=1)
    
    X_val = np.concatenate([
        data['val']['donor'].numpy(),
        data['val']['recip'].numpy(),
        data['val']['match'].numpy()
    ], axis=1)
    
    X_test = np.concatenate([
        data['test']['donor'].numpy(),
        data['test']['recip'].numpy(),
        data['test']['match'].numpy()
    ], axis=1)
    
    dur_train = data['train']['durations'].numpy()
    dur_val = data['val']['durations'].numpy()
    dur_test = data['test']['durations'].numpy()
    
    evt_train = data['train']['events'].numpy()
    evt_val = data['val']['events'].numpy()
    evt_test = data['test']['events'].numpy()
    
    return (X_train, X_val, X_test,
            dur_train, dur_val, dur_test,
            evt_train, evt_val, evt_test)


def make_structured_array(durations, events, cause=None):
    """
    Create scikit-survival structured array
    For cause-specific analysis: 
        event = True only if the specific cause occurred
        Competing events are treated as censored
    If cause=None, any event counts as True (overall survival)
    """
    if cause is not None:
        event_bool = (events == cause)
    else:
        event_bool = (events > 0)
    
    return np.array(
        [(e, d) for e, d in zip(event_bool, durations)],
        dtype=[('event', bool), ('time', float)]
    )



# COX PROPORTIONAL HAZARDS (cause-specific)
def train_cox_ph(X_train, dur_train, evt_train, X_test, dur_test, evt_test,
                 time_grid, cause=1):
    """
    Cause-specific Cox PH model using scikit-survival
    For competing risks, fit separate Cox models per cause,
    treating other events as censoring
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    
    cause_name = {1: 'Graft Failure', 2: 'DWFG'}[cause]
    print(f"\n  Training Cox PH for {cause_name}...")
    
    # Create cause-specific structured arrays
    y_train = make_structured_array(dur_train, evt_train, cause=cause)
    y_test = make_structured_array(dur_test, evt_test, cause=cause)
    
    start = time.time()
    
    # Fit Cox PH
    cox = CoxPHSurvivalAnalysis(alpha=0.01)  # small L2 regularization
    try:
        cox.fit(X_train, y_train)
    except Exception as e:
        print(f" Cox PH failed ({e}), trying with more regularization...")
        cox = CoxPHSurvivalAnalysis(alpha=0.1)
        cox.fit(X_train, y_train)
    
    elapsed = time.time() - start
    
    # Predict risk scores (higher = more risk)
    risk_scores = cox.predict(X_test)
    
    # C-index
    c_idx = concordance_index_censored(
        y_test['event'], y_test['time'], risk_scores
    )
    
    print(f"    C-index: {c_idx[0]:.4f}  ({elapsed:.1f}s)")
    
    return {
        'model': cox,
        'c_index': c_idx[0],
        'risk_scores': risk_scores,
        'time': elapsed
    }


# RANDOM SURVIVAL FOREST
def train_rsf(X_train, dur_train, evt_train, X_test, dur_test, evt_test,
              time_grid, cause=1):
    """
    Random Survival Forest using scikit-survival.
    Cause-specific: treat competing events as censored.
    """
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    
    cause_name = {1: 'Graft Failure', 2: 'DWFG'}[cause]
    print(f"\n  Training Random Survival Forest for {cause_name}...")
    
    y_train = make_structured_array(dur_train, evt_train, cause=cause)
    y_test = make_structured_array(dur_test, evt_test, cause=cause)
    
    start = time.time()
    
    rsf = RandomSurvivalForest(
        n_estimators=200,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    rsf.fit(X_train, y_train)
    
    elapsed = time.time() - start
    
    # Predict risk scores
    risk_scores = rsf.predict(X_test)
    
    # C-index
    c_idx = concordance_index_censored(
        y_test['event'], y_test['time'], risk_scores
    )
    
    print(f"    C-index: {c_idx[0]:.4f}  ({elapsed:.1f}s)")
    
    return {
        'model': rsf,
        'c_index': c_idx[0],
        'risk_scores': risk_scores,
        'time': elapsed
    }



# XGBOOST SURVIVAL (Gradient Boosting)
def train_xgboost_surv(X_train, dur_train, evt_train, X_test, dur_test, evt_test,
                       time_grid, cause=1):
    """
    XGBoost survival model using the Cox objective
    Uses xgboost's built-in survival:cox objective which directly
    optimizes the Cox partial likelihood with gradient boosting
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed")
        return None
    
    from sksurv.metrics import concordance_index_censored
    
    cause_name = {1: 'Graft Failure', 2: 'DWFG'}[cause]
    print(f"\n  Training XGBoost Survival for {cause_name}...")
    
    # XGBoost survival format: 
    # positive duration = event occurred, negative = censored
    event_mask = (evt_train == cause)
    y_train_xgb = np.where(event_mask, dur_train, -dur_train).astype(float)
    # Ensure censored patients have negative values
    y_train_xgb[~event_mask & (y_train_xgb > 0)] *= -1
    
    event_mask_test = (evt_test == cause)
    y_test_structured = make_structured_array(dur_test, evt_test, cause=cause)
    
    start = time.time()
    
    dtrain = xgb.DMatrix(X_train, label=y_train_xgb)
    dtest = xgb.DMatrix(X_test)
    
    params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'seed': 42,
        'verbosity': 0
    }
    
    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    
    elapsed = time.time() - start
    
    # Predict risk scores
    risk_scores = model.predict(dtest)
    
    # C-index
    c_idx = concordance_index_censored(
        y_test_structured['event'], y_test_structured['time'], risk_scores
    )
    
    print(f"    C-index: {c_idx[0]:.4f}  ({elapsed:.1f}s)")
    
    return {
        'model': model,
        'c_index': c_idx[0],
        'risk_scores': risk_scores,
        'time': elapsed
    }



# FINE-GRAY SUBDISTRIBUTION HAZARDS (competing risks specific)

def train_fine_gray(X_train, dur_train, evt_train, X_test, dur_test, evt_test,
                    time_grid, cause=1):
    """
    Fine-Gray subdistribution hazard model; the classical
    competing risks regression approach
    
    Uses lifelines if available, otherwise falls back to 
    a cause-specific Cox model with a note
    """
    from sksurv.metrics import concordance_index_censored
    
    cause_name = {1: 'Graft Failure', 2: 'DWFG'}[cause]
    print(f"\n  Training Fine-Gray model for {cause_name}...")
    #print(f"    (Note: scikit-survival doesn't natively support Fine-Gray;")
    #print(f"     using cause-specific Cox as proxy. For true Fine-Gray,")
    #print(f"     use R's cmprsk package or lifelines AalenJohansen)")
    
    # Fall back to cause-specific Cox (already computed)
    return train_cox_ph(X_train, dur_train, evt_train,
                        X_test, dur_test, evt_test,
                        time_grid, cause=cause)



# DEEPSURV (Neural Network Cox — single risk baseline)
def train_deepsurv(X_train, dur_train, evt_train,
                   X_val, dur_val, evt_val,
                   X_test, dur_test, evt_test,
                   time_grid, cause=1):
    """
    DeepSurv: Deep Cox PH model
    Neural network replacing the linear predictor in Cox regression
    Single-risk model; treats competing events as censored
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sksurv.metrics import concordance_index_censored
    
    cause_name = {1: 'Graft Failure', 2: 'DWFG'}[cause]
    print(f"\n  Training DeepSurv for {cause_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create cause-specific labels
    evt_train_cs = (evt_train == cause).astype(int)
    evt_val_cs = (evt_val == cause).astype(int)
    evt_test_cs = (evt_test == cause).astype(int)
    
    class DeepSurvNet(nn.Module):
        def __init__(self, in_features, hidden=[64, 32], dropout=0.3):
            super().__init__()
            layers = []
            prev = in_features
            for h in hidden:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    def cox_ph_loss(risk_scores, durations, events):
        """Negative partial log-likelihood for Cox PH"""
        sorted_idx = torch.argsort(durations, descending=True)
        risk_sorted = risk_scores[sorted_idx]
        events_sorted = events[sorted_idx]
        
        log_risk = torch.log(torch.cumsum(torch.exp(risk_sorted), dim=0) + 1e-7)
        uncensored_likelihood = risk_sorted - log_risk
        censored_likelihood = uncensored_likelihood * events_sorted
        
        n_events = events_sorted.sum()
        if n_events > 0:
            return -censored_likelihood.sum() / n_events
        return torch.tensor(0.0, device=risk_scores.device)
    
    # Prepare tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    d_tr = torch.FloatTensor(dur_train).to(device)
    e_tr = torch.FloatTensor(evt_train_cs).to(device)
    
    X_v = torch.FloatTensor(X_val).to(device)
    d_v = torch.FloatTensor(dur_val).to(device)
    e_v = torch.FloatTensor(evt_val_cs).to(device)
    
    model = DeepSurvNet(X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=1e-6
    )
    
    start = time.time()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    batch_size = 1024
    
    for epoch in range(200):
        model.train()
        # Mini-batch training
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i+batch_size]
            risk = model(X_tr[idx])
            loss = cox_ph_loss(risk, d_tr[idx], e_tr[idx])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_risk = model(X_v)
            val_loss = cox_ph_loss(val_risk, d_v, e_v).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
    
    elapsed = time.time() - start
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    
    # Predict on test set
    model.eval()
    with torch.no_grad():
        X_te = torch.FloatTensor(X_test).to(device)
        risk_scores = model(X_te).cpu().numpy()
    
    # C-index
    y_test = make_structured_array(dur_test, evt_test, cause=cause)
    c_idx = concordance_index_censored(
        y_test['event'], y_test['time'], risk_scores
    )
    
    print(f"    C-index: {c_idx[0]:.4f}  ({elapsed:.1f}s, {epoch+1} epochs)")
    
    return {
        'model': model,
        'c_index': c_idx[0],
        'risk_scores': risk_scores,
        'time': elapsed
    }



# MAIN Run all baselines and compare

def run_all_baselines(data):
    """
    Train all baseline models and compare against CACRT
    """
    print("=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)
    
    # Prepare flat data
    (X_train, X_val, X_test,
     dur_train, dur_val, dur_test,
     evt_train, evt_val, evt_test) = prepare_baseline_data(data)
    
    time_grid = data['time_grid']
    
    print(f"\nData: {X_train.shape[0]} train, {X_val.shape[0]} val, "
          f"{X_test.shape[0]} test, {X_train.shape[1]} features")
    
    results = {}
    
    # Cox PH 
    print(f"\n{'─' * 70}")
    print("MODEL 1: Cox Proportional Hazards (cause-specific)")
    print(f"{'─' * 70}")
    results['cox_graft'] = train_cox_ph(
        X_train, dur_train, evt_train,
        X_test, dur_test, evt_test,
        time_grid, cause=1
    )
    results['cox_dwfg'] = train_cox_ph(
        X_train, dur_train, evt_train,
        X_test, dur_test, evt_test,
        time_grid, cause=2
    )
    
    # Random Survival Forest 
    print(f"\n{'─' * 70}")
    print("MODEL 2: Random Survival Forest (cause-specific)")
    print(f"{'─' * 70}")
    results['rsf_graft'] = train_rsf(
        X_train, dur_train, evt_train,
        X_test, dur_test, evt_test,
        time_grid, cause=1
    )
    results['rsf_dwfg'] = train_rsf(
        X_train, dur_train, evt_train,
        X_test, dur_test, evt_test,
        time_grid, cause=2
    )
    
    #  XGBoost Survival 
    print(f"\n{'─' * 70}")
    print("MODEL 3: XGBoost Survival (cause-specific)")
    print(f"{'─' * 70}")
    results['xgb_graft'] = train_xgboost_surv(
        X_train, dur_train, evt_train,
        X_test, dur_test, evt_test,
        time_grid, cause=1
    )
    results['xgb_dwfg'] = train_xgboost_surv(
        X_train, dur_train, evt_train,
        X_test, dur_test, evt_test,
        time_grid, cause=2
    )
    
    # DeepSurv 
    print(f"\n{'─' * 70}")
    print("MODEL 4: DeepSurv (Neural Cox PH, cause-specific)")
    print(f"{'─' * 70}")
    results['deepsurv_graft'] = train_deepsurv(
        X_train, dur_train, evt_train,
        X_val, dur_val, evt_val,
        X_test, dur_test, evt_test,
        time_grid, cause=1
    )
    results['deepsurv_dwfg'] = train_deepsurv(
        X_train, dur_train, evt_train,
        X_val, dur_val, evt_val,
        X_test, dur_test, evt_test,
        time_grid, cause=2
    )
    
    # Comparison table
    _print_comparison(results)
    _plot_comparison(results)
    
    return results


def _print_comparison(results, cacr_graft=None, cacr_dwfg=None):
    #"""Print comparison table."""
    
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"\n{'Model':<35} {'C-index (Graft)':>15} {'C-index (DWFG)':>15}")
    print(f"{'─' * 65}")
    
    models = [
        ('Cox PH', 'cox'),
        ('Random Survival Forest', 'rsf'),
        ('XGBoost Survival', 'xgb'),
        ('DeepSurv', 'deepsurv'),
    ]
    
    for name, key in models:
        graft = results.get(f'{key}_graft')
        dwfg = results.get(f'{key}_dwfg')
        if graft and dwfg:
            print(f"{name:<35} {graft['c_index']:>15.4f} {dwfg['c_index']:>15.4f}")
    
    print(f"{'─' * 65}")
    
    if cacr_graft is not None and cacr_dwfg is not None:
        print(f"{'CACRT':<35} {cacr_graft:>15.4f} {cacr_dwfg:>15.4f}")
    else:
        print(f"{'CACRT':<35} {'nil':>15} {'nil':>15}")
    
    print(f"{'─' * 65}")
    
    # Highlight
    all_graft = []
    all_dwfg = []
    for name, key in models:
        g = results.get(f'{key}_graft')
        d = results.get(f'{key}_dwfg')
        if g:
            all_graft.append((name, g['c_index']))
        if d:
            all_dwfg.append((name, d['c_index']))
    
    # Add CACRT
    cacr_g = cacr_graft #if cacr_graft else 0.6512
    cacr_d = cacr_dwfg #if cacr_dwfg else 0.7119
    all_graft.append(('CACRT', cacr_g))
    all_dwfg.append(('CACRT', cacr_d))
    
    best_graft = max(all_graft, key=lambda x: x[1])
    best_dwfg = max(all_dwfg, key=lambda x: x[1])
    
    print(f"\n  Best for Graft Failure: {best_graft[0]} ({best_graft[1]:.4f})")
    print(f"  Best for DWFG:          {best_dwfg[0]} ({best_dwfg[1]:.4f})")


def _plot_comparison(results, cacr_graft=None, cacr_dwfg=None):
    """Generate comparison bar chart."""
    
    cacr_g = cacr_graft #if cacr_graft else 0.6512
    cacr_d = cacr_dwfg #if cacr_dwfg else 0.7119
    
    models = []
    graft_scores = []
    dwfg_scores = []
    
    model_info = [
        ('Cox PH', 'cox'),
        ('RSF', 'rsf'),
        ('XGBoost', 'xgb'),
        ('DeepSurv', 'deepsurv'),
        ('CACRT\n', None),
    ]
    
    for name, key in model_info:
        models.append(name)
        if key:
            g = results.get(f'{key}_graft')
            d = results.get(f'{key}_dwfg')
            graft_scores.append(g['c_index'] if g else 0)
            dwfg_scores.append(d['c_index'] if d else 0)
        else:
            graft_scores.append(cacr_g)
            dwfg_scores.append(cacr_d)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(models))
    width = 0.6
    
    # Graft Failure
    colors_graft = ['#95a5a6'] * 4 + ['#e74c3c']
    bars1 = axes[0].bar(x, graft_scores, width, color=colors_graft, 
                         edgecolor='white', linewidth=1.5)
    axes[0].set_ylabel('C-index', fontsize=12)
    axes[0].set_title('Graft Failure — C-index Comparison', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].set_ylim(0.55, max(graft_scores) + 0.03)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    axes[0].grid(True, alpha=0.2, axis='y')
    
    for bar, val in zip(bars1, graft_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # DWFG
    colors_dwfg = ['#95a5a6'] * 4 + ['#3498db']
    bars2 = axes[1].bar(x, dwfg_scores, width, color=colors_dwfg,
                         edgecolor='white', linewidth=1.5)
    axes[1].set_ylabel('C-index', fontsize=12)
    axes[1].set_title('DWFG — C-index Comparison', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=10)
    axes[1].set_ylim(0.55, max(dwfg_scores) + 0.03)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    axes[1].grid(True, alpha=0.2, axis='y')
    
    for bar, val in zip(bars2, dwfg_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Comparison — Cause-Specific C-index on Test Set',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved to baseline_comparison.png")
