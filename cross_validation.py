"""
Cross-Validation Pipeline with Statistical Testing
Runs 5-fold stratified CV for CACRT and all baselines,
computes confidence intervals and paired t-tests.
    pip install scikit-survival xgboost scipy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

from model import CACRT
from losses import CompetingRisksLoss



#Prepare data for one fold
def prepare_fold_data(X_donor, X_recip, X_match, durations, events,
                      train_idx, test_idx, num_time_bins=100):
    """
    Standardize and discretize data for a single fold.
    """
    # Standardize; fit on train only
    scaler_d = StandardScaler()
    scaler_r = StandardScaler()
    scaler_m = StandardScaler()

    Xd_train = scaler_d.fit_transform(X_donor[train_idx]).astype('float32')
    Xd_test = scaler_d.transform(X_donor[test_idx]).astype('float32')

    Xr_train = scaler_r.fit_transform(X_recip[train_idx]).astype('float32')
    Xr_test = scaler_r.transform(X_recip[test_idx]).astype('float32')

    Xm_train = scaler_m.fit_transform(X_match[train_idx]).astype('float32')
    Xm_test = scaler_m.transform(X_match[test_idx]).astype('float32')

    dur_train = durations[train_idx]
    dur_test = durations[test_idx]
    evt_train = events[train_idx]
    evt_test = events[test_idx]

    # Discretize time
    time_grid = np.linspace(0, dur_train.max(), num_time_bins + 1)[1:]
    dur_train_disc = np.clip(np.digitize(dur_train, time_grid), 0, num_time_bins - 1).astype('int64')
    dur_test_disc = np.clip(np.digitize(dur_test, time_grid), 0, num_time_bins - 1).astype('int64')

    # Split train into train/val (85/15) for early stopping
    n_train = len(train_idx)
    val_size = int(n_train * 0.15)
    perm = np.random.permutation(n_train)
    val_idx_local = perm[:val_size]
    train_idx_local = perm[val_size:]

    fold_data = {
        'train': {
            'donor': torch.FloatTensor(Xd_train[train_idx_local]),
            'recip': torch.FloatTensor(Xr_train[train_idx_local]),
            'match': torch.FloatTensor(Xm_train[train_idx_local]),
            'durations': torch.LongTensor(dur_train_disc[train_idx_local]),
            'events': torch.LongTensor(evt_train[train_idx_local]),
        },
        'val': {
            'donor': torch.FloatTensor(Xd_train[val_idx_local]),
            'recip': torch.FloatTensor(Xr_train[val_idx_local]),
            'match': torch.FloatTensor(Xm_train[val_idx_local]),
            'durations': torch.LongTensor(dur_train_disc[val_idx_local]),
            'events': torch.LongTensor(evt_train[val_idx_local]),
        },
        'test': {
            'donor': torch.FloatTensor(Xd_test),
            'recip': torch.FloatTensor(Xr_test),
            'match': torch.FloatTensor(Xm_test),
            'durations': torch.LongTensor(dur_test_disc),
            'events': torch.LongTensor(evt_test.astype('int64')),
        },
        'time_grid': time_grid,
        'num_time_bins': num_time_bins,
        # Flat arrays for baselines
        'X_train_flat': np.concatenate([Xd_train, Xr_train, Xm_train], axis=1),
        'X_test_flat': np.concatenate([Xd_test, Xr_test, Xm_test], axis=1),
        'X_val_flat': np.concatenate([
            Xd_train[val_idx_local], Xr_train[val_idx_local], Xm_train[val_idx_local]
        ], axis=1),
        'dur_train': dur_train_disc,
        'dur_test': dur_test_disc,
        'evt_train': evt_train,
        'evt_test': evt_test,
        'dur_val': dur_train_disc[val_idx_local],
        'evt_val': evt_train[val_idx_local],
    }

    return fold_data



# C-INDEX COMPUTATION ;consistent with full evaluation
def compute_cindex_numpy(cif, durations, events, cause=1):
    """Cause-specific C-index on numpy arrays"""
    cause_idx = cause - 1
    event_indices = np.where(events == cause)[0]

    concordant = 0
    discordant = 0
    tied = 0

    for i in event_indices:
        t_i = int(min(durations[i], cif.shape[2] - 1))
        at_risk = durations > durations[i]
        if not at_risk.any():
            continue

        risk_i = cif[i, cause_idx, t_i]
        risk_j = cif[at_risk, cause_idx, t_i]

        concordant += np.sum(risk_i > risk_j)
        discordant += np.sum(risk_i < risk_j)
        tied += np.sum(risk_i == risk_j)

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total



# TRAIN CACRT FOR ONE FOLD
def train_cacr_fold(fold_data, config, device, verbose=False):
  
    from torch.utils.data import DataLoader, TensorDataset

    # Create dataloaders
    train_ds = TensorDataset(
        fold_data['train']['donor'], fold_data['train']['recip'],
        fold_data['train']['match'], fold_data['train']['durations'],
        fold_data['train']['events']
    )
    val_ds = TensorDataset(
        fold_data['val']['donor'], fold_data['val']['recip'],
        fold_data['val']['match'], fold_data['val']['durations'],
        fold_data['val']['events']
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'] * 2,
                            shuffle=False, num_workers=0)

    # Initialize model
    model = CACRT(
        n_donor_features=fold_data['train']['donor'].size(1),
        n_recip_features=fold_data['train']['recip'].size(1),
        n_match_features=fold_data['train']['match'].size(1),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_self_layers=config['n_self_layers'],
        n_cross_layers=config['n_cross_layers'],
        d_ff=config['d_ff'],
        num_risks=config['num_risks'],
        num_time_bins=config['num_time_bins'],
        dropout=config['dropout']
    ).to(device)

    criterion = CompetingRisksLoss(
        alpha=config['alpha'], sigma=config['sigma'],
        num_risks=config['num_risks']
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(config['max_epochs']):
        # Train
        model.train()
        for batch in train_loader:
            d, r, m, dur, evt = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pmf, cif, surv = model(d, r, m)
            loss, _, _ = criterion(pmf, cif, surv, dur, evt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                d, r, m, dur, evt = [b.to(device) for b in batch]
                pmf, cif, surv = model(d, r, m)
                loss, _, _ = criterion(pmf, cif, surv, dur, evt)
                val_loss += loss.item()
                n_val += 1
        val_loss /= n_val

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        d = fold_data['test']['donor'].to(device)
        r = fold_data['test']['recip'].to(device)
        m = fold_data['test']['match'].to(device)

        # Batch prediction
        all_cif = []
        bs = 2048
        for start in range(0, len(d), bs):
            end = min(start + bs, len(d))
            _, cif, _ = model(d[start:end], r[start:end], m[start:end])
            all_cif.append(cif.cpu().numpy())
        cif_np = np.concatenate(all_cif, axis=0)

    dur_test = fold_data['test']['durations'].numpy()
    evt_test = fold_data['test']['events'].numpy()

    c_graft = compute_cindex_numpy(cif_np, dur_test, evt_test, cause=1)
    c_dwfg = compute_cindex_numpy(cif_np, dur_test, evt_test, cause=2)

    return c_graft, c_dwfg, epoch + 1



# TRAIN BASELINES FOR ONE FOLD
def make_structured_array(durations, events, cause):
    event_bool = (events == cause)
    return np.array(
        [(e, d) for e, d in zip(event_bool, durations)],
        dtype=[('event', bool), ('time', float)]
    )


def train_cox_fold(fold_data, cause=1):
  #  Cox PH for one fold
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    X_train = fold_data['X_train_flat']
    X_test = fold_data['X_test_flat']
    y_train = make_structured_array(fold_data['dur_train'], fold_data['evt_train'], cause)
    y_test = make_structured_array(fold_data['dur_test'], fold_data['evt_test'], cause)

    cox = CoxPHSurvivalAnalysis(alpha=0.01)
    try:
        cox.fit(X_train, y_train)
    except Exception:
        cox = CoxPHSurvivalAnalysis(alpha=0.1)
        cox.fit(X_train, y_train)

    risk = cox.predict(X_test)
    c_idx = concordance_index_censored(y_test['event'], y_test['time'], risk)[0]
    return c_idx


def train_rsf_fold(fold_data, cause=1):
    #Random Survival Forest for one fold
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored

    X_train = fold_data['X_train_flat']
    X_test = fold_data['X_test_flat']
    y_train = make_structured_array(fold_data['dur_train'], fold_data['evt_train'], cause)
    y_test = make_structured_array(fold_data['dur_test'], fold_data['evt_test'], cause)

    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=8, min_samples_split=20,
        min_samples_leaf=10, max_features='sqrt',
        n_jobs=-1, random_state=42
    )
    rsf.fit(X_train, y_train)
    risk = rsf.predict(X_test)
    c_idx = concordance_index_censored(y_test['event'], y_test['time'], risk)[0]
    return c_idx


def train_xgb_fold(fold_data, cause=1):
    #XGBoost survival for one fold
    try:
        import xgboost as xgb
    except ImportError:
        return None
    from sksurv.metrics import concordance_index_censored

    X_train = fold_data['X_train_flat']
    X_test = fold_data['X_test_flat']
    evt_train = fold_data['evt_train']
    dur_train = fold_data['dur_train']

    event_mask = (evt_train == cause)
    y_xgb = np.where(event_mask, dur_train, -dur_train).astype(float)
    y_xgb[~event_mask & (y_xgb > 0)] *= -1

    dtrain = xgb.DMatrix(X_train, label=y_xgb)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'survival:cox', 'eval_metric': 'cox-nloglik',
        'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 50,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'tree_method': 'hist', 'seed': 42, 'verbosity': 0
    }
    model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    risk = model.predict(dtest)

    y_test = make_structured_array(fold_data['dur_test'], fold_data['evt_test'], cause)
    c_idx = concordance_index_censored(y_test['event'], y_test['time'], risk)[0]
    return c_idx


def train_deepsurv_fold(fold_data, cause=1):
    #"DeepSurv for one fold
    from sksurv.metrics import concordance_index_censored

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = fold_data['X_train_flat']
    X_val = fold_data['X_val_flat']
    X_test = fold_data['X_test_flat']

    evt_train_cs = (fold_data['evt_train'] == cause).astype(int)
    evt_val_cs = (fold_data['evt_val'] == cause).astype(int)

    class DeepSurvNet(nn.Module):
        def __init__(self, in_f):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_f, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def cox_loss(risk, dur, evt):
        idx = torch.argsort(dur, descending=True)
        risk_s = risk[idx]
        evt_s = evt[idx]
        log_risk = torch.log(torch.cumsum(torch.exp(risk_s), dim=0) + 1e-7)
        loss = -(risk_s - log_risk) * evt_s
        n = evt_s.sum()
        return loss.sum() / n if n > 0 else loss.sum()

    X_tr_t = torch.FloatTensor(X_train).to(device)
    d_tr_t = torch.FloatTensor(fold_data['dur_train']).to(device)
    e_tr_t = torch.FloatTensor(evt_train_cs).to(device)
    X_v_t = torch.FloatTensor(X_val).to(device)
    d_v_t = torch.FloatTensor(fold_data['dur_val']).to(device)
    e_v_t = torch.FloatTensor(evt_val_cs).to(device)

    model = DeepSurvNet(X_train.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    best_vl = float('inf')
    best_st = None
    pat = 0

    for ep in range(200):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), 1024):
            idx = perm[i:i+1024]
            r = model(X_tr_t[idx])
            loss = cox_loss(r, d_tr_t[idx], e_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = cox_loss(model(X_v_t), d_v_t, e_v_t).item()
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 20:
                break

    if best_st:
        model.load_state_dict(best_st)
        model.to(device)

    model.eval()
    with torch.no_grad():
        risk = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    y_test = make_structured_array(fold_data['dur_test'], fold_data['evt_test'], cause)
    c_idx = concordance_index_censored(y_test['event'], y_test['time'], risk)[0]
    return c_idx



# MAIN 5-FOLD CROSS-VALIDATION

def run_cross_validation(filepath, config, donor_features, recip_features,
                         match_features, n_folds=5):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"{n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(filepath)

    # Extract features and targets
    X_donor = df[donor_features].values.astype('float32')
    X_recip = df[recip_features].values.astype('float32')
    X_match = df[match_features].values.astype('float32')

    durations = df['GTIME_YEARS'].values.astype('float32')

    # Event encoding
    events = np.where(
        df['GSTATUS_KI'] == 0, 0,
        np.where(df['DWFG_KI'] == 1, 2, 1)
    ).astype('int64')

    print(f"  Samples: {len(durations):,}")
    print(f"  Features: {X_donor.shape[1]} donor + {X_recip.shape[1]} recip + "
          f"{X_match.shape[1]} match = {X_donor.shape[1]+X_recip.shape[1]+X_match.shape[1]}")

    # Storage for results
    model_names = ['CACRT', 'Cox PH', 'RSF', 'XGBoost', 'DeepSurv']
    results = {name: {'graft': [], 'dwfg': []} for name in model_names}

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    total_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_donor, events)):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"  Train: {len(train_idx):,} | Test: {len(test_idx):,}")

        # Prepare fold data
        fold_data = prepare_fold_data(
            X_donor, X_recip, X_match, durations, events,
            train_idx, test_idx, config['num_time_bins']
        )

        # CACRT
        print(f"\n  Training CACRT...", end=" ")
        start = time.time()
        c_graft, c_dwfg, n_epochs = train_cacr_fold(fold_data, config, device)
        elapsed = time.time() - start
        results['CACRT']['graft'].append(c_graft)
        results['CACRT']['dwfg'].append(c_dwfg)
        print(f"Graft={c_graft:.4f}, DWFG={c_dwfg:.4f} "
              f"({elapsed:.0f}s, {n_epochs} epochs)")

        # Cox PH 
        print(f"  Training Cox PH...", end=" ")
        start = time.time()
        c_g = train_cox_fold(fold_data, cause=1)
        c_d = train_cox_fold(fold_data, cause=2)
        elapsed = time.time() - start
        results['Cox PH']['graft'].append(c_g)
        results['Cox PH']['dwfg'].append(c_d)
        print(f"Graft={c_g:.4f}, DWFG={c_d:.4f} ({elapsed:.0f}s)")

        # RSF
        print(f"  Training RSF...", end=" ")
        start = time.time()
        c_g = train_rsf_fold(fold_data, cause=1)
        c_d = train_rsf_fold(fold_data, cause=2)
        elapsed = time.time() - start
        results['RSF']['graft'].append(c_g)
        results['RSF']['dwfg'].append(c_d)
        print(f"Graft={c_g:.4f}, DWFG={c_d:.4f} ({elapsed:.0f}s)")

        # XGBoost
        print(f"  Training XGBoost...", end=" ")
        start = time.time()
        c_g = train_xgb_fold(fold_data, cause=1)
        c_d = train_xgb_fold(fold_data, cause=2)
        elapsed = time.time() - start
        if c_g is not None:
            results['XGBoost']['graft'].append(c_g)
            results['XGBoost']['dwfg'].append(c_d)
            print(f"Graft={c_g:.4f}, DWFG={c_d:.4f} ({elapsed:.0f}s)")
        else:
            print("Skipped (xgboost not installed)")

        # DeepSurv 
        print(f"  Training DeepSurv...", end=" ")
        start = time.time()
        c_g = train_deepsurv_fold(fold_data, cause=1)
        c_d = train_deepsurv_fold(fold_data, cause=2)
        elapsed = time.time() - start
        results['DeepSurv']['graft'].append(c_g)
        results['DeepSurv']['dwfg'].append(c_d)
        print(f"Graft={c_g:.4f}, DWFG={c_d:.4f} ({elapsed:.0f}s)")

    total_elapsed = time.time() - total_start
    print(f"\n\nTotal CV time: {total_elapsed/60:.1f} minutes")

    # Summary statistics
    _print_cv_summary(results, model_names)
    _plot_cv_results(results, model_names)

    return results



# SUMMARY TABLE WITH CONFIDENCE INTERVALS AND P-VALUES
def _print_cv_summary(results, model_names):
    #"""Print CV summary with mean ± std and paired t-tests

    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION RESULTS (mean ± std)")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<25} {'C-index Graft Failure':>25} {'C-index DWFG':>25}")
    print(f"{'─' * 75}")

    for name in model_names:
        graft = results[name]['graft']
        dwfg = results[name]['dwfg']
        if len(graft) > 0:
            print(f"{name:<25} "
                  f"{np.mean(graft):.4f} ± {np.std(graft):.4f}      "
                  f"{np.mean(dwfg):.4f} ± {np.std(dwfg):.4f}")

    #  Paired t-tests: CACRT vs each baseline 
    print(f"\n{'=' * 80}")
    print("PAIRED T-TESTS: CACRT vs Baselines")
    print(f"{'=' * 80}")
    print(f"\n{'Comparison':<40} {'Graft p-value':>15} {'DWFG p-value':>15} {'Significance':>12}")
    print(f"{'─' * 82}")

    cacr_graft = np.array(results['CACRT']['graft'])
    cacr_dwfg = np.array(results['CACRT']['dwfg'])

    for name in model_names[1:]:  # skip CACRT
        graft = np.array(results[name]['graft'])
        dwfg = np.array(results[name]['dwfg'])

        if len(graft) == 0:
            continue

        # Paired t-test 
        t_graft, p_graft = stats.ttest_rel(cacr_graft, graft)
        t_dwfg, p_dwfg = stats.ttest_rel(cacr_dwfg, dwfg)

        # One-sided p-value 
        p_graft_one = p_graft / 2 if t_graft > 0 else 1 - p_graft / 2
        p_dwfg_one = p_dwfg / 2 if t_dwfg > 0 else 1 - p_dwfg / 2

        sig = ""
        if p_graft_one < 0.05:
            sig = "✓ (p<0.05)"
        if p_graft_one < 0.01:
            sig = "✓✓ (p<0.01)"
        if p_graft_one < 0.001:
            sig = "✓✓✓ (p<0.001)"

        print(f"{'CACR vs ' + name:<40} "
              f"{p_graft_one:>15.4f} {p_dwfg_one:>15.4f} {sig:>12}")

    print(f"{'─' * 82}")
    print("  (One-sided paired t-test: H₁ = CACRTransformer > baseline)")
    print("  p < 0.05 = significant, p < 0.01 = highly significant")



# VISUALIZATION
def _plot_cv_results(results, model_names):
    #"""Box plots showing distribution across folds

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Prepare data for box plots
    graft_data = []
    dwfg_data = []
    labels = []

    for name in model_names:
        if len(results[name]['graft']) > 0:
            graft_data.append(results[name]['graft'])
            dwfg_data.append(results[name]['dwfg'])
            labels.append(name.replace('CACRTransformer', 'CACR\n(ours)'))

    # Colors
    colors = ['#95a5a6'] * (len(labels) - 1) + ['#e74c3c']
    colors_dwfg = ['#95a5a6'] * (len(labels) - 1) + ['#3498db']

    # Graft Failure
    bp1 = axes[0].boxplot(graft_data, labels=labels, patch_artist=True,
                           widths=0.6, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='black',
                                        markersize=6))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].set_ylabel('C-index', fontsize=12)
    axes[0].set_title('Graft Failure — 5-Fold CV', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)

    # Add mean ± std annotations
    for i, data in enumerate(graft_data):
        mean = np.mean(data)
        std = np.std(data)
        axes[0].text(i + 1, max(data) + 0.003,
                    f'{mean:.4f}\n±{std:.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # DWFG
    bp2 = axes[1].boxplot(dwfg_data, labels=labels, patch_artist=True,
                           widths=0.6, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='black',
                                        markersize=6))
    for patch, color in zip(bp2['boxes'], colors_dwfg):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_ylabel('C-index', fontsize=12)
    axes[1].set_title('DWFG — 5-Fold CV', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)

    for i, data in enumerate(dwfg_data):
        mean = np.mean(data)
        std = np.std(data)
        axes[1].text(i + 1, max(data) + 0.003,
                    f'{mean:.4f}\n±{std:.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('5-Fold Cross-Validation — C-index Distribution',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('cv_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nPlot saved to cv_comparison.png")
