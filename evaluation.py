"""
Comprehensive Evaluation Module for CACRT
Evaluates competing risks survival predictions using:
1. Cause-specific C-index (discrimination)
2. Time-dependent AUC at 1, 3, 5, 10 years (clinical timepoints)
3. Integrated Brier Score (calibration + discrimination)
4. Calibration plots (predicted vs observed)
5. Cumulative incidence curves (population-level)

    pip install scikit-survival lifelines
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score


#CAUSE-SPECIFIC C-INDEX (improved version with full test set

def compute_cindex(cif, durations, events, cause=1):
    
    #Cause-specific C-index using ALL valid pairs
    #Consistent with training evaluation
    
    if torch.is_tensor(cif):
        cif = cif.detach().cpu().numpy()
    if torch.is_tensor(durations):
        durations = durations.detach().cpu().numpy()
    if torch.is_tensor(events):
        events = events.detach().cpu().numpy()
    
    cause_idx = cause - 1
    n = len(durations)
    
    event_mask = events == cause
    event_indices = np.where(event_mask)[0]
    
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in event_indices:
        t_i = int(min(durations[i], cif.shape[2] - 1))
        
        # ALL patients still at risk at time t_i
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
        return 0.5, 0, 0, 0
    
    c_index = (concordant + 0.5 * tied) / total
    return c_index, concordant, discordant, tied


# TIME-DEPENDENT AUC AT CLINICAL TIMEPOINTS

def compute_td_auc(cif, durations, events, time_grid, cause=1,
                   eval_times_years=[1, 3, 5, 10]):
    """
    Time-dependent AUC at specific clinical timepoints
    At each timepoint t:
    - Positive class: patients who experienced event k by time t
    - Negative class: patients who survived beyond time t
    - Score: predicted CIF_k(t)
    
        cif:             (n, num_risks, num_time_bins)
        durations:       (n,) — discretized times
        events:          (n,) — event indicators
        time_grid:       array of time bin edges
        cause:           which cause
        eval_times_years: list of years to evaluate at
    
        dict: {year: auc_value}
    """
    if torch.is_tensor(cif):
        cif = cif.detach().cpu().numpy()
    if torch.is_tensor(durations):
        durations = durations.detach().cpu().numpy()
    if torch.is_tensor(events):
        events = events.detach().cpu().numpy()
    
    cause_idx = cause - 1
    results = {}
    
    for year in eval_times_years:
        # Find the time bin closest to this year
        t_idx = np.argmin(np.abs(time_grid - year))
        
        # Binary labels: did event k happen by time t?
        # Positives: event k occurred AND duration <= t
        positives = (events == cause) & (durations <= t_idx)
        # Negatives: survived beyond t (no event of any kind by t)
        negatives = durations > t_idx
        
        # Skip if no positives or negatives
        if positives.sum() == 0 or negatives.sum() == 0:
            results[year] = None
            continue
        
        # Create binary classification problem
        mask = positives | negatives
        y_true = positives[mask].astype(int)
        y_score = cif[mask, cause_idx, t_idx]
        
        try:
            auc = roc_auc_score(y_true, y_score)
            results[year] = auc
        except ValueError:
            results[year] = None
    
    return results


# BRIER SCORE AND INTEGRATED BRIER SCORE
def compute_brier_score(cif, durations, events, time_grid, cause=1,
                        eval_times_years=None):
    """
    Cause-specific Brier Score at each timepoint
    """
    if torch.is_tensor(cif):
        cif = cif.detach().cpu().numpy()
    if torch.is_tensor(durations):
        durations = durations.detach().cpu().numpy()
    if torch.is_tensor(events):
        events = events.detach().cpu().numpy()
    
    cause_idx = cause - 1
    n_times = cif.shape[2]
    n_samples = len(durations)
    
    brier_scores = np.zeros(n_times)
    
    for t in range(n_times):
        # Actual- did event k happen by time t?
        actual = ((durations <= t) & (events == cause)).astype(float)
        predicted = cif[:, cause_idx, t]
        
        brier_scores[t] = np.mean((predicted - actual) ** 2)
    
    # Integrated Brier Score ;trapezoidal integration
    ibs = np.trapz(brier_scores, time_grid[:n_times]) / (time_grid[-1] - time_grid[0])
    
    # BS at specific timepoints
    bs_at_times = {}
    if eval_times_years is not None:
        for year in eval_times_years:
            t_idx = np.argmin(np.abs(time_grid - year))
            bs_at_times[year] = brier_scores[t_idx]
    
    return brier_scores, ibs, bs_at_times



# CALIBRATION ANALYSIS
def compute_calibration(cif, durations, events, time_grid, cause=1,
                        eval_year=5, n_groups=10):
    """
    Calibration: do predicted probabilities match observed outcomes?
    Groups patients into deciles by predicted CIF at a given timepoint,
    then compares predicted vs observed event rates within each group
    Perfect calibration = points on the 45-degree diagonal.
    
        predicted_means: mean predicted CIF per decile
        observed_means:  observed event rate per decile
        group_sizes:     number of patients per decile
    """
    if torch.is_tensor(cif):
        cif = cif.detach().cpu().numpy()
    if torch.is_tensor(durations):
        durations = durations.detach().cpu().numpy()
    if torch.is_tensor(events):
        events = events.detach().cpu().numpy()
    
    cause_idx = cause - 1
    t_idx = np.argmin(np.abs(time_grid - eval_year))
    
    # Predicted CIF at eval_year for this cause
    predicted = cif[:, cause_idx, t_idx]
    
    # Observed: did event k happen by eval_year?
    observed = ((durations <= t_idx) & (events == cause)).astype(float)
    
    # Only include patients who were observed long enough
    # (either had an event by t or were followed past t)
    valid = (durations > t_idx) | (events > 0)
    predicted = predicted[valid]
    observed = observed[valid]
    
    # Group into deciles
    try:
        quantiles = np.percentile(predicted, np.linspace(0, 100, n_groups + 1))
        quantiles = np.unique(quantiles)
        groups = np.digitize(predicted, quantiles[1:-1])
    except Exception:
        groups = np.repeat(np.arange(n_groups), len(predicted) // n_groups + 1)[:len(predicted)]
    
    predicted_means = []
    observed_means = []
    group_sizes = []
    
    for g in range(len(np.unique(groups))):
        mask = groups == g
        if mask.sum() > 0:
            predicted_means.append(predicted[mask].mean())
            observed_means.append(observed[mask].mean())
            group_sizes.append(mask.sum())
    
    return np.array(predicted_means), np.array(observed_means), np.array(group_sizes)



# POPULATION-LEVEL CUMULATIVE INCIDENCE CURVES
def compute_population_cif(cif, time_grid):
    
    #Average CIF across all patients — shows population-level
    #cumulative incidence for each competing risk.
    if torch.is_tensor(cif):
        cif = cif.detach().cpu().numpy()
    
    mean_cif = cif.mean(axis=0)  # (num_risks, num_time_bins)
    return mean_cif



# FULL EVALUATION PIPELINE
@torch.no_grad()
def full_evaluation(model, data, device, eval_times_years=[1, 3, 5, 10]):
    
    model.eval()
    time_grid = data['time_grid']
    
    # Get predictions on test set 
    donor = data['test']['donor'].to(device)
    recip = data['test']['recip'].to(device)
    match = data['test']['match'].to(device)
    durations = data['test']['durations'].numpy()
    events = data['test']['events'].numpy()
    
    # Batch prediction for memory efficiency
    batch_size = 2048
    all_cif = []
    all_surv = []
    n = len(donor)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pmf, cif, surv = model(
            donor[start:end], recip[start:end], match[start:end]
        )
        all_cif.append(cif.cpu().numpy())
        all_surv.append(surv.cpu().numpy())
    
    cif = np.concatenate(all_cif, axis=0)
    surv = np.concatenate(all_surv, axis=0)
    
    # Compute all metrics
    results = {'cause_names': {1: 'Graft Failure', 2: 'DWFG'}}
    
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION — Test Set")
    print("=" * 70)
    
    for cause in [1, 2]:
        cause_name = results['cause_names'][cause]
        print(f"\n{'─' * 70}")
        print(f"  CAUSE {cause}: {cause_name}")
        print(f"{'─' * 70}")
        
        # C-index
        c_idx, n_conc, n_disc, n_tied = compute_cindex(
            cif, durations, events, cause=cause
        )
        results[f'c_index_cause{cause}'] = c_idx
        print(f"\n  C-index:  {c_idx:.4f}")
        print(f"    Concordant: {n_conc:,} | Discordant: {n_disc:,} | Tied: {n_tied:,}")
        
        # Time-dependent AUC
        td_auc = compute_td_auc(
            cif, durations, events, time_grid,
            cause=cause, eval_times_years=eval_times_years
        )
        results[f'td_auc_cause{cause}'] = td_auc
        print(f"\n  Time-dependent AUC:")
        for year, auc in td_auc.items():
            if auc is not None:
                print(f"    {year:2d}-year:  {auc:.4f}")
            else:
                print(f"    {year:2d}-year:  N/A (insufficient events)")
        
        # Brier Score
        bs_curve, ibs, bs_at_times = compute_brier_score(
            cif, durations, events, time_grid,
            cause=cause, eval_times_years=eval_times_years
        )
        results[f'ibs_cause{cause}'] = ibs
        results[f'brier_curve_cause{cause}'] = bs_curve
        results[f'brier_at_times_cause{cause}'] = bs_at_times
        print(f"\n  Integrated Brier Score:  {ibs:.4f}")
        print(f"  Brier Score at timepoints:")
        for year, bs in bs_at_times.items():
            print(f"    {year:2d}-year:  {bs:.4f}")
        
        # Calibration
        for year in eval_times_years:
            pred, obs, sizes = compute_calibration(
                cif, durations, events, time_grid,
                cause=cause, eval_year=year
            )
            results[f'calibration_cause{cause}_{year}yr'] = {
                'predicted': pred, 'observed': obs, 'sizes': sizes
            }
    
    # Population CIF
    results['population_cif'] = compute_population_cif(cif, time_grid)
    results['time_grid'] = time_grid
    results['cif'] = cif
    results['surv'] = surv
    results['durations'] = durations
    results['events'] = events
    
    # Generate plots
    print(f"\n{'=' * 70}")
    print("Generating evaluation plots...")
    print("=" * 70)
    
    _plot_evaluation(results, eval_times_years)
    
    return results



# PLOTS

def _plot_evaluation(results, eval_times_years):
   # """Generate a comprehensive evaluation figure
    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)
    
    time_grid = results['time_grid']
    
    #  Plot 1: Time-dependent AUC bar chart 
    ax1 = fig.add_subplot(gs[0, 0])
    years = eval_times_years
    auc_graft = [results['td_auc_cause1'].get(y) for y in years]
    auc_dwfg = [results['td_auc_cause2'].get(y) for y in years]
    
    x = np.arange(len(years))
    width = 0.35
    bars1 = ax1.bar(x - width/2, [a if a else 0 for a in auc_graft],
                    width, label='Graft Failure', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, [a if a else 0 for a in auc_dwfg],
                    width, label='DWFG', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Timepoint (years)')
    ax1.set_ylabel('AUC')
    ax1.set_title('Time-Dependent AUC')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{y}yr' for y in years])
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(0.4, 0.85)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # Add value labels on bars
    for bar in bars1:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    
    # Plot 2: Brier Score curves 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_grid, results['brier_curve_cause1'],
             color='#e74c3c', linewidth=2, label='Graft Failure')
    ax2.plot(time_grid, results['brier_curve_cause2'],
             color='#3498db', linewidth=2, label='DWFG')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Brier Score')
    ax2.set_title(f'Brier Score Over Time\n'
                  f'IBS: Graft={results["ibs_cause1"]:.4f}, '
                  f'DWFG={results["ibs_cause2"]:.4f}')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Population-level CIF 
    ax3 = fig.add_subplot(gs[0, 2])
    pop_cif = results['population_cif']
    ax3.fill_between(time_grid, 0, pop_cif[0], alpha=0.3, color='#e74c3c')
    ax3.fill_between(time_grid, pop_cif[0], pop_cif[0] + pop_cif[1],
                     alpha=0.3, color='#3498db')
    ax3.plot(time_grid, pop_cif[0], color='#e74c3c', linewidth=2,
             label='Graft Failure')
    ax3.plot(time_grid, pop_cif[0] + pop_cif[1], color='#3498db', linewidth=2,
             label='DWFG (stacked)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Cumulative Incidence')
    ax3.set_title('Population-Level CIF\n(stacked competing risks)')
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Plots 4-5: Calibration plots -Graft Failure
    for idx, year in enumerate([y for y in eval_times_years if y <= 10][:3]):
        ax = fig.add_subplot(gs[1, idx])
        cal = results.get(f'calibration_cause1_{year}yr')
        if cal is not None:
            pred, obs, sizes = cal['predicted'], cal['observed'], cal['sizes']
            # Scatter with size proportional to group size
            scatter = ax.scatter(pred, obs, s=sizes/5, c='#e74c3c',
                               alpha=0.7, edgecolors='darkred', linewidth=1)
            ax.plot([0, max(max(pred), max(obs)) * 1.1],
                    [0, max(max(pred), max(obs)) * 1.1],
                    'k--', alpha=0.5, label='Perfect calibration')
            ax.set_xlabel('Predicted CIF')
            ax.set_ylabel('Observed proportion')
            ax.set_title(f'Calibration — Graft Failure\n{year}-year')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Plots 6-8: Calibration plots (DWFG)
    for idx, year in enumerate([y for y in eval_times_years if y <= 10][:3]):
        ax = fig.add_subplot(gs[2, idx])
        cal = results.get(f'calibration_cause2_{year}yr')
        if cal is not None:
            pred, obs, sizes = cal['predicted'], cal['observed'], cal['sizes']
            scatter = ax.scatter(pred, obs, s=sizes/5, c='#3498db',
                               alpha=0.7, edgecolors='darkblue', linewidth=1)
            ax.plot([0, max(max(pred), max(obs)) * 1.1],
                    [0, max(max(pred), max(obs)) * 1.1],
                    'k--', alpha=0.5, label='Perfect calibration')
            ax.set_xlabel('Predicted CIF')
            ax.set_ylabel('Observed proportion')
            ax.set_title(f'Calibration — DWFG\n{year}-year')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.savefig('comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary table 
    _print_summary_table(results, eval_times_years)


def _print_summary_table(results, eval_times_years):
    
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE)")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':<30} {'Graft Failure':>15} {'DWFG':>15}")
    print(f"{'─' * 60}")
    
    # C-index
    print(f"{'C-index':<30} "
          f"{results['c_index_cause1']:>15.4f} "
          f"{results['c_index_cause2']:>15.4f}")
    
    # IBS
    print(f"{'Integrated Brier Score':<30} "
          f"{results['ibs_cause1']:>15.4f} "
          f"{results['ibs_cause2']:>15.4f}")
    
    # td-AUC at each timepoint
    for year in eval_times_years:
        auc1 = results['td_auc_cause1'].get(year)
        auc2 = results['td_auc_cause2'].get(year)
        print(f"{'td-AUC @ ' + str(year) + ' year':<30} "
              f"{auc1 if auc1 else 'N/A':>15} "
              f"{auc2 if auc2 else 'N/A':>15}")
    
    # BS at each timepoint
    for year in eval_times_years:
        bs1 = results['brier_at_times_cause1'].get(year)
        bs2 = results['brier_at_times_cause2'].get(year)
        if bs1 is not None:
            print(f"{'Brier Score @ ' + str(year) + ' year':<30} "
                  f"{bs1:>15.4f} "
                  f"{bs2:>15.4f}")
    
    print(f"{'─' * 60}")
