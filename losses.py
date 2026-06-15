
import torch
import torch.nn as nn
import numpy as np


class CompetingRisksLoss(nn.Module):
    """
    Combined loss for competing risks survival analysis
    
    L_total = L_nll + alpha * L_rank
    
    L_nll:  Negative log-likelihood that properly handles censoring.
            For uncensored: -log P(T=t_i, K=k_i | x_i)
            For censored:   -log S(t_i | x_i) = -log [1 - sum_k CIF_k(t_i)]
    
    L_rank: Cause-specific ranking loss that encourages the model to
            assign higher cumulative incidence to patients who experienced
            the event earlier. Uses a sigmoid approximation controlled by sigma.
    
    Args:
        alpha: Weight for ranking loss (0 = pure likelihood, 1 = balanced)
        sigma: Smoothness parameter for ranking loss sigmoid (smaller = sharper)
        num_risks: Number of competing events
    """
    def __init__(self, alpha=0.2, sigma=0.1, num_risks=2):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.num_risks = num_risks
    
    def forward(self, pmf, cif, surv, durations, events):
        
        batch_size = pmf.size(0)
        
        #  Negative Log-Likelihood Loss 
        nll_loss = self._nll_loss(pmf, surv, durations, events)
        
        # Cause-Specific Ranking Loss
        if self.alpha > 0:
            rank_loss = self._ranking_loss(cif, durations, events)
        else:
            rank_loss = torch.tensor(0.0, device=pmf.device)
        
        # Combined 
        total_loss = nll_loss + self.alpha * rank_loss
        
        return total_loss, nll_loss.detach(), rank_loss.detach()
    
    def _nll_loss(self, pmf, surv, durations, events, label_smoothing=0.05):
        batch_size = pmf.size(0)
        durations = durations.long()
        events = events.long()
        
        uncensored = events > 0
        censored = events == 0
        
        eps = 1e-7
        loss = torch.zeros(batch_size, device=pmf.device)
        
        if uncensored.any():
            unc_idx = uncensored.nonzero(as_tuple=True)[0]
            unc_events = events[unc_idx] - 1
            unc_times = durations[unc_idx].clamp(0, pmf.size(2) - 1)
            
            unc_probs = pmf[unc_idx, unc_events, unc_times]
            # Label smoothing: blend target with uniform distribution
            smooth_target = (1 - label_smoothing) * 1.0 + label_smoothing/(self.num_risks * pmf.size(2))
            loss[unc_idx] = -smooth_target * torch.log(unc_probs + eps)
        
        if censored.any():
            cen_idx = censored.nonzero(as_tuple=True)[0]
            cen_times = durations[cen_idx].clamp(0, surv.size(1) - 1)
            cen_surv = surv[cen_idx, cen_times]
            loss[cen_idx] = -torch.log(cen_surv.clamp(min=eps))
        
        return loss.mean()
    
    def _ranking_loss(self, cif, durations, events):
        
        batch_size = cif.size(0)
        durations = durations.long()
        events = events.long()
        
        total_rank_loss = torch.tensor(0.0, device=cif.device)
        n_pairs = 0
        
        for k in range(self.num_risks):
            cause_k = k + 1  # event indicator (1 or 2)
            
            # Patients who experienced event k
            event_mask = events == cause_k
            if event_mask.sum() < 2:
                continue
            
            event_idx = event_mask.nonzero(as_tuple=True)[0]
            event_times = durations[event_idx]
            
            # Sample pairs efficiently (max 256 pairs per cause per batch)
            max_pairs = min(256, len(event_idx) * (batch_size - len(event_idx)))
            if max_pairs == 0:
                continue
            
            # For each event-k patient, compare with patients who survived longer
            for i_local in range(min(len(event_idx), 32)):
                i = event_idx[i_local]
                t_i = durations[i].clamp(0, cif.size(2) - 1)
                
                # Find patients j where t_j > t_i (still at risk at t_i)
                at_risk = durations > durations[i]
                if not at_risk.any():
                    continue
                
                j_indices = at_risk.nonzero(as_tuple=True)[0]
                
                # Sample subset for efficiency
                if len(j_indices) > 16:
                    perm = torch.randperm(len(j_indices), device=cif.device)[:16]
                    j_indices = j_indices[perm]
                
                # CIF_k at time t_i for patient i vs patients j
                cif_i = cif[i, k, t_i]       # scalar — high risk patient
                cif_j = cif[j_indices, k, t_i]  # (n_j,) — lower risk patients
                
                # Ranking: we want cif_i > cif_j → penalize when cif_j >= cif_i
                rank_diff = (cif_j - cif_i) / self.sigma
                pair_loss = torch.sigmoid(rank_diff)
                
                total_rank_loss = total_rank_loss + pair_loss.sum()
                n_pairs += len(j_indices)
        
        if n_pairs > 0:
            total_rank_loss = total_rank_loss / n_pairs
        
        return total_rank_loss


class CIndexMetric:
    
    @staticmethod
    def compute(cif, durations, events, cause=1, time_idx=None):
        
        cif = cif.detach().cpu().numpy() if torch.is_tensor(cif) else cif
        durations = durations.detach().cpu().numpy() if torch.is_tensor(durations) else durations
        events = events.detach().cpu().numpy() if torch.is_tensor(events) else events
        
        cause_idx = cause - 1  # 0-indexed
        n = len(durations)
        concordant = 0
        discordant = 0
        tied = 0
        
        # Find patients who experienced this cause
        event_mask = events == cause
        event_indices = np.where(event_mask)[0]
        
        for i in event_indices:
            t_i = int(min(durations[i], cif.shape[2] - 1))
            
            # Compare with patients who survived past t_i
            for j in range(n):
                if i == j:
                    continue
                if durations[j] <= durations[i] and events[j] != 0:
                    # j had an event at or before t_i — not a valid pair
                    # unless j also had the same event, in which case skip
                    continue
                if durations[j] <= durations[i]:
                    continue
                
                # Valid pair: i had event k at t_i, j survived past t_i
                risk_i = cif[i, cause_idx, t_i]
                risk_j = cif[j, cause_idx, t_i]
                
                if risk_i > risk_j:
                    concordant += 1
                elif risk_i < risk_j:
                    discordant += 1
                else:
                    tied += 1
        
        total = concordant + discordant + tied
        if total == 0:
            return 0.5
        
        return (concordant + 0.5 * tied) / total


class IntegratedBrierScore:
    
    @staticmethod
    def compute(cif, durations, events, cause=1, time_grid=None):
        
        cif_np = cif.detach().cpu().numpy() if torch.is_tensor(cif) else cif
        dur = durations.detach().cpu().numpy() if torch.is_tensor(durations) else durations
        evt = events.detach().cpu().numpy() if torch.is_tensor(events) else events
        
        cause_idx = cause - 1
        n_times = cif_np.shape[2]
        n_samples = len(dur)
        
        brier_scores = []
        for t in range(n_times):
            # Actual outcome: did event k happen by time t?
            actual = ((dur <= t) & (evt == cause)).astype(float)
            predicted = cif_np[:, cause_idx, t]
            
            bs = np.mean((predicted - actual) ** 2)
            brier_scores.append(bs)
        
        # Integrate using trapezoidal rule
        ibs = np.trapz(brier_scores) / n_times
        return ibs
