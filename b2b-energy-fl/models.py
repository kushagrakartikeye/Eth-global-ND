import torch
import torch.nn as nn

class FedPersonalizedMLP(nn.Module):
    def __init__(self, in_dim, shared_dim=32, personal_dim=16):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
        )
        self.personal = nn.Linear(shared_dim, personal_dim)
        self.out = nn.Linear(personal_dim, 1)  # binary classification

    def forward(self, x):
        x = self.shared(x)
        x = self.personal(x)
        x = self.out(x)
        return x

def split_global_state(local_sd):
    """Return dict with only shared weights (for aggregation)."""
    return {k: v.cpu() for k, v in local_sd.items() if k.startswith('shared')}

def merge_global_state(local_sd, new_shared):
    """Overwrite local shared weights with aggregated new_shared weights."""
    for k, v in new_shared.items():
        local_sd[k].data.copy_(v)
