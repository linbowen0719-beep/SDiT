"""Priority and remote-summary rules used by the manuscript configuration."""
import torch
import torch.nn.functional as F


def detail_residual_score(clean, current):
    """Eq. 3 before region averaging and exp(lambda * tau), shaped [B,H,W].

    A four-neighbor Laplacian uses replicated image boundaries. The 2x2
    latent patch and channel averages are equivalent to the token-feature
    average because both terms use the same fixed-size patch layout.
    """
    if current is None or clean.shape != current.shape or clean.ndim != 4:
        raise ValueError("Matching clean/current tensors of shape [B,C,H,W] are required")
    x = clean.float()
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    laplacian = (p[..., :-2, 1:-1] + p[..., 2:, 1:-1]
                 + p[..., 1:-1, :-2] + p[..., 1:-1, 2:] - 4 * x)
    return (0.5 * laplacian.abs() + 0.5 * (current.float() - x).abs()).mean(dim=1)


def token_coordinates(height, width, device):
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing="ij")
    return torch.stack((y, x), dim=-1).reshape(-1, 2).float()


def spatial_anchor_groups(indices, coordinates, count):
    """Centroid-nearest keys and spatial subgroups for mean-pooled values.

    For m>1, deterministic farthest-point initialization followed by up to
    eight centroid/medoid refinements gives disjoint nonempty subgroups.
    Only coordinates enter this selection; latent content and scores do not.
    """
    count = min(max(int(count), 0), indices.numel())
    if count == 0:
        groups = indices.new_empty((0, 0))
        return indices[:0], groups, groups.bool(), coordinates.new_empty((0,))
    points = coordinates.index_select(0, indices).float()
    selected = [((points - points.mean(0)).square().sum(-1)).argmin()]
    while len(selected) < count:
        distances = torch.cdist(points, points[torch.stack(selected)])
        selected.append(distances.min(dim=1).values.argmax())
    representatives = torch.stack(selected)
    for _ in range(8 if count > 1 else 0):
        assignment = torch.cdist(points, points[representatives]).argmin(dim=1)
        updated = []
        for group in range(count):
            members = torch.where(assignment == group)[0]
            center = points[members].mean(0)
            updated.append(members[(points[members] - center).square().sum(-1).argmin()])
        updated = torch.stack(updated)
        if torch.equal(updated, representatives):
            break
        representatives = updated
    assignment = torch.cdist(points, points[representatives]).argmin(dim=1)
    weights = torch.bincount(assignment, minlength=count).float()
    groups = indices.new_zeros((count, int(weights.max().item())))
    mask = torch.zeros_like(groups, dtype=torch.bool)
    for group in range(count):
        members = indices[assignment == group]
        groups[group, :members.numel()] = members
        mask[group, :members.numel()] = True
    return indices[representatives], groups, mask, weights
