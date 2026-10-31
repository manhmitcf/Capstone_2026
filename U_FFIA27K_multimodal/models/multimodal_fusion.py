import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple, Union, Sequence


def _make_subspace_head(dim: int, hidden_dims: Optional[Union[int, Sequence[int]]] = None) -> nn.Sequential:
    """
    Helper to build a subspace classification head with native PyTorch initialization.
    If hidden_dims is None or empty ([] or ()), builds a single direct projection: Linear(dim -> 1).
    If hidden_dims is provided (e.g. [192] or [256, 128]), builds an MLP: Linear -> GELU -> LayerNorm -> ... -> Linear(last -> 1).
    """
    if hidden_dims is None:
        h_dims = []
    elif isinstance(hidden_dims, int):
        h_dims = [hidden_dims]
    else:
        h_dims = [int(h) for h in hidden_dims]

    if not h_dims:
        return nn.Sequential(nn.Linear(dim, 1))

    layers = []
    in_d = dim
    for h in h_dims:
        layers.extend([
            nn.Linear(in_d, h),
            nn.GELU(),
            nn.LayerNorm(h)
        ])
        in_d = h
    layers.append(nn.Linear(in_d, 1))
    return nn.Sequential(*layers)


def _parse_tie_breakers(
    tie_breakers: Any,
    enable_b12: bool = False,
    enable_b23: bool = True,
    enable_b13: bool = False
) -> Tuple[bool, bool, bool]:
    """
    Robustly parses tie_breakers configuration from Pydantic models (v1/v2),
    dictionaries, or objects with attributes for video tie-breakers ablation.
    """
    if tie_breakers is not None:
        if hasattr(tie_breakers, "model_dump"):
            d = tie_breakers.model_dump()
        elif hasattr(tie_breakers, "dict"):
            d = tie_breakers.dict()
        elif isinstance(tie_breakers, dict):
            d = dict(tie_breakers)
        else:
            d = {}

        def _extract(v, default):
            if isinstance(v, dict):
                if "enable_video" in v:
                    return bool(v["enable_video"])
                if "enable" in v:
                    return bool(v["enable"])
                if "video" in v:
                    return bool(v["video"])
                return default
            if hasattr(v, "enable_video"):
                return bool(getattr(v, "enable_video"))
            if hasattr(v, "enable"):
                return bool(getattr(v, "enable"))
            return bool(v)

        if d:
            def _get_val(k_flat, k_short, def_v):
                if k_flat in d:
                    return _extract(d[k_flat], def_v)
                if k_short in d:
                    return _extract(d[k_short], def_v)
                v_vid = d.get(f"{k_flat}_video", d.get(f"{k_short}_video", None))
                if v_vid is not None:
                    return _extract(v_vid, def_v)
                return def_v

            enable_b12 = _get_val("enable_b12", "b12", enable_b12)
            enable_b23 = _get_val("enable_b23", "b23", enable_b23)
            enable_b13 = _get_val("enable_b13", "b13", enable_b13)
        else:
            def _get_attr(k_flat, k_short, def_v):
                if hasattr(tie_breakers, k_flat):
                    return _extract(getattr(tie_breakers, k_flat), def_v)
                if hasattr(tie_breakers, k_short):
                    return _extract(getattr(tie_breakers, k_short), def_v)
                v_vid = getattr(tie_breakers, f"{k_flat}_video", getattr(tie_breakers, f"{k_short}_video", None))
                if v_vid is not None:
                    return _extract(v_vid, def_v)
                return def_v

            enable_b12 = _get_attr("enable_b12", "b12", enable_b12)
            enable_b23 = _get_attr("enable_b23", "b23", enable_b23)
            enable_b13 = _get_attr("enable_b13", "b13", enable_b13)

    return bool(enable_b12), bool(enable_b23), bool(enable_b13)


def _parse_tournament_config(
    tournament: Any,
    tie_temperature: float = 1.0,
    voting_temperature: float = 1.0,
    head_hidden_dims: Tuple[int, ...] = ()
) -> Tuple[float, float, Tuple[int, ...]]:
    """
    Robustly parses tournament configuration parameters (tie_temperature, voting_temperature, head_hidden_dims).
    """
    if tournament is not None:
        if hasattr(tournament, "model_dump"):
            d = tournament.model_dump()
        elif hasattr(tournament, "dict"):
            d = tournament.dict()
        elif isinstance(tournament, dict):
            d = dict(tournament)
        else:
            d = {}

        if d:
            tie_temperature = float(d.get("tie_temperature", tie_temperature))
            voting_temperature = float(d.get("voting_temperature", voting_temperature))
            h = d.get("head_hidden_dims", head_hidden_dims)
            if isinstance(h, (list, tuple)):
                head_hidden_dims = tuple(int(x) for x in h)
            elif isinstance(h, int):
                head_hidden_dims = (int(h),)
        else:
            if hasattr(tournament, "tie_temperature"):
                tie_temperature = float(getattr(tournament, "tie_temperature"))
            if hasattr(tournament, "voting_temperature"):
                voting_temperature = float(getattr(tournament, "voting_temperature"))
            if hasattr(tournament, "head_hidden_dims"):
                h = getattr(tournament, "head_hidden_dims")
                if isinstance(h, (list, tuple)):
                    head_hidden_dims = tuple(int(x) for x in h)
                elif isinstance(h, int):
                    head_hidden_dims = (int(h),)

    return tie_temperature, voting_temperature, head_hidden_dims


class PairwiseBoundaryTournamentHead(nn.Module):
    """
    Hierarchical Pairwise Cross-Boundary Tournament Head with 3 Configurable Video Kinematics Tie-Breakers (Unified 384-dim).

    Level 1: Feeding Activity Gating
      - Distinguishes None (No feeding, quiet water) from Active Feeding (Weak, Medium, Strong).
      - p_feeding = sigmoid(w_act^T * f_fused) in (0, 1)
      - p_none = 1 - p_feeding

    Level 2: 3-Way Pairwise Cross-Boundary Tournament with 3 Video Kinematics Tie-Breakers
      - B12: Weak <-> Medium    -> Base Head on f_fused + Video Kinematics Referee on f_video
      - B23: Medium <-> Strong  -> Base Head on f_fused + Video Kinematics Referee on f_video
      - B13: Weak <-> Strong    -> Base Head on f_fused + Video Kinematics Referee on f_video

    Dynamic Tie-Breaker Intervention:
      u_tie = exp(-|logit_base| / tie_temperature)
      logit = logit_base + u_tie * logit_video

    Tournament Scoring (Borda count):
      - V_Weak   = P(W > M) + P(W > S)
      - V_Medium = (1 - P(W > M)) + P(M > S)
      - V_Strong = (1 - P(W > S)) + (1 - P(M > S))
      Sum(V_Weak + V_Medium + V_Strong) = 3.0 (Strict invariant)
    """
    def __init__(
        self,
        dim: int = 384,
        temperature: Optional[float] = None,
        tie_temperature: float = 1.0,
        voting_temperature: float = 1.0,
        head_hidden_dims: Tuple[int, ...] = (),
        enable_b12: bool = False,
        enable_b23: bool = True,
        enable_b13: bool = False,
        tie_breakers: Optional[Any] = None,
        tournament: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim

        # Parse tournament configurations
        t_tie, t_vote, h_dims = _parse_tournament_config(
            tournament,
            tie_temperature=kwargs.get("tie_temp", tie_temperature),
            voting_temperature=temperature if temperature is not None else voting_temperature,
            head_hidden_dims=head_hidden_dims
        )
        self.tie_temperature = t_tie
        self.voting_temperature = t_vote
        self.temperature = t_vote  # Backward-compatible alias
        self.head_hidden_dims = h_dims

        # Parse tie_breakers configuration robustly
        self.enable_b12, self.enable_b23, self.enable_b13 = _parse_tie_breakers(
            tie_breakers, enable_b12, enable_b23, enable_b13
        )

        # Level 1: Feeding Activity Gate (None vs Feeding) — Default Linear(dim -> 1)
        self.activity_head = _make_subspace_head(dim, self.head_hidden_dims)

        # Level 2: 3 Specialized Pairwise Subspace Heads + Video Tie-Breakers
        # B12: Weak vs Medium
        self.head_b12 = _make_subspace_head(dim, self.head_hidden_dims)
        self.head_b12_v = _make_subspace_head(dim, self.head_hidden_dims) if self.enable_b12 else None

        # B23: Medium vs Strong
        self.head_b23 = _make_subspace_head(dim, self.head_hidden_dims)
        self.head_b23_v = _make_subspace_head(dim, self.head_hidden_dims) if self.enable_b23 else None

        # B13: Weak vs Strong
        self.head_b13 = _make_subspace_head(dim, self.head_hidden_dims)
        self.head_b13_v = _make_subspace_head(dim, self.head_hidden_dims) if self.enable_b13 else None

    def forward(
        self,
        f: torch.Tensor,
        f_video: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if f.ndim == 1:
            f = f.unsqueeze(0)
            if f_video is not None and f_video.ndim == 1:
                f_video = f_video.unsqueeze(0)

        # 1. Level 1: Feeding Activity Gate
        logit_act = self.activity_head(f).squeeze(-1)       # [B]
        p_feeding = torch.sigmoid(logit_act)                # [B] in (0, 1)
        p_none = torch.clamp(1.0 - p_feeding, min=1e-6)     # [B]

        # 2. Level 2: 3 Pairwise Cross-Boundary Logits & Probabilities
        # B12 Base proposal on f_fused + Video Kinematics Tie-Breaker
        logit_12_base = self.head_b12(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Medium)
        if self.enable_b12 and self.head_b12_v is not None and f_video is not None:
            logit_12_v = self.head_b12_v(f_video).squeeze(-1)
            u_tie_12 = torch.exp(-torch.abs(logit_12_base) / self.tie_temperature)
            logit_12 = logit_12_base + u_tie_12 * logit_12_v
        else:
            logit_12 = logit_12_base
            logit_12_v = logit_12_base
            u_tie_12 = torch.zeros_like(logit_12_base)

        # B23 Base proposal on f_fused + Video Kinematics Tie-Breaker
        logit_23_base = self.head_b23(f).squeeze(-1)        # [B] (Positive -> Medium, Negative -> Strong)
        if self.enable_b23 and self.head_b23_v is not None and f_video is not None:
            logit_23_v = self.head_b23_v(f_video).squeeze(-1)
            u_tie_23 = torch.exp(-torch.abs(logit_23_base) / self.tie_temperature)
            logit_23 = logit_23_base + u_tie_23 * logit_23_v
        else:
            logit_23 = logit_23_base
            logit_23_v = logit_23_base
            u_tie_23 = torch.zeros_like(logit_23_base)

        # B13 Base proposal on f_fused + Video Kinematics Tie-Breaker
        logit_13_base = self.head_b13(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Strong)
        if self.enable_b13 and self.head_b13_v is not None and f_video is not None:
            logit_13_v = self.head_b13_v(f_video).squeeze(-1)
            u_tie_13 = torch.exp(-torch.abs(logit_13_base) / self.tie_temperature)
            logit_13 = logit_13_base + u_tie_13 * logit_13_v
        else:
            logit_13 = logit_13_base
            logit_13_v = logit_13_base
            u_tie_13 = torch.zeros_like(logit_13_base)

        # Head-to-head pairwise winning probabilities
        p_w_over_m = torch.sigmoid(logit_12)               # P(W > M)
        p_m_over_w = 1.0 - p_w_over_m                      # P(M > W)

        p_m_over_s = torch.sigmoid(logit_23)               # P(M > S)
        p_s_over_m = 1.0 - p_m_over_s                      # P(S > M)

        p_w_over_s = torch.sigmoid(logit_13)               # P(W > S)
        p_s_over_w = 1.0 - p_w_over_s                      # P(S > W)

        # 3. Tournament Borda Count Voting
        # Weak wins if it beats Medium AND beats Strong
        v_weak = p_w_over_m + p_w_over_s                   # [B] in [0, 2]
        # Medium wins if it beats Weak AND beats Strong
        v_medium = p_m_over_w + p_m_over_s                 # [B] in [0, 2]
        # Strong wins if it beats Weak AND beats Medium
        v_strong = p_s_over_w + p_s_over_m                 # [B] in [0, 2]

        v_voting = torch.stack([v_weak, v_medium, v_strong], dim=-1)  # [B, 3]

        # Softmax over normalized tournament votes (Rank order: Weak=1, Med=2, Strong=3)
        p_feeding_ranks = F.softmax(v_voting * self.voting_temperature, dim=-1)  # [B, 3]
        p_weak_given_feed = p_feeding_ranks[:, 0]
        p_med_given_feed = p_feeding_ranks[:, 1]
        p_strong_given_feed = p_feeding_ranks[:, 2]

        # 4. Final Hierarchical Combination
        p_final_none = p_none
        p_final_weak = p_feeding * p_weak_given_feed
        p_final_medium = p_feeding * p_med_given_feed
        p_final_strong = p_feeding * p_strong_given_feed

        # Map to raw dataset class indexing: [0: None, 1: Strong, 2: Medium, 3: Weak]
        p_raw = torch.stack([p_final_none, p_final_strong, p_final_medium, p_final_weak], dim=-1)
        p_raw = p_raw / torch.sum(p_raw, dim=-1, keepdim=True)
        logits_raw = torch.log(torch.clamp(p_raw, min=1e-7))

        # Expected physical intensity on 0..3 ordinal scale
        # None=0, Weak=1, Medium=2, Strong=3
        expected_intensity = (
            p_final_none * 0.0 +
            p_final_weak * 1.0 +
            p_final_medium * 2.0 +
            p_final_strong * 3.0
        ).unsqueeze(-1)

        return {
            "clipwise_output": logits_raw,
            "logits": logits_raw,
            "probabilities": p_raw,
            "expected_intensity": expected_intensity,
            "intensity_score": expected_intensity,
            # Pairwise logits & probabilities for specialized loss calculation
            "logit_act": logit_act,
            "p_feeding": p_feeding,
            "logit_12": logit_12,
            "logit_12_base": logit_12_base,
            "logit_12_v": logit_12_v,
            "u_tie_12": u_tie_12,
            "logit_23": logit_23,
            "logit_23_base": logit_23_base,
            "logit_23_v": logit_23_v,
            "u_tie_23": u_tie_23,
            "logit_13": logit_13,
            "logit_13_base": logit_13_base,
            "logit_13_v": logit_13_v,
            "u_tie_13": u_tie_13,
            # Pairwise matchup winning probabilities & Borda voting scores
            "p_w_over_m": p_w_over_m,
            "p_m_over_s": p_m_over_s,
            "p_w_over_s": p_w_over_s,
            "v_voting": v_voting
        }


class MultimodalTournamentFusion(nn.Module):
    """
    Multimodal Fusion with Hierarchical Pairwise Cross-Boundary Tournament Engine (Direct Sum + LayerNorm, 384-dim).
    1. Direct Sum + LayerNorm: f_fused = LayerNorm(f_V + f_A) -> Directly feeds Tournament Head.
    2. Pairwise Boundary Tournament Decision Head with Configurable Subspace Heads (default Linear(dim -> 1))
       and Configurable Video Kinematics Tie-Breakers on B12, B23, B13.
    """
    def __init__(
        self,
        dim: int = 384,
        enable_b12: bool = False,
        enable_b23: bool = True,
        enable_b13: bool = False,
        tie_breakers: Optional[Any] = None,
        tournament: Optional[Any] = None,
        tie_temperature: float = 1.0,
        voting_temperature: float = 1.0,
        head_hidden_dims: Tuple[int, ...] = (),
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.enable_b12, self.enable_b23, self.enable_b13 = _parse_tie_breakers(
            tie_breakers, enable_b12, enable_b23, enable_b13
        )

        # 1. Direct Sum + LayerNorm Fusion (No learnable gating parameters)
        self.norm_fused = nn.LayerNorm(dim)

        # 2. Pairwise Boundary Tournament Decision Head (Directly fed from f_fused)
        self.tournament_head = PairwiseBoundaryTournamentHead(
            dim=dim,
            enable_b12=self.enable_b12,
            enable_b23=self.enable_b23,
            enable_b13=self.enable_b13,
            tie_breakers=tie_breakers,
            tournament=tournament,
            tie_temperature=tie_temperature,
            voting_temperature=voting_temperature,
            head_hidden_dims=head_hidden_dims,
            **kwargs
        )

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if f_video.ndim == 1:
            f_video = f_video.unsqueeze(0)
        if f_audio.ndim == 1:
            f_audio = f_audio.unsqueeze(0)

        # Step 1: Direct Sum + LayerNorm Fusion
        f_fused = self.norm_fused(f_video + f_audio)  # [B, dim]

        # Step 2: Direct feed into Tournament Head
        out = self.tournament_head(f_fused, f_video=f_video)

        # Step 3: Package metrics
        out["f_fused"] = f_fused

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out

