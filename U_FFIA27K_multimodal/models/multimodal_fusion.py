import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple


def _make_subspace_head(dim: int, hidden_dim: Optional[int] = None) -> nn.Sequential:
    """Helper to build a 2-layer MLP head with LayerNorm using 100% native PyTorch initialization."""
    h_dim = hidden_dim if hidden_dim is not None else dim // 2
    return nn.Sequential(
        nn.Linear(dim, h_dim),
        nn.GELU(),
        nn.LayerNorm(h_dim),
        nn.Linear(h_dim, 1)
    )


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

    Dynamic Tie-Breaker Intervention (Sigmoid Reparameterization):
      u_tie = exp(-|logit_base| / 2.0)
      gamma_eff = sigmoid(gamma_raw) in (0, 1)
      logit = logit_base + gamma_eff * u_tie * logit_video
      where gamma_raw is a learnable scalar initialized to 0.0 (initial gamma_eff = 0.5).

    Tournament Scoring (Borda count):
      - V_Weak   = P(W > M) + P(W > S)
      - V_Medium = (1 - P(W > M)) + P(M > S)
      - V_Strong = (1 - P(W > S)) + (1 - P(M > S))
      Sum(V_Weak + V_Medium + V_Strong) = 3.0 (Strict invariant)
    """
    def __init__(
        self,
        dim: int = 384,
        temperature: float = 2.0,
        enable_b12: bool = False,
        enable_b23: bool = True,
        enable_b13: bool = False,
        tie_breakers: Optional[Any] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        hidden_dim = dim // 2  # 192 for dim=384

        # Parse tie_breakers configuration robustly
        self.enable_b12, self.enable_b23, self.enable_b13 = _parse_tie_breakers(
            tie_breakers, enable_b12, enable_b23, enable_b13
        )

        # Level 1: Feeding Activity Gate (None vs Feeding) — Unified 192-dim head with LayerNorm
        self.activity_head = _make_subspace_head(dim, hidden_dim)

        # Level 2: 3 Specialized Pairwise Subspace Heads + Video Tie-Breakers
        # B12: Weak vs Medium
        self.head_b12 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b12:
            self.head_b12_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_12 = nn.Parameter(torch.tensor(0.0))  # Raw parameter, sigmoid(0.0) = 0.5
        else:
            self.head_b12_v = None
            self.gamma_12 = None

        # B23: Medium vs Strong
        self.head_b23 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b23:
            self.head_b23_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_23 = nn.Parameter(torch.tensor(0.0))  # Raw parameter, sigmoid(0.0) = 0.5
        else:
            self.head_b23_v = None
            self.gamma_23 = None

        # B13: Weak vs Strong
        self.head_b13 = _make_subspace_head(dim, hidden_dim)
        if self.enable_b13:
            self.head_b13_v = _make_subspace_head(dim, hidden_dim)
            self.gamma_13 = nn.Parameter(torch.tensor(0.0))  # Raw parameter, sigmoid(0.0) = 0.5
        else:
            self.head_b13_v = None
            self.gamma_13 = None

    def forward(
        self,
        f: torch.Tensor,
        f_video: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # 1. Level 1: Feeding Activity Gate
        logit_act = self.activity_head(f).squeeze(-1)       # [B]
        p_feeding = torch.sigmoid(logit_act)                # [B] in (0, 1)
        p_none = torch.clamp(1.0 - p_feeding, min=1e-6)     # [B]

        # 2. Level 2: 3 Pairwise Cross-Boundary Logits & Probabilities
        # B12 Base proposal on f_fused + Video Kinematics Tie-Breaker
        logit_12_base = self.head_b12(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Medium)
        if self.enable_b12 and self.head_b12_v is not None and f_video is not None:
            logit_12_v = self.head_b12_v(f_video).squeeze(-1)
            u_tie_12 = torch.exp(-torch.abs(logit_12_base) / 2.0)
            gamma_12_eff = torch.sigmoid(self.gamma_12)
            logit_12 = logit_12_base + gamma_12_eff * u_tie_12 * logit_12_v
        else:
            logit_12 = logit_12_base
            logit_12_v = logit_12_base
            u_tie_12 = torch.zeros_like(logit_12_base)
            gamma_12_eff = torch.sigmoid(self.gamma_12) if self.gamma_12 is not None else None

        # B23 Base proposal on f_fused + Video Kinematics Tie-Breaker
        logit_23_base = self.head_b23(f).squeeze(-1)        # [B] (Positive -> Medium, Negative -> Strong)
        if self.enable_b23 and self.head_b23_v is not None and f_video is not None:
            logit_23_v = self.head_b23_v(f_video).squeeze(-1)
            u_tie_23 = torch.exp(-torch.abs(logit_23_base) / 2.0)
            gamma_23_eff = torch.sigmoid(self.gamma_23)
            logit_23 = logit_23_base + gamma_23_eff * u_tie_23 * logit_23_v
        else:
            logit_23 = logit_23_base
            logit_23_v = logit_23_base
            u_tie_23 = torch.zeros_like(logit_23_base)
            gamma_23_eff = torch.sigmoid(self.gamma_23) if self.gamma_23 is not None else None

        # B13 Base proposal on f_fused + Video Kinematics Tie-Breaker
        logit_13_base = self.head_b13(f).squeeze(-1)        # [B] (Positive -> Weak, Negative -> Strong)
        if self.enable_b13 and self.head_b13_v is not None and f_video is not None:
            logit_13_v = self.head_b13_v(f_video).squeeze(-1)
            u_tie_13 = torch.exp(-torch.abs(logit_13_base) / 2.0)
            gamma_13_eff = torch.sigmoid(self.gamma_13)
            logit_13 = logit_13_base + gamma_13_eff * u_tie_13 * logit_13_v
        else:
            logit_13 = logit_13_base
            logit_13_v = logit_13_base
            u_tie_13 = torch.zeros_like(logit_13_base)
            gamma_13_eff = torch.sigmoid(self.gamma_13) if self.gamma_13 is not None else None

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
        p_feeding_ranks = F.softmax(v_voting * self.temperature, dim=-1)  # [B, 3]
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
            "gamma_12": gamma_12_eff,
            "gamma_12_raw": self.gamma_12,
            "gamma_12_v": gamma_12_eff,
            "logit_23": logit_23,
            "logit_23_base": logit_23_base,
            "logit_23_v": logit_23_v,
            "u_tie_23": u_tie_23,
            "gamma_23": gamma_23_eff,
            "gamma_23_raw": self.gamma_23,
            "gamma_23_v": gamma_23_eff,
            "logit_13": logit_13,
            "logit_13_base": logit_13_base,
            "logit_13_v": logit_13_v,
            "u_tie_13": u_tie_13,
            "gamma_13": gamma_13_eff,
            "gamma_13_raw": self.gamma_13,
            "gamma_13_v": gamma_13_eff,
            # Pairwise matchup winning probabilities & Borda voting scores
            "p_w_over_m": p_w_over_m,
            "p_m_over_s": p_m_over_s,
            "p_w_over_s": p_w_over_s,
            "v_voting": v_voting
        }


class MultimodalTournamentFusion(nn.Module):
    """
    Multimodal Fusion with Hierarchical Pairwise Cross-Boundary Tournament Engine (Direct Gated, 384-dim).
    1. Direct Cross-Modal Reliability Gating: [w_V, w_A] = softmax(W[f_V || f_A]).
       f_fused = LayerNorm(w_V * f_V + w_A * f_A) -> Directly feeds Tournament Head.
    2. Pairwise Boundary Tournament Decision Head with Unified 192-dim Subspace Heads
       and Configurable Video Kinematics Tie-Breakers on B12, B23, B13.
    """
    def __init__(
        self,
        dim: int = 384,
        classes_num: int = 4,
        enable_b12: bool = False,
        enable_b23: bool = True,
        enable_b13: bool = False,
        tie_breakers: Optional[Any] = None,
        fusion_type: str = "gating",
        head_type: str = "tournament",
        gate_activation: str = "sigmoid",
        **kwargs
    ) -> None:
        super().__init__()
        kwargs.pop("dropout", None)

        self.dim = dim
        self.classes_num = classes_num
        self.fusion_type = str(fusion_type).strip().lower()
        self.head_type = str(head_type).strip().lower()
        self.gate_activation = str(kwargs.get("gate_activation", gate_activation)).strip().lower()

        if self.head_type == "flat":
            if self.fusion_type == "gating":
                if self.gate_activation == "softmax":
                    gate_act_layer = nn.Softmax(dim=-1)
                elif self.gate_activation == "sigmoid":
                    gate_act_layer = nn.Sigmoid()
                else:
                    raise ValueError(f"Unknown gate_activation: '{self.gate_activation}'. Expected 'sigmoid' or 'softmax'.")
                self.gate = nn.Sequential(
                    nn.Linear(dim * 2, 2),
                    gate_act_layer
                )
                self.norm_fused = nn.LayerNorm(dim)
                self.classifier = nn.Linear(dim, classes_num)
            elif self.fusion_type == "sum":
                self.gate = None
                self.norm_fused = nn.LayerNorm(dim)
                self.classifier = nn.Linear(dim, classes_num)
            elif self.fusion_type == "concat":
                self.gate = None
                self.norm_fused = nn.LayerNorm(dim * 2)
                self.classifier = nn.Linear(dim * 2, classes_num)
            else:
                raise ValueError(f"Unknown fusion_type: '{self.fusion_type}'. Expected 'gating', 'sum', or 'concat'.")
            self.tournament_head = None
        else:
            self.enable_b12, self.enable_b23, self.enable_b13 = _parse_tie_breakers(
                tie_breakers, enable_b12, enable_b23, enable_b13
            )
            t_gate_act = str(kwargs.get("gate_activation", self.gate_activation)).strip().lower()
            if t_gate_act == "sigmoid":
                gate_act_layer = nn.Sigmoid()
            else:
                gate_act_layer = nn.Softmax(dim=-1)
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, 2),
                gate_act_layer
            )
            self.norm_fused = nn.LayerNorm(dim)
            self.classifier = None
            self.tournament_head = PairwiseBoundaryTournamentHead(
                dim=dim,
                temperature=2.0,
                enable_b12=self.enable_b12,
                enable_b23=self.enable_b23,
                enable_b13=self.enable_b13,
                tie_breakers=tie_breakers,
                **kwargs
            )

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if self.head_type == "flat":
            if self.fusion_type == "gating":
                combined = torch.cat([f_video, f_audio], dim=-1)
                weights = self.gate(combined)
                w_video = weights[:, 0:1]
                w_audio = weights[:, 1:2]
                f_fused = self.norm_fused(w_video * f_video + w_audio * f_audio)
                logits = self.classifier(f_fused)
                return {
                    "logits": logits,
                    "clipwise_output": logits,
                    "f_fused": f_fused,
                    "gate": w_video,
                    "modality_weights": weights,
                }
            elif self.fusion_type == "sum":
                f_fused = self.norm_fused(f_video + f_audio)
                logits = self.classifier(f_fused)
                return {
                    "logits": logits,
                    "clipwise_output": logits,
                    "f_fused": f_fused,
                }
            elif self.fusion_type == "concat":
                combined = torch.cat([f_video, f_audio], dim=-1)
                f_fused = self.norm_fused(combined)
                logits = self.classifier(f_fused)
                return {
                    "logits": logits,
                    "clipwise_output": logits,
                    "f_fused": f_fused,
                }

        # Step 1: Direct Cross-modal adaptive reliability gating (Tournament Mode)
        combined = torch.cat([f_video, f_audio], dim=-1)  # [B, dim * 2]
        weights = self.gate(combined)                      # [B, 2]
        w_video = weights[:, 0:1]                         # [B, 1]
        w_audio = weights[:, 1:2]                         # [B, 1]
        f_fused = self.norm_fused(w_video * f_video + w_audio * f_audio)  # [B, dim]

        # Step 2: Direct feed into Tournament Head
        out = self.tournament_head(f_fused, f_video=f_video)

        # Step 3: Package metrics
        out["f_fused"] = f_fused
        out["gate"] = w_video
        out["modality_weights"] = weights

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out
