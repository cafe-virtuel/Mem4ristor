"""
Mem4ristor Configuration via Dataclasses.

Replaces nested dictionaries with typed, documented, IDE-friendly dataclasses.
Fully backward-compatible: call cfg.to_dict() to get the legacy dict format
expected by Mem4ristorV3.

Usage:
    from mem4ristor.config import Mem4Config

    # Use defaults
    cfg = Mem4Config()

    # Override specific fields
    cfg = Mem4Config(
        coupling=CouplingConfig(D=0.20, heretic_ratio=0.20),
        noise=NoiseConfig(sigma_v=0.10),
    )

    # Pass to engine
    model = Mem4ristorV3(config=cfg.to_dict())

    # Or load from YAML
    cfg = Mem4Config.from_yaml("config.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml


@dataclass
class DynamicsConfig:
    """FitzHugh-Nagumo dynamics parameters."""
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08
    alpha: float = 0.15
    v_cubic_divisor: float = 5.0
    dt: float = 0.05
    lambda_learn: float = 0.05
    tau_plasticity: float = 1000.0
    w_saturation: float = 2.0


@dataclass
class CouplingConfig:
    """Coupling and heretic configuration."""
    D: float = 0.15
    heretic_ratio: float = 0.15
    uniform_placement: bool = True


@dataclass
class DoubtConfig:
    """Constitutional doubt dynamics."""
    epsilon_u: float = 0.02
    k_u: float = 1.0
    sigma_baseline: float = 0.05
    u_clamp: List[float] = field(default_factory=lambda: [0.0, 1.0])
    tau_u: float = 1.0
    alpha_surprise: float = 2.0
    surprise_cap: float = 5.0


@dataclass
class NoiseConfig:
    """Noise parameters (hardware-realistic)."""
    sigma_v: float = 0.05
    use_rtn: bool = False
    rtn_amplitude: float = 0.1
    rtn_p_flip: float = 0.01


@dataclass
class Mem4Config:
    """
    Complete Mem4ristor v3.1.0 configuration.

    All parameters are documented and typed. Default values match
    the reference configuration from the preprint.
    """
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    doubt: DoubtConfig = field(default_factory=DoubtConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    def to_dict(self) -> dict:
        """Convert to the nested dict format expected by Mem4ristorV3."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Mem4Config":
        """Create from a nested dict (e.g., loaded from YAML)."""
        return cls(
            dynamics=DynamicsConfig(**d.get("dynamics", {})),
            coupling=CouplingConfig(**d.get("coupling", {})),
            doubt=DoubtConfig(**d.get("doubt", {})),
            noise=NoiseConfig(**d.get("noise", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Mem4Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def summary(self) -> str:
        """Human-readable summary of non-default parameters."""
        default = Mem4Config()
        changes = []
        for section_name in ["dynamics", "coupling", "doubt", "noise"]:
            current = getattr(self, section_name)
            default_section = getattr(default, section_name)
            for field_name in vars(current):
                val = getattr(current, field_name)
                default_val = getattr(default_section, field_name)
                if val != default_val:
                    changes.append(f"  {section_name}.{field_name}: {default_val} → {val}")
        if not changes:
            return "Mem4Config: all defaults"
        return "Mem4Config changes from defaults:\n" + "\n".join(changes)
