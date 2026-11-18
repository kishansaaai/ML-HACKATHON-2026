"""Decoder parameters. Defaults are starting engineering values, not tuned claims."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    beta_m: float = 50.0
    emission_sigma_m: float | None = None
    gap_threshold_s: float = 120.0
    transition_cutoff_m: float = 10_000.0
    unreachable_log_penalty: float = -20.0
    min_probability: float = 1e-12
    spatial_cell_m: float = 128.0
    max_cells_per_arc: int = 256
    distance_cache_size: int = 100_000
    source_cache_size: int = 4

    def __post_init__(self) -> None:
        if self.beta_m <= 0 or self.gap_threshold_s < 0 or self.transition_cutoff_m <= 0:
            raise ValueError("beta_m and transition_cutoff_m must be positive; gap threshold cannot be negative")
        if self.emission_sigma_m is not None and self.emission_sigma_m <= 0:
            raise ValueError("emission_sigma_m must be positive when set")
        if self.unreachable_log_penalty > 0:
            raise ValueError("unreachable_log_penalty must be non-positive")
        if not 0 < self.min_probability < 1:
            raise ValueError("min_probability must be between zero and one")


