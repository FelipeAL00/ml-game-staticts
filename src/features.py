"""Advanced feature engineering for Mega Sena lottery data."""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional


TOTAL_NUMBERS = 60
NUMBERS_PER_DRAW = 6


def cyclical_encode(value: float, period: float) -> tuple[float, float]:
    """Encode a periodic value as (sin, cos) pair to capture cyclical nature."""
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def build_cooccurrence_features(
    recent_draws: np.ndarray,
    n: int,
) -> dict[str, float]:
    """Compute co-occurrence features for number n.

    How often does n appear together with other numbers?

    Args:
        recent_draws: Array of shape (window, 6) with recent draws.
        n: The number to compute features for (1-60).

    Returns:
        Dictionary with co-occurrence features.
    """
    draws_with_n = [draw for draw in recent_draws if n in draw]

    if not draws_with_n:
        return {
            "cooc_avg_partner_freq": 0.0,
            "cooc_most_common_partner_freq": 0.0,
            "cooc_appearances": 0.0,
        }

    partner_counts: Counter = Counter()
    for draw in draws_with_n:
        for m in draw:
            if m != n:
                partner_counts[m] += 1

    total_draws = len(recent_draws)
    total_pairs = len(draws_with_n) * (NUMBERS_PER_DRAW - 1)

    avg_partner_freq = sum(partner_counts.values()) / total_pairs if total_pairs > 0 else 0.0
    most_common_freq = (
        partner_counts.most_common(1)[0][1] / total_draws
        if partner_counts
        else 0.0
    )

    return {
        "cooc_avg_partner_freq": float(avg_partner_freq),
        "cooc_most_common_partner_freq": float(most_common_freq),
        "cooc_appearances": float(len(draws_with_n) / total_draws),
    }


def build_frequency_trend_features(
    recent_draws: np.ndarray,
    n: int,
    windows: list[int],
) -> dict[str, float]:
    """Compute frequency trend features across multiple windows.

    Args:
        recent_draws: All draws available up to this point.
        n: Number to compute features for.
        windows: List of window sizes (e.g., [10, 30, 60]).

    Returns:
        Dictionary with trend features.
    """
    freqs = {}
    for w in windows:
        draws_w = recent_draws[-w:]
        freq_w = np.sum(draws_w == n) / (len(draws_w) * NUMBERS_PER_DRAW)
        freqs[w] = freq_w

    features: dict[str, float] = {}

    sorted_w = sorted(windows)
    if len(sorted_w) >= 2:
        short_w, long_w = sorted_w[0], sorted_w[-1]
        f_short = freqs.get(short_w, 0.0)
        f_long = freqs.get(long_w, 0.0)

        features[f"freq_trend_{short_w}_{long_w}"] = float(f_short - f_long)
        features[f"freq_acceleration_{short_w}_{long_w}"] = (
            float((f_short - f_long) / (f_long + 1e-8))
        )

    if len(sorted_w) >= 3:
        w_mid = sorted_w[1]
        f_short = freqs.get(sorted_w[0], 0.0)
        f_mid = freqs.get(w_mid, 0.0)
        f_long = freqs.get(sorted_w[-1], 0.0)
        features["freq_volatility"] = float(np.std([f_short, f_mid, f_long]))

    return features


def build_gap_sequence_features(recent_draws: np.ndarray, n: int) -> dict[str, float]:
    """Compute features based on the sequence of gaps between appearances.

    Args:
        recent_draws: Recent draws array.
        n: Number to analyze.

    Returns:
        Dictionary with gap sequence features.
    """
    appearances = [i for i, draw in enumerate(recent_draws) if n in draw]

    if len(appearances) < 2:
        expected_gap = TOTAL_NUMBERS / NUMBERS_PER_DRAW
        last_seen = len(recent_draws) - appearances[-1] if appearances else len(recent_draws)
        return {
            "gap_mean": float(expected_gap),
            "gap_std": 0.0,
            "gap_last": float(last_seen),
            "gap_overdue_ratio": float(last_seen / expected_gap),
            "gap_cv": 0.0,
        }

    gaps = [appearances[i + 1] - appearances[i] for i in range(len(appearances) - 1)]
    last_seen = len(recent_draws) - appearances[-1]
    expected_gap = TOTAL_NUMBERS / NUMBERS_PER_DRAW

    gap_mean = float(np.mean(gaps))
    gap_std = float(np.std(gaps))

    return {
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "gap_last": float(last_seen),
        "gap_overdue_ratio": float(last_seen / (gap_mean + 1e-8)),
        "gap_cv": float(gap_std / (gap_mean + 1e-8)),
    }


def build_regression_to_mean_features(
    freq_n: float,
    recent_draws: np.ndarray,
    n: int,
) -> dict[str, float]:
    """Features capturing tendency to regress toward expected frequency.

    In theory, each number should appear with frequency = 6/60 = 0.1.
    Numbers deviating strongly tend to revert over time.

    Args:
        freq_n: Current frequency of number n.
        recent_draws: Recent draws.
        n: Number being analyzed.

    Returns:
        Dictionary with regression features.
    """
    expected_freq = NUMBERS_PER_DRAW / TOTAL_NUMBERS
    deviation = freq_n - expected_freq
    z_score = deviation / (np.sqrt(expected_freq * (1 - expected_freq) / max(len(recent_draws), 1)) + 1e-8)

    return {
        "freq_deviation_from_expected": float(deviation),
        "freq_z_score": float(z_score),
        "freq_overperforming": float(deviation > 0),
        "freq_abs_deviation": float(abs(deviation)),
    }


def build_advanced_features(
    df: pd.DataFrame,
    window: int = 30,
    cooccurrence_window: int = 60,
) -> pd.DataFrame:
    """Build advanced feature set for all draws.

    Extends base features with:
    - Cyclical temporal encoding
    - Co-occurrence features
    - Frequency trends across windows
    - Gap sequence analysis
    - Regression-to-mean features

    Args:
        df: DataFrame with Mega Sena data (must have dezena_1..6 and data_parsed).
        window: Primary sliding window size.
        cooccurrence_window: Window for co-occurrence computation.

    Returns:
        DataFrame with all features.
    """
    dezena_cols = ["dezena_1", "dezena_2", "dezena_3", "dezena_4", "dezena_5", "dezena_6"]
    all_draws = df[dezena_cols].values.astype(int)

    multi_windows = [10, 30, 60]
    features_list = []

    for i in range(window, len(df)):
        current_date = df.iloc[i]["data_parsed"]

        month_sin, month_cos = cyclical_encode(current_date.month, 12)
        quarter_sin, quarter_cos = cyclical_encode((current_date.month - 1) // 3 + 1, 4)
        dow_sin, dow_cos = cyclical_encode(current_date.dayofweek, 7)

        recent_draws_base = all_draws[max(0, i - window): i]
        recent_draws_cooc = all_draws[max(0, i - cooccurrence_window): i]

        last_draw = sorted(all_draws[i - 1].tolist())

        per_number_features: dict[str, float] = {}

        for n in range(1, TOTAL_NUMBERS + 1):
            freq_30 = np.sum(recent_draws_base == n) / (len(recent_draws_base) * NUMBERS_PER_DRAW)

            trend_feats = build_frequency_trend_features(
                all_draws[max(0, i - max(multi_windows)): i],
                n,
                [w for w in multi_windows if i >= w],
            )
            gap_feats = build_gap_sequence_features(recent_draws_base, n)
            rtm_feats = build_regression_to_mean_features(freq_30, recent_draws_base, n)
            cooc_feats = build_cooccurrence_features(recent_draws_cooc, n)

            for k, v in trend_feats.items():
                per_number_features[f"{k}_{n}"] = v
            for k, v in gap_feats.items():
                per_number_features[f"{k}_{n}"] = v
            for k, v in rtm_feats.items():
                per_number_features[f"{k}_{n}"] = v
            for k, v in cooc_feats.items():
                per_number_features[f"{k}_{n}"] = v

        last_draw_arr = np.array(last_draw)
        soma_ultimo = int(np.sum(last_draw_arr))
        pares_ultimo = int(np.sum(last_draw_arr % 2 == 0))
        consecutivas = sum(
            1 for k in range(len(last_draw) - 1)
            if last_draw[k + 1] - last_draw[k] == 1
        )
        amplitude = int(last_draw[-1] - last_draw[0])
        mediana = float(np.median(last_draw_arr))

        gaps = [last_draw[k + 1] - last_draw[k] for k in range(len(last_draw) - 1)]

        decadas = {}
        for dec in range(6):
            low, high = dec * 10 + 1, (dec + 1) * 10
            decadas[f"decada_{low}_{high}"] = sum(1 for nn in last_draw if low <= nn <= high)

        q1 = sum(1 for nn in last_draw if 1 <= nn <= 15)
        q2 = sum(1 for nn in last_draw if 16 <= nn <= 30)
        q3 = sum(1 for nn in last_draw if 31 <= nn <= 45)
        q4 = sum(1 for nn in last_draw if 46 <= nn <= 60)

        row = {
            "concurso": df.iloc[i]["concurso"],
            "idx": i,
            "ano": current_date.year,
            "mes": current_date.month,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "quarter_sin": quarter_sin,
            "quarter_cos": quarter_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "soma_ultimo": soma_ultimo,
            "pares_ultimo": pares_ultimo,
            "impares_ultimo": 6 - pares_ultimo,
            "consecutivas": consecutivas,
            "amplitude": amplitude,
            "mediana": mediana,
            "gap_draw_medio": float(np.mean(gaps)) if gaps else 0.0,
            "gap_draw_max": float(max(gaps)) if gaps else 0.0,
            "gap_draw_min": float(min(gaps)) if gaps else 0.0,
            "gap_draw_std": float(np.std(gaps)) if len(gaps) > 1 else 0.0,
            "quadrante_1_15": q1,
            "quadrante_16_30": q2,
            "quadrante_31_45": q3,
            "quadrante_46_60": q4,
            **decadas,
            **per_number_features,
        }
        features_list.append(row)

    return pd.DataFrame(features_list)
