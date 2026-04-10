import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter


def cyclical_encode(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """Codifica uma coluna cíclica usando seno e cosseno."""
    df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / period)
    df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / period)
    return df


def _pair_cooccurrence_score(past_draws: np.ndarray, last_draw: list[int]) -> float:
    score = 0.0
    if len(past_draws) == 0:
        return score

    pair_counter = Counter()
    for draw in past_draws:
        for a, b in combinations(sorted(draw), 2):
            pair_counter[(a, b)] += 1

    for a, b in combinations(sorted(last_draw), 2):
        score += pair_counter.get((a, b), 0)

    return float(score)


def build_enhanced_features(
    df: pd.DataFrame,
    window: int = 30,
    windows: tuple[int, ...] = (10, 30, 60),
) -> pd.DataFrame:
    """Gera um conjunto de features mais rico para a Mega Sena."""
    df = df.copy()
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)

    df["mes"] = df["data_parsed"].dt.month
    df["dia"] = df["data_parsed"].dt.day
    df["trimestre"] = (df["mes"] - 1) // 3 + 1
    df = cyclical_encode(df, "mes", 12)
    df = cyclical_encode(df, "dia", 31)
    df = cyclical_encode(df, "trimestre", 4)

    dezena_cols = [f"dezena_{i}" for i in range(1, 7)]
    all_draws = df[dezena_cols].values.astype(int)

    features_list = []
    expected_freq = 1.0 / TOTAL_NUMBERS if (TOTAL_NUMBERS := 60) else 0.0

    for i in range(window, len(df)):
        recent_draws = all_draws[max(0, i - window) : i]
        recent_numbers = recent_draws.flatten()

        freq_features = {}
        delay_features = {}
        hot_cold_features = {}

        for w in windows:
            draws_w = all_draws[max(0, i - w) : i]
            freq_w = Counter(draws_w.flatten())
            suffix = f"_{w}" if w != window else ""

            for n in range(1, TOTAL_NUMBERS + 1):
                denom = len(draws_w) if len(draws_w) > 0 else 1
                freq_features[f"freq_{n}{suffix}"] = freq_w.get(n, 0) / denom
                last_seen = -1
                for j in range(len(draws_w) - 1, -1, -1):
                    if n in draws_w[j]:
                        last_seen = j
                        break
                delay_features[f"atraso_{n}{suffix}"] = (len(draws_w) - last_seen) if last_seen >= 0 else w + 1

            hot_cold_features[f"hot_count{suffix}"] = sum(
                1 for n in range(1, TOTAL_NUMBERS + 1) if freq_w.get(n, 0) / denom > 0.12
            )
            hot_cold_features[f"cold_count{suffix}"] = sum(
                1 for n in range(1, TOTAL_NUMBERS + 1) if freq_w.get(n, 0) / denom < 0.08
            )

        base_freq = Counter(recent_numbers)
        freq_features.update(
            {f"freq_{n}": base_freq.get(n, 0) / max(1, len(recent_draws)) for n in range(1, TOTAL_NUMBERS + 1)}
        )
        delay_features.update(
            {
                f"atraso_{n}": next(
                    (len(recent_draws) - j for j in range(len(recent_draws) - 1, -1, -1) if n in recent_draws[j]),
                    window + 1,
                )
                for n in range(1, TOTAL_NUMBERS + 1)
            }
        )

        last_draw = sorted(all_draws[i - 1])
        soma_ultimo = int(np.sum(last_draw))
        pares_ultimo = int(np.sum(np.array(last_draw) % 2 == 0))
        impares_ultimo = 6 - pares_ultimo

        somas = [int(np.sum(d)) for d in recent_draws]
        soma_media = np.mean(somas)
        soma_std = np.std(somas)

        consecutivas = sum(
            1 for k in range(len(last_draw) - 1) if last_draw[k + 1] - last_draw[k] == 1
        )

        q1 = sum(1 for n in last_draw if 1 <= n <= 15)
        q2 = sum(1 for n in last_draw if 16 <= n <= 30)
        q3 = sum(1 for n in last_draw if 31 <= n <= 45)
        q4 = sum(1 for n in last_draw if 46 <= n <= 60)

        gaps = [last_draw[k + 1] - last_draw[k] for k in range(len(last_draw) - 1)]
        gap_medio = np.mean(gaps) if gaps else 0
        gap_max = max(gaps) if gaps else 0
        gap_min = min(gaps) if gaps else 0
        gap_std = np.std(gaps) if len(gaps) > 1 else 0

        decadas = {}
        for dec in range(6):
            low = dec * 10 + 1
            high = (dec + 1) * 10
            decadas[f"decada_{low}_{high}"] = sum(1 for n in last_draw if low <= n <= high)

        streak_features = {}
        for n in range(1, TOTAL_NUMBERS + 1):
            streak = 0
            for j in range(len(recent_draws) - 1, -1, -1):
                if n in recent_draws[j]:
                    streak += 1
                else:
                    break
            streak_features[f"streak_{n}"] = streak

        hot_count = sum(
            1 for n in range(1, TOTAL_NUMBERS + 1) if base_freq.get(n, 0) / max(1, len(recent_draws)) > 0.12
        )
        cold_count = sum(
            1 for n in range(1, TOTAL_NUMBERS + 1) if base_freq.get(n, 0) / max(1, len(recent_draws)) < 0.08
        )

        pair_score = _pair_cooccurrence_score(all_draws[max(0, i - window) : i], last_draw)

        freq_short = Counter(all_draws[max(0, i - windows[0]) : i].flatten())
        freq_long = Counter(all_draws[max(0, i - windows[-1]) : i].flatten())
        trend_values = [
            (freq_short.get(n, 0) / max(1, windows[0])) - (freq_long.get(n, 0) / max(1, windows[-1]))
            for n in last_draw
        ]
        trend_mean = float(np.mean(trend_values))
        deviation_expected = float(np.mean([base_freq.get(n, 0) / max(1, len(recent_draws)) - expected_freq for n in last_draw]))

        current_date = df.iloc[i]["data_parsed"]
        row = {
            "concurso": df.iloc[i]["concurso"],
            "idx": i,
            "ano": current_date.year,
            "mes": current_date.month,
            "dia": current_date.day,
            "trimestre": (current_date.month - 1) // 3 + 1,
            "mes_sin": np.sin(2 * np.pi * current_date.month / 12),
            "mes_cos": np.cos(2 * np.pi * current_date.month / 12),
            "dia_sin": np.sin(2 * np.pi * current_date.day / 31),
            "dia_cos": np.cos(2 * np.pi * current_date.day / 31),
            "trimestre_sin": np.sin(2 * np.pi * ((current_date.month - 1) // 3 + 1) / 4),
            "trimestre_cos": np.cos(2 * np.pi * ((current_date.month - 1) // 3 + 1) / 4),
            **freq_features,
            **delay_features,
            **streak_features,
            **decadas,
            **hot_cold_features,
            "soma_ultimo": soma_ultimo,
            "pares_ultimo": pares_ultimo,
            "impares_ultimo": impares_ultimo,
            "soma_media_janela": soma_media,
            "soma_std_janela": soma_std,
            "consecutivas": consecutivas,
            "quadrante_1_15": q1,
            "quadrante_16_30": q2,
            "quadrante_31_45": q3,
            "quadrante_46_60": q4,
            "gap_medio": gap_medio,
            "gap_max": gap_max,
            "gap_min": gap_min,
            "gap_std": gap_std,
            "hot_count": hot_count,
            "cold_count": cold_count,
            "amplitude": last_draw[-1] - last_draw[0],
            "mediana": float(np.median(last_draw)),
            "pair_cooccurrence_score": pair_score,
            "trend_mean": trend_mean,
            "deviation_expected_freq": deviation_expected,
        }
        features_list.append(row)

    return pd.DataFrame(features_list)
