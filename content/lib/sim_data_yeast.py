

import numpy as np

def generate_pressure_dataset(
    times_s=None,
    glucose_mg_ml=None,
    n_reps=1,
    seed=42,
    p_max=1200.0,
    tau=220.0,
    g_half=12.0,
    hill=1.3,
    lin_time=0.18,
    noise_abs=8.0,
    noise_rel=0.035,
    missing_base=0.00,
    missing_pressure_scale=0.00025,
    missing_time_scale=0.00015,
):
    rng = np.random.default_rng(seed)

    if times_s is None:
        times_s = np.arange(30, 541, 30)
    else:
        times_s = np.asarray(times_s, dtype=float)

    if glucose_mg_ml is None:
        glucose_mg_ml = np.array([5, 10, 20, 40], dtype=float)
    else:
        glucose_mg_ml = np.asarray(glucose_mg_ml, dtype=float)

    f_g = (glucose_mg_ml**hill) / (g_half**hill + glucose_mg_ml**hill)

    f_t = 1.0 - np.exp(-times_s / tau)
    f_t = np.clip(f_t + lin_time * (times_s / times_s.max()), 0, 1.25)

    P_mean = p_max * np.outer(f_t, f_g)
    P_mean = P_mean * (1.0 - 0.10 * (P_mean / p_max)**2)
    P_mean = np.clip(P_mean, 0, p_max)

    pressure = np.empty((n_reps, times_s.size, glucose_mg_ml.size), dtype=float)

    for r in range(n_reps):
        eps = rng.normal(0.0, 1.0, size=P_mean.shape)
        sigma = noise_abs + noise_rel * P_mean
        P = np.clip(P_mean + sigma * eps, 0, None)

        p_miss = missing_base + missing_pressure_scale * P + missing_time_scale * (times_s[:, None])
        p_miss = np.clip(p_miss, 0, 0.35)

        miss_mask = rng.uniform(0, 1, size=P.shape) < p_miss
        high_g = glucose_mg_ml[None, :] >= 20
        late_t = times_s[:, None] >= 270
        miss_mask |= (rng.uniform(0, 1, size=P.shape) < 0.10) & high_g & late_t

        P[miss_mask] = np.nan
        pressure[r] = P

    return times_s, glucose_mg_ml, pressure





