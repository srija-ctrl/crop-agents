"""OPTIONAL smoke-test helper.

Creates a SYNTHETIC stand-in for Crop_recommendation.csv (approximate per-crop averages + noise)
so you can run the code before downloading the real Kaggle file.
DO NOT use this for your submission - use the real dataset:
  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Usage:  python scripts/make_sample_data.py [output_path]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# crop: (N, P, K, temperature, humidity, ph, rainfall)  approximate means
PROFILES = {
    "rice": (80, 48, 40, 23.7, 82, 6.4, 236), "maize": (77, 48, 20, 22.4, 65, 6.2, 85),
    "chickpea": (40, 68, 80, 18.9, 17, 7.3, 80), "kidneybeans": (21, 68, 20, 20, 21, 5.7, 106),
    "pigeonpeas": (21, 68, 20, 27.7, 48, 5.8, 149), "mothbeans": (21, 48, 20, 28.2, 53, 6.8, 51),
    "mungbean": (21, 48, 20, 28.5, 85, 6.7, 48), "blackgram": (40, 67, 19, 30, 65, 7.1, 68),
    "lentil": (19, 68, 19, 24.5, 65, 6.9, 46), "pomegranate": (19, 19, 40, 21.8, 90, 6.4, 108),
    "banana": (100, 82, 50, 27.4, 80, 6.0, 105), "mango": (20, 27, 30, 31.2, 50, 5.8, 95),
    "grapes": (23, 132, 200, 23.8, 82, 6.0, 69), "watermelon": (99, 17, 50, 25.6, 85, 6.5, 50),
    "muskmelon": (100, 17, 50, 28.7, 92, 6.4, 25), "apple": (21, 134, 199, 22.6, 92, 5.9, 113),
    "orange": (20, 10, 10, 22.8, 92, 7.0, 110), "papaya": (50, 59, 50, 33.7, 92, 6.7, 143),
    "coconut": (22, 17, 31, 27.4, 94, 6.0, 176), "cotton": (118, 46, 20, 24, 80, 6.9, 80),
    "jute": (78, 46, 40, 25, 80, 6.7, 175), "coffee": (101, 29, 30, 25.5, 58, 6.8, 158),
}
COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/Crop_recommendation_SYNTHETIC.csv")
    rng = np.random.default_rng(0)
    rows = []
    for crop, mean in PROFILES.items():
        mean = np.array(mean, dtype=float)
        sd = np.maximum(mean * 0.08, [3, 3, 3, 1.0, 3, 0.15, 6])
        x = rng.normal(mean, sd, size=(100, 7))
        x[:, 0:3] = np.clip(x[:, 0:3], 0, None)
        x[:, 4] = np.clip(x[:, 4], 10, 100)
        x[:, 5] = np.clip(x[:, 5], 3.5, 9.9)
        x[:, 6] = np.clip(x[:, 6], 15, None)
        d = pd.DataFrame(x, columns=COLS).round(2)
        d["label"] = crop
        rows.append(d)
    df = pd.concat(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote SYNTHETIC data ({len(df)} rows) to {out}. Use only for smoke-testing.")
    print("To use it:  export CROP_CSV=" + str(out))


if __name__ == "__main__":
    main()
