"""Steps 1-2: inspect the dataset, split with a fixed seed, train and save the classifier.
Run:  python train.py
"""
from crop_agents.data import DatasetError, inspect_dataset, load_dataset
from crop_agents.model_tool import train_and_save


def main() -> None:
    try:
        df = load_dataset()
    except DatasetError as exc:
        raise SystemExit(f"\n{exc}\n")
    print("=" * 70, "\nDATASET INSPECTION\n" + "=" * 70)
    print(inspect_dataset(df))
    print("\n" + "=" * 70, "\nTRAINING (RandomForest, fixed random seed, stratified 80/20 split)\n" + "=" * 70)
    m = train_and_save(df)
    print(f"Train rows: {m['n_train']} | Test rows: {m['n_test']} | Seed: {m['random_seed']}")
    print(f"Held-out test accuracy: {m['accuracy']:.2%}")
    print("Feature importance:", m["feature_importance"])
    print("\n" + m["report"])
    print("Saved model -> models/crop_model.joblib and metrics -> outputs/metrics.json")


if __name__ == "__main__":
    main()
