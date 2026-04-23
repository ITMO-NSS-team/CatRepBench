import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from genbench.generative.tabddpm.tabddpm import TabDDPMGenerative
from experiments.tabddpm_tuning import tune_tabddpm
from genbench.data.schema import TabularSchema


def prepare_iris_data() -> tuple[pd.DataFrame, TabularSchema]:
    """Load Iris and prepare data with StandardScaler."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Scale all numeric columns (including target)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

    # Create schema: all columns are continuous features (target included)
    schema = TabularSchema(
        continuous_cols=list(df_scaled.columns),
        discrete_cols=[],
        categorical_cols=[],
        target_col=None,  # target is treated as a regular feature
    )

    return df_scaled, schema


def main():
    print("=" * 60)
    print("TabDDPM Hyperparameter Tuning Demo on Iris")
    print("=" * 60)

    # 1. Prepare data
    print("\n[1] Loading and preparing Iris dataset...")
    df, schema = prepare_iris_data()
    print(f"    Dataset shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # 2. Run tuning with a small budget
    print("\n[2] Starting hyperparameter tuning...")
    print("    (This may take a few minutes depending on your hardware)")
    result = tune_tabddpm(
        df=df,
        schema=schema,
        dataset="iris_demo",
        encoding_method="one_hot_representation",  # No categorical encoding needed
        n_trials=5,  # Small number for demonstration
        seed=42,
        output_root=Path("experiments/optuna_results_demo"),
        save_model=True,
        device="cuda",  # Change to "cpu" if CUDA not available
        timeout_seconds=None,
    )

    # 3. Display results
    print("\n[3] Tuning completed!")
    print(f"    Best Wasserstein distance: {result.best_value:.4f}")
    print(f"    Best hyperparameters:")
    for k, v in result.best_params.items():
        print(f"        {k}: {v}")
    print(f"    Optimal number of training steps: {result.steps}")
    print(f"    Total tuning time: {result.duration_seconds:.1f} seconds")

    print("\n[4] Output files:")
    print(f"    Summary:      {result.summary_path}")
    print(f"    Trials CSV:   {result.trials_path}")
    print(f"    Best params:  {result.best_params_path}")
    if result.model_artifacts_dir:
        print(f"    Model artifacts: {result.model_artifacts_dir}")

    # 4. Quick test: generate samples with best model
    print("\n[5] Generating 10 synthetic samples with best model...")
    best_model = TabDDPMGenerative(**result.best_params)
    best_model.fit(df, schema, source_schema=schema)
    synthetic = best_model.sample(10)

    # Inverse scaling (optional, only for visualization)
    # Since we used StandardScaler during prep, inverse transform here
    from sklearn.datasets import load_iris
    original_data = load_iris()
    original_df = pd.DataFrame(original_data.data,
                               columns=original_data.feature_names)
    original_df['target'] = original_data.target
    num_cols = original_df.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    scaler.fit(original_df[num_cols])  # refit on original data
    synthetic_original_scale = synthetic.copy()
    synthetic_original_scale[num_cols] = scaler.inverse_transform(
        synthetic[num_cols])
    synthetic_original_scale['target'] = np.round(
        synthetic_original_scale['target']).astype(int)

    print("\n    Synthetic samples (original scale):")
    print(synthetic_original_scale.head())

    print("\n" + "=" * 60)
    print("Demo finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
