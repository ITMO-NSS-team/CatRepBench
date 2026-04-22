import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from genbench.generative.tabddpm import TabDDPMGenerative
from genbench.data.schema import TabularSchema


def create_dummy_schema(df, target_col=None) -> TabularSchema:
    """
    Create a simple schema for a DataFrame based on heuristics.

    WARNING: For unconditional generation with a target column it is better
    not to use a separate target_col but to include it among the features.
    """
    cont_cols = []
    disc_cols = []
    cat_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Treat as continuous if the number of unique values > 10
            if df[col].nunique() > 10:
                cont_cols.append(col)
            else:
                disc_cols.append(col)
        else:
            cat_cols.append(col)
    return TabularSchema(
        continuous_cols=cont_cols,
        discrete_cols=disc_cols,
        categorical_cols=cat_cols,
        target_col=target_col,
    )


def main() -> None:
    # 1. Load Iris dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Original dataset: {df.shape}")

    # 2. Scale ALL numeric columns (including target)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

    # 3. Create a new schema for the scaled data
    #    Important: target_col = None so that the model treats 'target' as a
    #    regular feature
    schema = TabularSchema(
        continuous_cols=list(df_scaled.columns),  # all columns -> continuous
        discrete_cols=[],
        categorical_cols=[],
        target_col=None,  # target is now part of the features
    )
    source_schema = schema  # preprocessing already done

    # 4. Initialize the model
    model = TabDDPMGenerative(
        name="tabddpm_test",
        num_timesteps=500,
        num_steps=5000,
        batch_size=32,
        device='cuda',  # or 'cpu'
        d_layers=[128, 128],
        dim_t=64
    )

    # 5. Train on the scaled data
    print("\nStarting training...")
    model.fit(df_scaled, schema, source_schema)
    print("Training finished.")

    # 6. Generate synthetic data (in StandardScaler scale)
    print("\nGenerating 50 synthetic records...")
    synthetic_scaled = model.sample(n=50)

    # 7. Inverse transform back to the original scale
    synthetic = synthetic_scaled.copy()
    synthetic[num_cols] = scaler.inverse_transform(synthetic_scaled[num_cols])

    # 8. Postprocessing: round target to integer classes (if present)
    if 'target' in synthetic.columns:
        synthetic['target'] = np.round(synthetic['target']).astype(int)

    print(f"Generated dataset: {synthetic.shape}")
    print("First 5 rows of synthetic data (real values):")
    print(synthetic.head())

    # 9. Check for NaNs
    assert not synthetic.isnull().any().any(), ("Error: synthetic data "
                                                "contains NaN.")
    print("\nCheck passed: no NaNs present.")

    # 10. Print final loss value
    loss_hist = model.get_loss_history()
    if loss_hist:
        print(f"Final loss value: {loss_hist['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
