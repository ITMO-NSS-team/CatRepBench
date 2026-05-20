import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
from typing import Dict

from genbench.data.schema import TabularSchema
from genbench.data.splits import SplitConfigHoldout
from genbench.transforms.pipeline import TransformPipeline
from genbench.transforms.continuous import ContinuousStandardScaler
from genbench.transforms.categorical import CategoricalRepresentationTransform
from genbench.transforms.missing import DropMissingRows
from genbench.transforms.target import TargetTypePreprocessor
from genbench.generative.tabddpm.tabddpm import TabDDPMGenerative
from genbench.generative.tabddpm.utils import FoundNANsError
from genbench.evaluation.distribution.wasserstein import \
    WassersteinDistanceMetric
from genbench.evaluation.distribution.marginal_kl import \
    compute_marginal_kl_mean
from genbench.evaluation.distribution.corr_frobenius import \
    compute_corr_frobenius
from experiments.tabddpm_tuning import tune_tabddpm

DEFAULT_ENCODINGS = [
    "one_hot_representation",
    "ordinal_representation",
    "frequency_representation",
    "polynomial_representation",
    "helmert_representation",
    "backward_difference_representation",
    "binary_representation",
    "sum_representation",
    "hash_representation",
    "similarity_representation",
    "gumbel_softmax_representation",
    "gel_representation",
]

TARGET_MAP = {
    "openml_conference_attendance": "isPresent",
    "uci_Heart_Disease": "num",
    "uci_Soybean__Large_": "class",
    "uci_Forest_Fires": "area",
    "uci_Student_Performance": "G3",
    "uci_Credit_Approval": "A16",
    "openml_eucalyptus": "Utility",
    "uci_Mammographic_Mass": "Severity",
    "uci_Statlog__German_Credit_Data_": "class",
    "openml_PhishingWebsites": "Result",
    "uci_Online_Shoppers_Purchasing_Intention_Dataset": "Revenue",
    "openml_nursery": "class",
    "uci_HTRU2": "class",
    "openml_MagicTelescope": "class:",
    "openml_letter": "class",
    "openml_default-of-credit-card-clients": "y",
}


def get_datasets_with_size(raw_dir: Path) -> pd.DataFrame:
    files = list(raw_dir.glob("*.csv"))
    sizes = []
    for f in files:
        try:
            n_rows = pd.read_csv(f).shape[0]
            sizes.append({"file": f, "name": f.stem, "n_rows": n_rows})
        except Exception as e:
            print(f"Reading error {f}: {e}")
    df_info = pd.DataFrame(sizes)
    if df_info.empty:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")
    df_info = df_info.sort_values("n_rows").reset_index(drop=True)
    return df_info


def get_target_column(df: pd.DataFrame, dataset_name: str) -> str:
    if dataset_name in TARGET_MAP:
        return TARGET_MAP[dataset_name]
    return df.columns[-1]


def ensure_tuning(df: pd.DataFrame, schema: TabularSchema, dataset_name: str,
                  encoding_method: str, n_trials: int,
                  device: str, seed: int, output_root: Path):
    """Runs tuning if best_params.json is missing. Returns a dict or None on
    error."""
    optuna_dir = output_root / "tabddpm" / dataset_name / encoding_method
    best_params_path = optuna_dir / "best_params.json"
    summary_path = optuna_dir / "summary.json"

    if best_params_path.exists() and summary_path.exists():
        print(f"  Tuning already done, loading parameters from {optuna_dir}")
        with open(best_params_path) as f:
            best_data = json.load(f)
        best_params = best_data["best_params"]
        with open(summary_path) as f:
            task_type = json.load(f)["task_type"]
        return {"best_params": best_params, "task_type": task_type}
    else:
        print(f"  Running tuning for {encoding_method}...")
        try:
            tuning_result = tune_tabddpm(
                df=df,
                schema=schema,
                dataset=dataset_name,
                encoding_method=encoding_method,
                n_trials=n_trials,
                seed=seed,
                task_type=None,
                holdout_cfg=SplitConfigHoldout(val_size=0.2, shuffle=True,
                                               random_seed=5),
                output_root=output_root,
                save_model=False,
                device=device,
            )
            with open(tuning_result.best_params_path) as f:
                best_data = json.load(f)
            best_params = best_data["best_params"]
            with open(tuning_result.summary_path) as f:
                task_type = json.load(f)["task_type"]
            return {"best_params": best_params, "task_type": task_type}
        except (FoundNANsError, Exception) as e:
            print(
                f"  Tuning for {encoding_method} failed: "
                f"{type(e).__name__}: {e}")
            return None


def tstr_catboost_classifier(
        train_real: pd.DataFrame,
        test_real: pd.DataFrame,
        synth_train: pd.DataFrame,
        schema: TabularSchema,
        random_seed: int = 42,
) -> Dict[str, float]:
    """TSTR evaluation for classification using CatBoostClassifier and F1
    score."""
    if schema.target_col is None:
        raise ValueError("schema.target_col must be set for TSTR evaluation.")

    from sklearn.preprocessing import LabelEncoder

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        required = list(
            dict.fromkeys(schema.feature_cols + [schema.target_col]))
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df[required].copy()

    train_real = _prepare(train_real)
    test_real = _prepare(test_real)
    synth_train = _prepare(synth_train)

    # Encode the target variable into numerical labels
    le = LabelEncoder()
    all_labels = pd.concat([
        train_real[schema.target_col],
        test_real[schema.target_col],
        synth_train[schema.target_col]
    ], axis=0)
    le.fit(all_labels)

    train_real['target_encoded'] = le.transform(train_real[schema.target_col])
    test_real['target_encoded'] = le.transform(test_real[schema.target_col])
    synth_train['target_encoded'] = le.transform(
        synth_train[schema.target_col])

    target_col_encoded = 'target_encoded'
    # Features for CatBoost - all feature columns except the target itself
    feature_cols = [col for col in schema.feature_cols if
                    col != schema.target_col]
    cat_features = [i for i, c in enumerate(feature_cols) if
                    c in schema.categorical_cols]

    def _fit_and_eval(train_df: pd.DataFrame) -> Dict[str, float]:
        train_pool = Pool(
            train_df[feature_cols],
            train_df[target_col_encoded],
            cat_features=cat_features
        )
        test_pool = Pool(
            test_real[feature_cols],
            test_real[target_col_encoded],
            cat_features=cat_features
        )
        catboost_device = 'GPU' if __import__(
            'torch').cuda.is_available() else 'CPU'
        model = CatBoostClassifier(random_seed=random_seed, verbose=100,
                                   iterations=150, task_type=catboost_device)
        model.fit(train_pool)
        preds_encoded = model.predict(test_pool)
        preds_original = le.inverse_transform(preds_encoded.astype(int))
        f1 = f1_score(test_real[schema.target_col], preds_original,
                      average='weighted')
        return {"f1": float(f1)}

    try:
        real_scores = _fit_and_eval(train_real)
        synth_scores = _fit_and_eval(synth_train)
    except Exception as e:
        print(f"    CatBoost error: {e}. Setting metrics to inf.")
        return {
            "f1_real": float('nan'),
            "f1_synth": float('nan'),
            "f1_pct_diff": float('inf'),
        }

    def _pct_diff(real_val: float, synth_val: float) -> float:
        if real_val == 0:
            return float("inf")
        return float(abs(real_val - synth_val) / abs(real_val) * 100.0)

    return {
        "f1_real": real_scores["f1"],
        "f1_synth": synth_scores["f1"],
        "f1_pct_diff": _pct_diff(real_scores["f1"], synth_scores["f1"]),
    }


def run_cv_for_encoding(df: pd.DataFrame, schema: TabularSchema,
                        encoding_method: str, best_params: dict,
                        task_type: str,
                        n_folds: int = 5, random_state: int = 42):
    """Cross-validation. Returns a dict with metrics or None on error."""
    target_col = schema.target_col

    if task_type == 'classification':
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=random_state)
        splits = skf.split(df, df[target_col])
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = kf.split(df)

    wd_metric = WassersteinDistanceMetric()
    results = {
        "wasserstein": [],
        "kl_divergence": [],
        "corr_frobenius": [],
        "tstr_real": [],
        "tstr_synth": [],
        "tstr_pct_diff": []
    }

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"    Fold {fold + 1}/{n_folds}")
        X_train = df.iloc[train_idx].copy()
        X_test = df.iloc[test_idx].copy()

        transforms = [
            DropMissingRows(),
            ContinuousStandardScaler(),
            CategoricalRepresentationTransform(
                representation_name=encoding_method),
            TargetTypePreprocessor(task_type=task_type)
        ]
        pipeline = TransformPipeline(transforms=transforms)
        pipeline.fit(X_train, schema)
        scaler = None
        for tr in pipeline.transforms:
            if getattr(tr, 'name', '') == 'continuous_standard_scaler':
                scaler = tr
                break
        X_train_processed = pipeline.transform(X_train)
        try:
            X_test_processed = pipeline.transform(X_test)
        except ValueError as e:
            if "Found target categories not present in train" in str(e):
                print(
                    f"    Skipping method due to new categories in test: {e}")
                return None
            else:
                raise
        if scaler is not None and scaler.fitted_ and scaler.continuous_cols_:
            X_train_raw = scaler.inverse_transform(X_train_processed)
        else:
            X_train_raw = X_train_processed

        transformed_schema = TabularSchema.infer_from_dataframe(
            X_train_processed, target_col=schema.target_col,
            id_col=schema.id_col,
            feature_cols=list(X_train_processed.columns)
        )

        if (task_type == 'classification' and transformed_schema.target_col
                is not None):
            target = transformed_schema.target_col
            if target in transformed_schema.continuous_cols:
                transformed_schema.continuous_cols.remove(target)
            if target in transformed_schema.discrete_cols:
                transformed_schema.discrete_cols.remove(target)
            if target not in transformed_schema.categorical_cols:
                transformed_schema.categorical_cols.append(target)

        try:
            model = TabDDPMGenerative(**best_params)
            print("Transformed schema feature cols:",
                  transformed_schema.feature_cols)
            model.fit(X_train_processed, transformed_schema,
                      source_schema=schema)

            print("Model numerical cols:", model.numerical_cols_)
            print("Model categorical cols:", model.categorical_cols_)
            print("Model column order:", model.column_order_)

            synth_train = model.sample(len(X_train_processed))
            print(synth_train)
            print("NaNs: ", synth_train.isna().sum())
        except (FoundNANsError, Exception) as e:
            print(
                f"    Error during training/generation in fold {fold + 1}: "
                f"{type(e).__name__}: {e}")
            return None

        # Inverse scale numerical columns (independent of categorical ones)
        if scaler is not None and scaler.fitted_ and scaler.continuous_cols_:
            synth_raw = scaler.inverse_transform(synth_train)
            test_raw = scaler.inverse_transform(X_test_processed)
        else:
            synth_raw = synth_train
            test_raw = X_test_processed
        # Attempt to inverse transform categorical features
        cat_transformer = None
        for tr in pipeline.transforms:
            if isinstance(tr, CategoricalRepresentationTransform):
                cat_transformer = tr
                break

        if cat_transformer is not None:
            try:
                synth_raw = cat_transformer.inverse_transform(synth_raw)
                test_raw = cat_transformer.inverse_transform(test_raw)
                X_train_raw = cat_transformer.inverse_transform(X_train_raw)
                metric_schema = schema
                print("    Metrics computed on raw (original) features.")
            except Exception as e:
                print(f"    WARNING: category inverse_transform failed ({e}). "
                      "Metrics computed on transformed features.")
                metric_schema = transformed_schema
        else:
            metric_schema = schema

        # For Corr dist, cast categorical columns to string
        for col in schema.categorical_cols:
            if col in test_raw.columns and col in synth_raw.columns:
                test_raw[col] = test_raw[col].astype(str)
                synth_raw[col] = synth_raw[col].astype(str)

        # WD - only continuous features
        wd_cols = metric_schema.continuous_cols
        if wd_cols:
            schema_wd = TabularSchema(
                continuous_cols=wd_cols,
                discrete_cols=[],
                categorical_cols=[],
            )
            wasserstein = wd_metric.compute(test_raw, synth_raw, schema_wd)
        else:
            wasserstein = float('nan')

        # KL - continuous + discrete features
        kl_cols = (metric_schema.continuous_cols +
                   metric_schema.discrete_cols)
        if kl_cols:
            schema_kl = TabularSchema(
                continuous_cols=metric_schema.continuous_cols,
                discrete_cols=metric_schema.discrete_cols,
                categorical_cols=[],
            )
            kl_div = compute_marginal_kl_mean(test_raw, synth_raw,
                                              schema_kl)
        else:
            kl_div = float('nan')

        # Corr - all features, using Spearman correlation
        corr_frob = compute_corr_frobenius(test_raw, synth_raw,
                                           metric_schema)

        if task_type == 'regression':
            from catboost import CatBoostRegressor, Pool as RegPool
            from sklearn.metrics import r2_score

            target_col = metric_schema.target_col
            feature_cols = [c for c in metric_schema.feature_cols if
                            c != target_col]
            cat_features = [i for i, c in enumerate(feature_cols) if
                            c in metric_schema.categorical_cols]

            def _fit_and_eval_reg(train_df, test_df):
                train_pool = RegPool(train_df[feature_cols],
                                     train_df[target_col],
                                     cat_features=cat_features)
                test_pool = RegPool(test_df[feature_cols], test_df[target_col],
                                    cat_features=cat_features)
                device = 'GPU' if __import__(
                    'torch').cuda.is_available() else 'CPU'
                model = CatBoostRegressor(random_seed=random_state,
                                          verbose=100, iterations=150,
                                          task_type=device)
                model.fit(train_pool)
                preds = model.predict(test_pool)
                r2 = r2_score(test_df[target_col], preds)
                return {"r2": float(r2)}

            try:
                real_scores = _fit_and_eval_reg(X_train_raw, test_raw)
                synth_scores = _fit_and_eval_reg(synth_raw, test_raw)
                real_score = real_scores["r2"]
                synth_score = synth_scores["r2"]
                pct_diff = abs(real_score - synth_score) / abs(
                    real_score) * 100.0 if real_score != 0 else float('inf')
            except Exception as e:
                print(
                    f"    CatBoost regressor error: {e}. Setting metrics to "
                    f"inf.")
                real_score = float('nan')
                synth_score = float('nan')
                pct_diff = float('inf')
        else:  # classification
            print("Train columns:", X_train_processed.columns.tolist())
            print("Synth columns:", synth_train.columns.tolist())
            print("Missing in synth:",
                  set(X_train_processed.columns) - set(synth_train.columns))
            tstr_scores = tstr_catboost_classifier(
                train_real=X_train_raw,
                test_real=test_raw,
                synth_train=synth_raw,
                schema=metric_schema,
                random_seed=random_state
            )
            real_score = tstr_scores["f1_real"]
            synth_score = tstr_scores["f1_synth"]
            pct_diff = tstr_scores["f1_pct_diff"]

        results["wasserstein"].append(wasserstein)
        results["kl_divergence"].append(kl_div)
        results["corr_frobenius"].append(corr_frob)
        results["tstr_real"].append(real_score)
        results["tstr_synth"].append(synth_score)
        results["tstr_pct_diff"].append(pct_diff)

    if not results["wasserstein"]:
        return None

    avg_results = {
        "Mean WD": np.mean(results["wasserstein"]),
        "Mean KL": np.mean(results["kl_divergence"]),
        "Corr dist": np.mean(results["corr_frobenius"]),
    }
    if task_type == 'regression':
        avg_results["R2_real"] = np.mean(results["tstr_real"])
        avg_results["R2_synth"] = np.mean(results["tstr_synth"])
        avg_results["delta_R2_%"] = np.mean(results["tstr_pct_diff"])
    else:
        avg_results["F1_real"] = np.mean(results["tstr_real"])
        avg_results["F1_synth"] = np.mean(results["tstr_synth"])
        avg_results["delta_F1_%"] = np.mean(results["tstr_pct_diff"])

    # Standard deviations across folds
    avg_results["Std WD"] = np.std(results["wasserstein"], ddof=1) if len(
        results["wasserstein"]) > 1 else float('nan')
    avg_results["Std KL"] = np.std(results["kl_divergence"], ddof=1) if len(
        results["kl_divergence"]) > 1 else float('nan')
    avg_results["Std Corr dist"] = np.std(results["corr_frobenius"],
                                          ddof=1) if len(
        results["corr_frobenius"]) > 1 else float('nan')
    if task_type == 'regression':
        avg_results["Std delta_R2_%"] = np.std(results["tstr_pct_diff"],
                                               ddof=1) if len(
            results["tstr_pct_diff"]) > 1 else float('nan')
    else:
        avg_results["Std delta_F1_%"] = np.std(results["tstr_pct_diff"],
                                               ddof=1) if len(
            results["tstr_pct_diff"]) > 1 else float('nan')

    return avg_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="datasets/raw")
    parser.add_argument("--output_root", type=str,
                        default="experiments/optuna_results")
    parser.add_argument("--encodings", nargs="+", default=DEFAULT_ENCODINGS)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if __import__(
        'torch').cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_datasets", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Folder {raw_dir} not found")

    datasets_info = get_datasets_with_size(raw_dir)
    print(f"Datasets found: {len(datasets_info)}")
    print(datasets_info[["name", "n_rows"]])

    if args.max_datasets:
        datasets_info = datasets_info.head(args.max_datasets)
        print(f"Limiting to first {args.max_datasets} by increasing row count")

    all_summaries = []

    for idx, row in datasets_info.iterrows():
        dataset_path = row["file"]
        dataset_name = row["name"]
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name} ({row['n_rows']} rows)")
        print(f"{'=' * 60}")

        df = pd.read_csv(dataset_path)
        if dataset_name == "openml_PhishingWebsites":
            df["Result"] = df["Result"].replace(-1, 0)
        target_col = get_target_column(df, dataset_name)
        if target_col not in df.columns:
            print(f"Target column {target_col} not found, skipping")
            continue
        schema = TabularSchema.infer_from_dataframe(df, target_col=target_col)

        dataset_results = []

        for enc in args.encodings:
            print(f"\n--- Method: {enc} ---")
            out_csv = Path(f"cv_results_{dataset_name}_{enc}.csv")
            if args.skip_existing and out_csv.exists():
                print(f"  Result already exists: {out_csv}, skipping")
                existing = pd.read_csv(out_csv)
                if not existing.empty:
                    row_vals = existing.iloc[0].to_dict()
                    row_vals["dataset"] = dataset_name
                    row_vals["encoding"] = enc
                    row_vals["n_rows"] = row["n_rows"]
                    all_summaries.append(row_vals)
                continue

            # If the dataset has no categorical features and the method is not
            # the first one, copy the result of any already computed method
            if not schema.categorical_cols:
                any_existing = list(
                    Path.cwd().glob(f"cv_results_{dataset_name}_*.csv"))
                if any_existing:
                    first_csv = any_existing[0]
                    first_metrics = pd.read_csv(first_csv).iloc[0].to_dict()
                    first_metrics["encoding"] = enc
                    pd.DataFrame([first_metrics]).to_csv(out_csv, index=False)
                    print(
                        f"  Copied results from {first_csv.stem} (dataset "
                        f"has no categorical features).")
                    record = first_metrics.copy()
                    record["dataset"] = dataset_name
                    record["n_rows"] = row["n_rows"]
                    all_summaries.append(record)
                    dataset_results.append(record)
                    continue
                else:
                    pass

            tuning_info = ensure_tuning(
                df=df, schema=schema, dataset_name=dataset_name,
                encoding_method=enc, n_trials=args.trials,
                device=args.device, seed=args.seed,
                output_root=Path(args.output_root)
            )
            if tuning_info is None:
                print(f"  Skipping method {enc} due to tuning error")
                continue

            best_params = tuning_info["best_params"]
            task_type = tuning_info["task_type"]

            print(f"  Running cross-validation...")
            metrics = run_cv_for_encoding(
                df=df, schema=schema, encoding_method=enc,
                best_params=best_params, task_type=task_type,
                n_folds=args.n_folds, random_state=args.seed
            )
            if metrics is None:
                print(
                    f"  Skipping method {enc} due to error in "
                    f"cross-validation")
                continue

            print(f"  Results: {metrics}")
            result_df = pd.DataFrame([metrics])
            result_df.to_csv(out_csv, index=False)
            print(f"  Saved to {out_csv}")

            record = metrics.copy()
            record["dataset"] = dataset_name
            record["encoding"] = enc
            record["n_rows"] = row["n_rows"]
            all_summaries.append(record)
            dataset_results.append(record)

        if dataset_results:
            df_dataset = pd.DataFrame(dataset_results)
            df_dataset.to_csv(f"summary_{dataset_name}.csv", index=False)
            print(f"Dataset summary saved to summary_{dataset_name}.csv")

    if all_summaries:
        final_df = pd.DataFrame(all_summaries)
        final_df.to_csv("summary_all.csv", index=False)
        print(f"\nFinal summary table saved to summary_all.csv")
        print(final_df.head())
    else:
        print("No results were obtained.")


if __name__ == "__main__":
    main()
