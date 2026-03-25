# tests/test_ctr_model.py

"""Test CTR model pipeline."""

import numpy as np


def test_synthetic_data_generation():
    from adwork.models.ctr_model import CriteoDataLoader

    df = CriteoDataLoader.generate_synthetic(n_rows=1000)
    assert len(df) == 1000
    assert "label" in df.columns
    assert "I1" in df.columns
    assert "C1" in df.columns
    assert df["label"].isin([0, 1]).all()


def test_preprocessor_fit_transform():
    from adwork.models.ctr_model import CriteoDataLoader, CriteoPreprocessor

    df = CriteoDataLoader.generate_synthetic(n_rows=5000)
    preprocessor = CriteoPreprocessor(max_cardinality=100)
    X, y = preprocessor.fit_transform(df)

    assert len(X) == 5000
    assert len(y) == 5000
    assert X.shape[1] == 39  # 13 numeric + 26 categorical
    assert y.dtype == int
    # All categoricals should be integer-encoded
    for col in preprocessor.categorical_feature_names:
        assert X[col].dtype in [int, np.int64, np.int32]


def test_preprocessor_transform_unseen():
    """Unseen categories should map to 0 (rare)."""
    from adwork.models.ctr_model import CriteoDataLoader, CriteoPreprocessor

    df1 = CriteoDataLoader.generate_synthetic(n_rows=5000, random_state=1)
    df2 = CriteoDataLoader.generate_synthetic(n_rows=2000, random_state=99)

    preprocessor = CriteoPreprocessor(max_cardinality=100)
    X1, y1 = preprocessor.fit_transform(df1)

    # Transform with different data — unseen categories should become 0
    X2, y2 = preprocessor.transform(df2)

    assert len(X2) == 2000
    assert y2 is not None
    # No crashes on unseen categories
    for col in preprocessor.categorical_feature_names:
        assert (X2[col] >= 0).all()


def test_full_train_evaluate_pipeline():
    """End-to-end: generate → preprocess → train → evaluate."""
    from sklearn.model_selection import train_test_split

    from adwork.models.ctr_model import CriteoDataLoader, CriteoPreprocessor, CTRModel

    df = CriteoDataLoader.generate_synthetic(n_rows=10_000)
    preprocessor = CriteoPreprocessor(max_cardinality=100)
    X, y = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
    )

    model = CTRModel(params={
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "verbose": -1,
        "n_jobs": 1,
    })

    train_result = model.train(
        X_train, y_train, X_val, y_val,
        categorical_features=preprocessor.categorical_feature_names,
        num_boost_round=50,
        early_stopping_rounds=10,
    )

    assert train_result["best_iteration"] > 0
    assert train_result["best_val_logloss"] > 0

    # Evaluate
    evaluation = model.evaluate(X_test, y_test)

    assert 0.0 < evaluation.auc_roc <= 1.0
    assert evaluation.log_loss > 0
    assert 0.0 <= evaluation.pr_auc <= 1.0
    assert evaluation.calibration_error >= 0
    assert evaluation.test_size == len(X_test)
    assert len(evaluation.feature_importance) > 0


def test_predict_proba_range():
    """Predicted probabilities should be between 0 and 1."""
    from sklearn.model_selection import train_test_split

    from adwork.models.ctr_model import CriteoDataLoader, CriteoPreprocessor, CTRModel

    df = CriteoDataLoader.generate_synthetic(n_rows=5000)
    preprocessor = CriteoPreprocessor(max_cardinality=50)
    X, y = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    model = CTRModel(params={"verbose": -1, "n_jobs": 1, "num_leaves": 15})
    model.train(X_train, y_train, X_test, y_test,
                num_boost_round=20, early_stopping_rounds=10)

    preds = model.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert (preds >= 0).all()
    assert (preds <= 1).all()


def test_model_save_load(tmp_path):
    """Verify model can be saved and loaded."""
    from adwork.models.ctr_model import CriteoDataLoader, CriteoPreprocessor, CTRModel

    df = CriteoDataLoader.generate_synthetic(n_rows=5000)
    preprocessor = CriteoPreprocessor(max_cardinality=50)
    X, y = preprocessor.fit_transform(df)

    model = CTRModel(params={"verbose": -1, "n_jobs": 1, "num_leaves": 15})
    model.train(X, y, X, y, num_boost_round=10, early_stopping_rounds=10)

    preds_before = model.predict_proba(X[:100])

    # Save
    model_path = tmp_path / "test_model.lgb"
    model.save_model(model_path)
    assert model_path.exists()

    # Load into new instance
    loaded = CTRModel()
    loaded.load_model(model_path)
    preds_after = loaded.predict_proba(X[:100])

    np.testing.assert_array_almost_equal(preds_before, preds_after)