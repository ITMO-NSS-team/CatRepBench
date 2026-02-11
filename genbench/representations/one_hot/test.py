import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from genbench.data.schema import TabularSchema
from genbench.representations.one_hot.one_hot import OneHotRepresentation


class BaseRepresentationTest(unittest.TestCase):
    """Base test case for all representation classes."""

    representation_class = OneHotRepresentation
    representation_kwargs = {}  # optional constructor parameters
    requires_fit_expected = True
    is_invertible_expected = True

    @classmethod
    def setUpClass(cls):
        """Skip the base class if no representation_class is set."""
        if cls.representation_class is None:
            raise unittest.SkipTest(
                "BaseRepresentationTest should not be run directly")

    def setUp(self):
        """Create synthetic train/test data and a fitted schema."""
        self.train_df = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red'],
            'size': ['small', 'medium', 'large', 'small'],
            'price': [10.0, 15.0, 20.0, 12.0]
        }).astype({'color': 'object', 'size': 'object'})

        self.test_df = pd.DataFrame({
            'color': ['purple', 'red'],
            'size': ['tiny', 'small'],
            'price': [11.0, 13.0]
        }).astype({'color': 'object', 'size': 'object'})

        self.schema = TabularSchema.infer_from_dataframe(
            self.train_df,
            categorical_cols=['color', 'size'],
            continuous_cols=['price']
        )

        self.rep = self.representation_class(**self.representation_kwargs)

    # ----- Interface tests -----
    def test_methods_exist(self):
        """Check that all required methods are implemented."""
        methods = ['fit', 'transform', 'inverse_transform', 'requires_fit',
                   'is_invertible', 'get_state', 'from_state']
        for method in methods:
            self.assertTrue(hasattr(self.rep, method),
                            f"Missing method {method}")

    def test_requires_fit(self):
        """Verify the value of requires_fit()."""
        self.assertEqual(self.rep.requires_fit(), self.requires_fit_expected)

    def test_is_invertible(self):
        """Verify the value of is_invertible()."""
        self.assertEqual(self.rep.is_invertible(), self.is_invertible_expected)

    def test_fit_sets_fitted_flag(self):
        """After fit, the fitted_ attribute must be True."""
        self.rep.fit(self.train_df, self.schema)
        self.assertTrue(self.rep.fitted_)

    def test_transform_requires_fit(self):
        """transform() without fit must raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.rep.transform(self.train_df)

    def test_inverse_transform_requires_fit(self):
        """inverse_transform() without fit must raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.rep.inverse_transform(self.train_df)

    def test_transform_returns_dataframe(self):
        """transform() must return a pandas DataFrame."""
        self.rep.fit(self.train_df, self.schema)
        result = self.rep.transform(self.train_df)
        self.assertIsInstance(result, pd.DataFrame)

    # ----- Serialization tests -----
    def test_get_state_from_state_roundtrip(self):
        """get_state() → from_state() should recover the original object."""
        self.rep.fit(self.train_df, self.schema)
        state = self.rep.get_state()
        new_rep = self.representation_class.from_state(state)
        self.assertEqual(state.params, new_rep.get_state().params)

    # ----- Round‑trip tests (only if invertible) -----
    def test_inverse_transform_roundtrip_on_train(self):
        """For invertible representations, transform + inverse_transform
        should recover the original categorical columns on training data.
        """
        if not self.rep.is_invertible():
            self.skipTest("Representation is not invertible")
        self.rep.fit(self.train_df, self.schema)
        transformed = self.rep.transform(self.train_df)
        recovered = self.rep.inverse_transform(transformed)

        for col in self.schema.categorical_cols:
            self.assertEqual(recovered[col].tolist(),
                             self.train_df[col].tolist())

    # ----- Handling of unknown categories -----
    def test_unknown_category_handling(self):
        """Basic check for handling unseen categories."""
        self.rep.fit(self.train_df, self.schema)
        transformed = self.rep.transform(self.test_df)

        self.assertIsInstance(transformed, pd.DataFrame)

        if self.rep.is_invertible():
            recovered = self.rep.inverse_transform(transformed)
            self.assertEqual(len(recovered), len(self.test_df))


if __name__ == '__main__':
    unittest.main()
