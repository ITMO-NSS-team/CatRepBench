import unittest
import json
import importlib.util

import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState
from genbench.representations.polynomial.polynomial import PolynomialRepresentation


HAS_CATEGORY_ENCODERS = importlib.util.find_spec("category_encoders") is not None


class BaseRepresentationTest(unittest.TestCase):
    """Base test case for polynomial representation."""

    representation_class = PolynomialRepresentation
    representation_kwargs = {}
    requires_fit_expected = True
    is_invertible_expected = False

    def setUp(self):
        if not HAS_CATEGORY_ENCODERS:
            self.skipTest("category_encoders is not installed")

        self.train_df = pd.DataFrame(
            {
                "color": ["red", "blue", "green", "red"],
                "size": ["small", "medium", "large", "small"],
                "price": [10.0, 15.0, 20.0, 12.0],
            }
        ).astype({"color": "object", "size": "object"})

        self.test_df = pd.DataFrame(
            {
                "color": ["purple", "red"],
                "size": ["tiny", "small"],
                "price": [11.0, 13.0],
            }
        ).astype({"color": "object", "size": "object"})

        self.schema = TabularSchema.infer_from_dataframe(
            self.train_df,
            categorical_cols=["color", "size"],
            continuous_cols=["price"],
        )
        self.rep = self.representation_class(**self.representation_kwargs)

    def test_requires_fit(self):
        self.assertEqual(self.rep.requires_fit(), self.requires_fit_expected)

    def test_is_invertible(self):
        self.assertEqual(self.rep.is_invertible(), self.is_invertible_expected)

    def test_transform_requires_fit(self):
        with self.assertRaises(RuntimeError):
            self.rep.transform(self.train_df)

    def test_fit_and_transform_output_columns(self):
        self.rep.fit(self.train_df, self.schema)
        result = self.rep.transform(self.train_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(any(col.startswith("color_") for col in result.columns))
        self.assertTrue(any(col.startswith("size_") for col in result.columns))

    def test_unknown_category_handling(self):
        self.rep.fit(self.train_df, self.schema)
        transformed = self.rep.transform(self.test_df)
        self.assertIsInstance(transformed, pd.DataFrame)

    def test_get_state_from_state_roundtrip(self):
        self.rep.fit(self.train_df, self.schema)
        state = self.rep.get_state()
        new_rep = self.representation_class.from_state(state)
        self.assertEqual(state.params, new_rep.get_state().params)

    def test_json_roundtrip_state(self):
        self.rep.fit(self.train_df, self.schema)
        state = self.rep.get_state()
        payload = {"name": state.name, "params": state.params}
        payload_json = json.loads(json.dumps(payload))
        restored = self.representation_class.from_state(
            RepresentationState(
                name=payload_json["name"],
                params=payload_json["params"],
            )
        )
        transformed = restored.transform(self.test_df)
        self.assertIsInstance(transformed, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
