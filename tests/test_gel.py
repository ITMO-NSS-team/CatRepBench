import json
import unittest

import numpy as np
import pandas as pd

from genbench.data.schema import TabularSchema
from genbench.representations.base import RepresentationState
from genbench.representations.gel.gel import GELRepresentation


class GELRepresentationTest(unittest.TestCase):
    representation_class = GELRepresentation
    representation_kwargs = {}
    requires_fit_expected = True
    is_invertible_expected = False

    def setUp(self):
        self.train_df = pd.DataFrame(
            {
                "color": ["red", "blue", "green", "red", "blue", "green"],
                "size": ["small", "medium", "large", "small", "medium", "large"],
                "price": [10.0, 15.0, 20.0, 12.0, 14.0, 19.0],
                "target": [0, 0, 1, 0, 0, 1],
            }
        ).astype({"color": "object", "size": "object"})

        self.test_df = pd.DataFrame(
            {
                "color": ["purple", "red"],
                "size": ["tiny", "small"],
                "price": [11.0, 13.0],
                "target": [1, 0],
            }
        ).astype({"color": "object", "size": "object"})

        self.schema = TabularSchema.infer_from_dataframe(
            self.train_df,
            categorical_cols=["color", "size"],
            continuous_cols=["price"],
            target_col="target",
        )
        self.rep = self.representation_class(**self.representation_kwargs)

    def test_requires_fit(self):
        self.assertEqual(self.rep.requires_fit(), self.requires_fit_expected)

    def test_is_invertible(self):
        self.assertEqual(self.rep.is_invertible(), self.is_invertible_expected)

    def test_transform_requires_fit(self):
        with self.assertRaises(RuntimeError):
            self.rep.transform(self.train_df)

    def test_inverse_transform_requires_fit(self):
        with self.assertRaises(RuntimeError):
            self.rep.inverse_transform(self.train_df)

    def test_fit_and_transform_output_columns(self):
        self.rep.fit(self.train_df, self.schema)
        result = self.rep.transform(self.train_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertNotIn("color", result.columns)
        self.assertNotIn("size", result.columns)
        self.assertTrue(any(col.startswith("color__gel_") for col in result.columns))
        self.assertTrue(any(col.startswith("size__gel_") for col in result.columns))

    def test_unknown_category_handling(self):
        self.rep.fit(self.train_df, self.schema)
        transformed = self.rep.transform(self.test_df)

        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertFalse(transformed.filter(like="__gel_").isna().any().any())

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

    def test_supervised_and_unsupervised_modes_produce_different_embeddings(self):
        supervised = self.representation_class(embedding_dim=3, supervision="supervised")
        unsupervised = self.representation_class(embedding_dim=3, supervision="unsupervised")

        supervised.fit(self.train_df, self.schema)
        unsupervised.fit(self.train_df, self.schema)

        supervised_out = supervised.transform(self.train_df)
        unsupervised_out = unsupervised.transform(self.train_df)

        self.assertFalse(
            np.allclose(
                supervised_out.filter(like="__gel_").to_numpy(dtype=float),
                unsupervised_out.filter(like="__gel_").to_numpy(dtype=float),
            )
        )


if __name__ == "__main__":
    unittest.main()

