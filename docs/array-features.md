# Array features (embeddings and other vectors)

A DataFrame column whose cells are equal-length numeric vectors (a text or image
embedding, a vector of distances to centroids, PCA scores, ...) becomes one
**array feature** in the schema.

This is a deliberately minimal first version. Read "What it does and does not do"
before relying on it.

## Quick start

```python
import pandas as pd
import proxyml

# df["emb"] holds one list / tuple / numpy vector per row (at least 50 rows)
schema = proxyml.get_schema(df.drop(columns=["label"]))   # emb -> ArrayFeature
proxyml.put_schema(schema, name="reviews", n_samples=len(df), source="data")

# Train a surrogate on rows you already have. The embedding column is nulled before
# anything is uploaded, and the surrogate is fit on the other features.
proxyml.train_auto_surrogate(df, target_col="label", schema_name="reviews")

# Explanations hold the embedding fixed at the instance's value.
instance = df.drop(columns=["label"]).iloc[0].tolist()   # the embedding is one element, a list/ndarray
proxyml.explain_local(instance)
proxyml.find_counterfactual(instance, target="positive")    # returned row carries your embedding back

# Data drift: only a summary of the embedding column is sent.
proxyml.check_data_drift(model_id, recent_df)
```

## What leaves your machine

Only a **summary** of each array column is stored: its width, its centroid
(the mean vector) and its rms radius (the typical distance of a row from the
centroid). **Array values are never sent**: in every instance and training sample the
array's slot is sent as `null`, and the server rejects a request that puts a vector
there. The SDK does the nulling for you (any list, tuple or numpy array inside a sample
is replaced by `None`) and puts your local value back into results.

A summary is still derived from your data. The SDK therefore refuses to build one from
fewer than 50 rows, because the centroid of a handful of vectors is close to the
vectors themselves.

## What it does and does not do

- **Array features are always immutable.** The surrogate is fit on the other
  features; the array has no coefficient and no importance. Explanations and
  counterfactuals hold it fixed at the value you pass in. This answers "what would
  change this prediction, other things (including the embedding) being equal", not
  "what in the embedding drives it".
- **Global synthetic data cannot synthesize arrays.** `synthesize_data()` with no
  `sample` returns `None` in array columns. Pass `array_source=df` and each synthetic
  row gets the array of a randomly chosen row of your own DataFrame (never uploaded),
  so your model sees real vectors, but the link between the array and the other
  features is lost. With `sample=`, every row carries that sample's array.
- **Drift is a centroid / spread check.** It reports a shift of the centroid (in units
  of the baseline radius) and a change in the radius, with the same sampling-noise
  allowances as other features. On real sentence embeddings it did not false-alarm
  across 900 random splits and detected a change to a single topic, but it did **not**
  detect a moderate change in topic mix (25% to 50% of one topic). It is not a
  distribution test.
- **Not supported:** variable-length arrays, very sparse high-dimensional vectors,
  matrices/tensors, and arrays with missing rows (clean them first). The dashboard
  and the MCP CSV tools cannot carry vectors: check drift for such models with
  `proxyml.check_data_drift()`.
- **Versions:** a schema containing an array feature can only be read by this version
  or later of `proxyml`, `proxyml-core` and the server.
- The local challenger ignores array columns (`train_challenger` warns and trains on the other features; a schema with only array features raises `ValueError`).

Why this shape: see `proxyml-core/docs/array-features-spec.md`, which records what we
tried for a fuller solution (modelling the array inside the surrogate) and why it was
not good enough yet.
