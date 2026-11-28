#!/usr/bin/env python
"""Smoke test for array (embedding) features against a live ProxyML server.

Uses SYNTHETIC data only (random vectors), so nothing of yours is involved. It creates
a uniquely named schema and one surrogate model, exercises the array-feature flow
end to end, checks that no embedding value appears in any request body, and then
deletes what it created.

    PROXYML_API_KEY=... python examples/array_features_smoke_test.py
    PROXYML_BASE_URL=http://localhost:8000/api/v1 ...   # optional: another server

Needs proxyml >= 0.14.0 and a server running alposit >= 0.7.0. Uses about 15 API
calls, up to ~8 more if the counterfactual search keeps missing (the free tier allows
100 a month); it refuses to start with fewer than 30 left.
Exit code 0 = everything passed (warnings are allowed), 1 = at least one failure.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import proxyml
from proxyml import client as sdk
from proxyml_core.schema import ArrayFeature

WIDTH = 16
N_ROWS = 300
MIN_CALLS_REMAINING = 30
MIN_SERVER_VERSION = (0, 7, 0)

_results: list[tuple[str, str, str]] = []


def _record(status: str, name: str, detail: str = "") -> bool:
    _results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return status != "FAIL"


def check(name: str, cond: bool, detail: str = "") -> bool:
    return _record("PASS" if cond else "FAIL", name, detail)


def warn(name: str, detail: str = "") -> None:
    _record("WARN", name, detail)


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())


def _make_data(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    emb = rng.normal(size=(N_ROWS, WIDTH))
    df = pd.DataFrame({
        "age": rng.uniform(18, 80, N_ROWS),
        "income": rng.uniform(1e4, 2e5, N_ROWS),
    })
    df["emb"] = pd.Series([v.tolist() for v in emb], dtype=object)
    return df, emb


def _black_box(frame: pd.DataFrame) -> list[str]:
    """A stand-in model that uses the embedding (its first component) as well as the scalars."""
    return [
        "yes" if a > 40 and i > 8e4 and e[0] > -0.5 else "no"
        for a, i, e in zip(frame["age"], frame["income"], frame["emb"])
    ]


def main() -> int:
    rng = np.random.default_rng(7)
    schema_name = f"smoke_array_{int(time.time())}"
    version: str | None = None
    schema_uploaded = False
    start_remaining: int | None = None

    # Record every request body the SDK sends during the main flow (negative tests are
    # excluded: they deliberately send a vector to prove the server refuses it).
    sent: list[str] = []
    recording = {"on": True}
    real_post, real_put = sdk.post, sdk.put

    def _rec(fn):
        def wrapper(*args, **kwargs):
            payload = kwargs.get("payload")
            if payload is None and len(args) > 1:
                payload = args[1]
            if recording["on"] and payload is not None:
                sent.append(json.dumps(payload, default=str))
            return fn(*args, **kwargs)
        return wrapper

    sdk.post, sdk.put = _rec(real_post), _rec(real_put)

    try:
        print("Connectivity")
        health = proxyml.health_check()
        if not check("server reachable", bool(health)):
            print("  No response: is PROXYML_API_KEY set and PROXYML_BASE_URL right?")
            return 1
        server = health.get("version", "0")
        check(f"server version >= {'.'.join(map(str, MIN_SERVER_VERSION))}",
              _version_tuple(server) >= MIN_SERVER_VERSION, f"server reports {server}")
        usage = proxyml.get_usage() or {}
        remaining = usage.get("calls_remaining")
        start_remaining = remaining
        print(f"  tier={usage.get('tier')} calls_remaining={remaining}")
        if remaining is not None and remaining < MIN_CALLS_REMAINING:
            print(f"  Only {remaining} API calls left this period; need ~{MIN_CALLS_REMAINING}. Aborting.")
            return 1
        if any(r[0] == "FAIL" for r in _results):
            return 1

        df, emb = _make_data(rng)

        print("Schema")
        schema = proxyml.get_schema(df)
        arr = schema.features[2]
        check("get_schema makes an ArrayFeature", isinstance(arr, ArrayFeature) and arr.width == WIDTH and arr.immutable)
        stored = proxyml.put_schema(schema, name=schema_name, n_samples=len(df), source="data")
        schema_uploaded = stored is not None
        check("schema uploaded", schema_uploaded)
        if not schema_uploaded:
            return 1
        fetched = proxyml.fetch_schema(schema_name)
        check("schema fetched back with the array feature",
              fetched is not None and fetched.features[2].type == "array" and fetched.features[2].immutable)

        print("Synthesize")
        glob = proxyml.synthesize_data(400, schema_name=schema_name, array_source=df)
        check("global synthesis fills arrays from local rows",
              glob is not None and glob["emb"].map(lambda v: v is not None and len(v) == WIDTH).all())
        i0 = int(np.argmax(emb[:, 0] > 0.5))
        inst = [35.0, 70000.0, emb[i0]]
        blended = proxyml.synthesize_data(50, sample=inst, schema_name=schema_name)
        check("blended synthesis carries the sample's own array",
              blended is not None and all(np.allclose(v, emb[i0]) for v in blended["emb"]))

        print("Train")
        res = proxyml.train_surrogate(
            samples=glob.values.tolist(), predictions=_black_box(glob),
            feature_names=None, task="classification", schema_name=schema_name,
            name="array smoke test", comments="created by array_features_smoke_test.py; safe to delete",
        )
        check("surrogate trained", res is not None, "training failed" if res is None else f"metrics {res.get('metrics')}")
        if res is None:
            return 1
        version = res["version"]

        print("Explain")
        imp = proxyml.get_feature_importances(version)
        names = {e["feature"] for e in (imp or {}).get("feature_importances", [])}
        check("importances cover only the mutable features", names == {"age", "income"}, f"got {sorted(names)}")
        loc = proxyml.explain_local(inst)
        lnames = {c["feature"] for c in (loc or {}).get("feature_contributions", [])}
        check("local explanation works with the array in the instance", lnames == {"age", "income"}, f"got {sorted(lnames)}")

        cf = None
        for _ in range(8):  # the counterfactual search is stochastic and can miss
            cf = proxyml.find_counterfactual(inst, "yes", perturbation_scale=1.0, n_neighbors=3000)
            if cf is not None:
                break
        if cf is None:
            warn("no counterfactual found in 8 tries", "the search is stochastic; not an array problem by itself")
        else:
            check("counterfactual keeps your embedding", np.allclose(cf.loc[0, "emb"], emb[i0]))

        print("Drift (only a summary is sent)")
        same = proxyml.check_data_drift(version, df.sample(200, random_state=1))
        check("same-distribution data passes", bool(same) and same["passed"], f"status {(same or {}).get('status')}")
        shifted = df.copy()
        shifted["emb"] = pd.Series([(v + 1.5).tolist() for v in emb], dtype=object)
        bad = proxyml.check_data_drift(version, shifted)
        emb_finding = next((f for f in (bad or {}).get("features", []) if f["name"] == "emb"), {})
        check("shifted embeddings are flagged", bool(bad) and not bad["passed"] and emb_finding.get("status") == "drift",
              f"emb finding: {emb_finding.get('status')} {emb_finding.get('metrics')}")

        print("Privacy")
        leaks = [repr(float(x)) for x in emb[:30].ravel() if repr(float(x)) in " ".join(sent)]
        check(f"no embedding value in any of {len(sent)} request bodies", not leaks, f"{len(leaks)} found")

        print("Server refuses vectors (negative tests)")
        recording["on"] = False
        base = {"n": 5, "schema_name": schema_name}
        r = sdk.post(endpoint="/synthesize/blended", payload={**base, "instance": [35.0, 70000.0, emb[i0].tolist()]})
        check("a vector in the array slot is rejected", r.status_code == 422, f"status {r.status_code}")
        r = sdk.post(endpoint="/synthesize/blended", payload={**base, "instance": [35.0, 70000.0, 0.5]})
        check("a scalar in the array slot is rejected with a helpful message",
              r.status_code == 422 and "array feature" in r.text, f"status {r.status_code}")
        r = sdk.post(endpoint="/synthesize/blended", payload={**base, "instance": [None, 70000.0, None]})
        check("a null in a scalar slot is rejected", r.status_code == 422, f"status {r.status_code}")
    finally:
        recording["on"] = False
        sdk.post, sdk.put = real_post, real_put
        print("Cleanup")
        if version is not None:
            check("model deleted", bool(proxyml.delete_model(version)))
        if schema_uploaded:
            check("schema deleted", bool(proxyml.delete_schema(schema_name)))

    end = (proxyml.get_usage() or {}).get("calls_remaining")
    if start_remaining is not None and end is not None:
        print(f"\nAPI calls used by this run: about {start_remaining - end} ({end} left this period)")
    fails = [r for r in _results if r[0] == "FAIL"]
    warns = [r for r in _results if r[0] == "WARN"]
    print(f"\n{len(_results) - len(fails) - len(warns)} passed, {len(warns)} warnings, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
