/**
 * ProxyML basic usage — TypeScript version
 *
 * Prerequisites: Node 18+ (uses native fetch)
 *   npx tsx basic_usage.ts
 *   # or: bun run basic_usage.ts
 *
 * Build your schema at https://proxyml.github.io/schema-builder/
 * then paste it into step 1 below.
 *
 *   export PROXYML_API_KEY="your-api-key"
 */

const BASE_URL = process.env.PROXYML_BASE_URL ?? "https://api.proxyml.ai/api/v1";
const API_KEY  = process.env.PROXYML_API_KEY  ?? "";

if (!API_KEY) throw new Error("PROXYML_API_KEY is not set");

const headers = {
  "Content-Type": "application/json",
  "X-API-KEY": API_KEY,
};

async function api<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${path} → ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// 1. Upload a schema
//    Replace the features array with the JSON from the schema builder.
// ---------------------------------------------------------------------------
const schema = {
  features: [
    { type: "continuous", name: "age",         mean: 45,    std: 15,    min: 18,  max: 90,     immutable: false },
    { type: "continuous", name: "income",       mean: 60000, std: 25000, min: 0,   max: 200000, immutable: false },
    { type: "continuous", name: "credit_score", mean: 680,   std: 80,    min: 300, max: 850,    immutable: false },
  ],
};

const schemaResult = await api("PUT", "/schema/default", schema);
console.log("Schema upload:", schemaResult);

// ---------------------------------------------------------------------------
// 2. Synthesize training data
// ---------------------------------------------------------------------------
const synth = await api<{ samples: number[][]; feature_names: string[]; feature_types: string[] }>(
  "POST", "/synthesize/neighbors", { n: 500, schema_name: "default" }
);
console.log(`Synthesized ${synth.samples.length} rows, features: ${synth.feature_names}`);

// ---------------------------------------------------------------------------
// 3. Score with your black-box model
//    Replace this function with calls to your actual model.
//    It receives one sample (values in feature_names order) and returns a label.
// ---------------------------------------------------------------------------
const creditScoreIdx = synth.feature_names.indexOf("credit_score");

function blackBox(sample: number[]): number {
  return sample[creditScoreIdx] > 650 ? 1 : 0;
}

const predictions = synth.samples.map(blackBox);
const labelCounts = predictions.reduce<Record<number, number>>((acc, p) => {
  acc[p] = (acc[p] ?? 0) + 1;
  return acc;
}, {});
console.log("Label distribution:", labelCounts);

// ---------------------------------------------------------------------------
// 4. Train a surrogate
// ---------------------------------------------------------------------------
const trainResult = await api<{ version: string }>("POST", "/surrogate/train", {
  samples:       synth.samples,
  predictions,
  feature_names: synth.feature_names,
  task:          "classification",
  test_size:     0.2,
  schema_name:   "default",
});
console.log("Surrogate training result:", trainResult);

// ---------------------------------------------------------------------------
// 5. Feature importances
// ---------------------------------------------------------------------------
const importances = await api("GET", `/explain/importance?version=${trainResult.version}`);
console.log("Feature importances:", importances);

// ---------------------------------------------------------------------------
// 6. Query the surrogate
// ---------------------------------------------------------------------------
const sample = [50, 75000, 720]; // age=50, income=75000, credit_score=720
const prediction = await api("POST", "/surrogate/predict", {
  inputs:  sample,
  version: trainResult.version,
});
console.log("Surrogate prediction:", prediction);
