# Evelyn: architecture and staged delivery

Evelyn is a transformer language system. Its feed-forward ("perceptron")
layers, where a transformer stores most of its factual knowledge, are replaced
by spiking neuron/synapse populations in the AARNN system neural network.
Attention, embeddings and normalisation stay as tensor operations curated by
Evelyn. The knowledge path is AARNN's dynamic, self-evolving network, so Evelyn
keeps learning after import rather than staying a static set of weights.

Everything is Rust. Gail serves Evelyn as one more model (`evelyn/<variant>`)
alongside its LLMs and SNNs.

## Design principles (owner requirements, 2026-10-05)

1. **Distributed and low-latency.** Evelyn and AARNN use every available
   compute node.
   - Converted FFN populations are sharded across AARNN authoritative shards on
     the qc, sm, n1sdp and DeskPi nodes.
   - Layers are pipelined across nodes, and each layer's populations are split
     within it (tensor-parallel style). Hidden states travel over AARNN's AER
     transport.
   - Placement uses estate telemetry (Prometheus, capacity planner,
     estate_balancer). Latency is measured and budgeted at every stage.
2. **AARNN biology is authoritative.** The conversion targets AARNN's own
   biomimetic neuron and synapse models, and its automated
   computational-detail selection, which chooses the level of detail per region.
   - It never replaces them with simplified neurons.
   - The integrate-and-fire neurons in this crate are only a verification proxy
     for the maths.
   - The import emits a description of the neuron mesh (populations, synapses,
     dynamics targets) that AARNN instantiates at whatever detail level it
     selects. Verification then runs against AARNN's real dynamics.
3. **Model-agnostic conversion.** Any open-weights model's perceptron section
   must convert into the equivalent neuron mesh, with no per-model code:
   - architecture discovered from the checkpoint and its config: dense FFN;
     gated FFN (SwiGLU, GeGLU, ReGLU); mixture-of-experts expert banks and
     routers; any activation (ReLU, GELU, SiLU, tanh and others);
   - each activation mapped by fitting a neuron-response or population code to
     it, not by hard-coding;
   - weight formats: safetensors f16/bf16/f32, and GGUF quantisations
     (dequantised);
   - per-layer conversion error reported against the source model.

## Components

| Component | Role | Where |
|---|---|---|
| `evelyn::dense` | Reference ANN blocks: dense, ReLU MLP, SwiGLU MLP. The ground truth for verification. | this crate |
| `evelyn::snn` | ANN→SNN conversion: IF neurons, data-based threshold balancing, coincidence gating, signed E/I channels | this crate |
| importer | Reads open-weights checkpoints (safetensors/GGUF) and extracts per-layer FFN tensors | this crate (stage 2) |
| AARNN knowledge region | Hosts converted FFN populations as neuron/synapse entanglements; serves FFN queries and direct SNN stimuli | aarnn_rust (stage 3) |
| Evelyn runtime | Tokeniser, embeddings, attention, norms; calls AARNN for each FFN; OpenAI-compatible API | this crate (stage 4) |
| Gail provider | `evelyn/*` models routed by Gail, governed by Aria | gail (stage 5) |

AARNN must serve two kinds of request side by side:
- **Direct SNN stimuli:** the existing AER/peripheral-session path.
- **Evelyn FFN queries:** a batch of token hidden states in, the FFN output out.

Both run on the same populations, so learning from either one changes the knowledge.

## Conversion method

- **ReLU units → IF neurons:** reset by subtraction. The threshold is the
  99.9th-percentile calibration activation. Rate × threshold converges to ReLU,
  clipped at the threshold.
- **Readout:** the output layer is a non-spiking integrator (membrane readout),
  so signed outputs need no spike encoding.
- **Gated FFN** (`down(act(gate x) ⊙ up x)`, as in Llama/Qwen/Gemma):
  - gate and up neurons fire stochastically;
  - a coincidence-detecting unit's rate is the product of the two rates;
  - signed up-projections use excitatory and inhibitory channel pairs.

## Stages and verification gates

Each stage ships only when its gate passes. Results are recorded here.

| Stage | Deliverable | Gate | Status |
|---|---|---|---|
| 0 | Conversion core (ReLU MLP, coincidence-gated FFN), deterministic tests | ReLU MLP rel. error < 5 % at T=2048; gated < 10 % at T=4096; error falls with T | **PASS 2026-10-05**: 3.9 %, 5.0 % |
| 1 | Generic activation mapping: fit a neuron or population response to any activation (SiLU, GELU, tanh, …) rather than hard-coding, expressed in AARNN-native neuron terms | SwiGLU (true SiLU) rel. error < 5 % on random and real FFN blocks | **PASS 2026-10-05 (random blocks)**: true SwiGLU 1.40 %, GeGLU 1.37 % (stage-0 ReLU-fication gap 25 %); activation fits ReLU/SiLU/GELU/tanh/sigmoid/custom < 0.003 max error with 25–49 units. Real-model blocks are checked in stage 2. |
| 2 | Importer for one real FFN layer of a ~9 B open-weights model (candidate: Qwen 3.5 9B, already served by llama.cpp on qc02; also Gemma/Llama 8–9 B), dequantised to f32 | per-layer rel. error < 5 % on activations captured from real prompts | **PASS 2026-10-05 (proxy inputs)**: Qwen 3.5 9B (`qwen35`, Q4_K/Q6_K) layers 0/16/31: spiking 2.74 % / 2.00 % / 1.42 %, analog population 0.37 % / 0.35 % / 0.21 %. Inputs were real token embeddings RMS-normalised with each layer's own norm; captured mid-network activations follow in stage 4. |
| 3a | Fit population codes against AARNN's *measured* transfer curves (its own kernels, with membrane noise), with automatic population sizing per neuron model and detail depth | every activation < 0.01 max error on measured AARNN neurons | **PASS 2026-10-05**: LIF 0.007–0.009 (65–97 units); Izhikevich RS depth 0 and 2: 0.008 (129–385 units) |
| 3b | AARNN knowledge region: instantiate the emitted neuron mesh with AARNN's biomimetic models and automatic detail selection, sharded across nodes; FFN-query API next to AER stimuli; plasticity off by default | AARNN layer output matches stage 2 within 1 % extra error; direct SNN stimuli unaffected | **PASS 2026-10-05**: Qwen 3.5 9B layer 0 executed by AARNN (`aarnn-knowledge-run`, real LIF kernels, membrane noise, 40 shards): **1.84 %** vs exact layer (gate 3.74 %); spiking noise 0.78 %; analog mesh 1.57 %. 1.38 M neurons, 6.1 G neuron-steps, 7.5 s per token per layer on 40 qc02 cores. AARNN lib suite: see aarnn_rust#32 |
| 4a | Pure-Rust transformer runtime (GGUF, quantised in memory, parallel, KV cache, pluggable FFN) | greedy output identical to llama.cpp | **PASS 2026-10-05**: qwen3:8b, 24/24 tokens identical; PPL 2.874; 1.14 tok/s on qc02 |
| 4b | One real layer's FFN served live by AARNN (network service) inside the runtime, calibrated on real activations | ΔPPL < 5 %, no fallbacks | **PASS 2026-10-05**: qwen3:8b layer 18 on AARNN LIF: PPL 4.0337 → 4.0332 (−0.01 %), top-1 agreement 100 %, 0 fallbacks; 7.2 s/token (vs 0.9 dense) |
| 4c | Scale to many layers across estate nodes; minimize measured latency | ΔPPL < 5 % with N layers; latency trend; zero dense fallbacks | **PASS 2026-10-05**: qwen3:8b real-activation LIF layers 18–21 served across qc04/qc05. The lowest complete passing budget tested was 119 neuron steps / 11 warm-up steps: N=1/2/4 all pass; N=2 spans both hosts at +4.288 % ΔPPL, 100 % top-1, 1.313 s/token and zero retries/fallbacks on 32 README tokens. A 64-token confirmation passes at +3.755 % ΔPPL, 98.44 % top-1 and 1.311 s/token. Lower budgets 118, 117, 112, 100 and 62 fail the N=2 quality gate. Full evidence is in `swarmhpc/docs/evidence/evelyn-stage4c-20261005-*`. |
| 5 | Gail provider `evelyn/qwen3.5-9b-aarnn` (shadow first) | governed chats 200; quality spot-checks; no regression to other Gail routes | **IMPLEMENTATION IN PROGRESS 2026-10-05**: authenticated OpenAI-compatible `evelyn-serve`, multi-host AARNN route validation, and opt-in explicit-only Gail profile are implemented; the hosted 200-chat/quality/regression gate remains pending |
| 6 | Continuous learning: AARNN plasticity on the knowledge region, with drift guards and rollback snapshots | no catastrophic-forgetting regressions on a fixed eval set; snapshots restorable | |

### Stage 4c sweep tool

`evelyn-scale` reads a versioned JSON route manifest mapping model layer indexes
to `host:port` AARNN knowledge-region endpoints, then evaluates increasing
layer counts against the same teacher-forced text. Its largest sweep must use
at least two distinct hosts. The report includes perplexity delta, top-1
agreement, whole-runtime seconds per token, and each remote layer's calls,
mean/max latency, retries and failures. It exits non-zero unless every sweep
stays within an absolute 5% perplexity delta with zero dense fallbacks. It
reports the measured latency trend at each layer count so placements can be
ranked by the lowest observed latency; there is no arbitrary ceiling.

Example manifest (replace the endpoints with the deployed AARNN service
addresses):

```json
{
  "schema_version": 1,
  "routes": {
    "12": "aarnn-qc02.example:38112",
    "18": "aarnn-qc03.example:38118",
    "24": "aarnn-sm01.example:38124",
    "30": "aarnn-n1sdp.example:38130"
  }
}
```

Run the requested increasing layer counts with:

```sh
cargo run --release --bin evelyn-scale -- model.gguf http://llama-server:8080 eval.txt 256 routes.json 1,2,4
```

The executable provides the sweep and evidence format; it does not provision
knowledge regions. **Stage 4c passed on 2026-10-05.** On qc04/qc05, the 119-step
N=1/2/4 sweep passed with zero fallbacks; the fastest multi-host point was
N=2 at 1.313 s/token. Its 64-token confirmation passed at 1.311 s/token.
Lower budgets were measured through 62 steps; 119/11 was the lowest complete
passing configuration among the tested counts. The dense baseline measured
0.595 s/token on 32 tokens and 0.642 s/token on 64 tokens, so the AARNN-backed multi-host path remains slower than dense; 119/11 is the fastest passing SNN configuration measured, and reducing that gap remains future performance work.

Mesh runtime is a second latency variable. `evelyn-layer-mesh` accepts an
optional final `steps` argument (default 4000) and sets warm-up to one tenth
of that count. Re-export the same routed layers at successively lower step
counts, then run the same text and layer-count sweep; retain the lowest
measured latency only while every perplexity and fallback gate still passes.
This makes the biological simulation budget evidence-driven rather than
assuming the stage-3b 4000-step setting is optimal for transformer quality.

```sh
cargo run --release --bin evelyn-layer-mesh -- model.gguf 18 curves.json lif mesh-layer18 http://llama-server:8080 eval.txt 256 2.0 1000
```

### Stage 5 shadow provider

`evelyn-serve` exposes the Rust runtime through Gail's existing OpenAI-compatible
provider adapter. It accepts text-only Qwen chat completions, tokenises and
detokenises through the matching llama.cpp tokenizer endpoint, and routes the
configured FFN layers to the AARNN endpoints in a version-1 route manifest.
The tokenizer parses Qwen special tokens, and the AARNN TCP connections remain
open between requests. The manifest must use at least two distinct hosts. The
service requires `EVELYN_API_KEY` with at least 32 bytes; health is available
at `/healthz`, while `/v1/models` and `/v1/chat/completions` require the bearer
key. If any AARNN layer falls back to dense weights, Evelyn rejects that
completion with HTTP 503 so Gail never receives a response that silently
bypassed AARNN. Greedy and temperature sampling are supported; text-only
JSON-object mode is validated before a response is returned.

```sh
EVELYN_API_KEY="$EVELYN_API_KEY" cargo run --release --bin evelyn-serve -- \
  qwen3.5-9b.gguf http://llama-server:8080 docs/evidence/qwen35-routes.json 0.0.0.0:8080
```

Gail's standard `openai` adapter can address this as
`evelyn/qwen3.5-9b-aarnn`. In the SwarmHPC Ansible role, the profile is
optional and appears only when `EVELYN_AARNN_BASE_URL` is set. It has the
dedicated `evelyn_shadow` role and zero candidate weight, so normal `gail-auto`
selection and existing routes are unchanged; governed evaluation requests
select the explicit Evelyn model alias through Gail. The shared bearer secret
is supplied as `EVELYN_API_KEY` to both services.

**Stage 5 is not yet passed.** Software tests cover authentication, the Qwen
text-chat template and system instructions, greedy sampling, rejection of
unsupported multimodal/tool input, and the two-host route requirement. The 9B
Qwen 3.5 mesh endpoints must be staged and the opt-in Gail profile enabled
before collecting 200 governed shadow chats, quality spot-checks, and
regression evidence for existing Gail routes.

## Stage 1 method

Each activation is a *heterogeneous-threshold population*:
`f(z) ~= c + sum a_k relu(z - b_k) + sum d_k relu(b_k - z)`.
- **Units:** each unit is a rate-coded neuron with threshold `b_k`. Its
  signed weight is an excitatory or inhibitory synapse. `c` is a tonic unit.
- **Placement:** thresholds are curvature-adaptive, equidistributed in
  `sqrt(|f''|)` with `f''` estimated numerically, so any activation works.
- **Fit:** weights come from a ridge least-squares fit, then sparse pruning.
- **Gated FFNs:** `act(gate) * up` distributes over the population sum, so
  every unit gets its own coincidence detector with the up neuron. The true
  SiLU/GELU gate converts directly, with no ReLU-fication or fine-tuning.
- **Biology:** the result is the population and threshold-diversity code that
  real neural populations use. AARNN instantiates these units with its own
  neuron models (principle 2).

## Stage 2 method and findings

- **Importer** (`gguf.rs`, `import.rs`): a pure-Rust GGUF v2/v3 reader. It
  reads only the tensors it needs and dequantises F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K.
  The FFN layout comes from tensor names, so there is no per-model code.
  Activation comes from metadata, or failing that the family default.
- **Saturation dominates real layers, not fitting.** With ranges set from 64
  samples at the 99.9th percentile:
  - the fit alone was 0.3–0.7 %;
  - gate and up-neuron clipping took the error to 11 %, because LLM
    activations are heavy-tailed.

  Calibrating 512 samples at the maximum with 2× headroom (gain control)
  gives 0.37 % analog error.
- **Dithered spike timing.** Low-discrepancy, regular-spiking streams with
  rationally independent phase increments reduce coincidence sampling error
  roughly as 1/T, against 1/sqrt(T) for random spikes. At T=1024 that is
  6.0 % against 25.4 %, so the same accuracy needs far fewer timesteps and
  latency stays low.
- **Performance:** calibration and verification run on every core with
  scoped threads. Conversion of one layer takes 2.5 s on qc02, down from 61 s
  single-threaded.
- **Reproduce:** `cargo run --release --bin evelyn-verify-layer -- <model.gguf> <layer>`
  (the defaults are the recommended settings). It exits non-zero on a gate
  failure.
- **Cost signal for stage 3:** about 33–39 M spikes per token per layer at
  T=4096. Sharding across nodes and lowering T, with tighter ranges and
  AARNN's detail selection, are the latency levers.

## Stage 3a method and findings

- **AARNN measures, Evelyn fits** (modular; nothing is duplicated). aarnn_rust's
  `knowledge` module drives its own LIF and Izhikevich/AARNN kernels and
  exports transfer curves (`aarnn-knowledge-curves`). Evelyn fits against
  them as data, with no code dependency on AARNN.
- **Discrete-time staircase.** A constant drive gives an integer firing
  period, so a noiseless AARNN neuron's f-I curve is a staircase (1/6, 1/7,
  ...), with a narrow dynamic range: LIF rheobase 0.049, saturation 0.19.
  Biological membrane noise (20 % of the rheobase-to-saturation span) grades
  it into a smooth response. Knowledge regions must run with the same noise.
- **Saturating units need even spacing.** Shifted copies of one saturating
  response sum to a near-exact line, so thresholds are uniform here, unlike
  the curvature-adaptive placement used for ideal rectifiers. The fit uses a
  15 % margin beyond the range (no edge error), a floor on response width, a
  dense grid and light ridge regularisation.
- **Size cost.** Richer AARNN dynamics (Izhikevich) need about 4x the
  neurons of LIF for the same accuracy. In 3b, AARNN's detail selection
  should therefore choose the model per region, balancing fidelity against
  latency.

## Stage 3b method and findings

- **Mesh description** (`mesh.rs` → `aarnn_rust::knowledge_region::FfnMesh`):
  - input and readout synapse matrices as raw f32 files;
  - gate and up population codes;
  - per-channel up synaptic scales;
  - the exact AARNN neuron spec, membrane noise, steps and warm-up.
- **Execution in AARNN:**
  1. dendritic summation;
  2. each active unit is a real AARNN neuron simulated with noise;
  3. its rate is decoded through its output synapse;
  4. active dendritic multiplication of gate and up;
  5. synaptic readout.

  Hidden channels are sharded in parallel, and noise streams are seeded per
  neuron, so runs are reproducible in any shard layout.
- **Per-channel synaptic scaling was essential.** A single up range shared
  across channels gave 24 % analog error, because small channels were
  swamped. Normalising each channel by its own calibrated range, so that one
  identity code serves every channel, brought it to 1.57 %. Gate tolerances
  must be absolute: most gate currents sit near zero.
- **Latency is now the binding constraint:** 7.5 s per token per layer on
  one 46-core node. Levers for stage 4:
  - shard each layer across nodes (32 layers across the estate's qc, sm and
    n1sdp nodes);
  - fewer steps per token, using the dithered timing of stage 2 and shorter
    warm-up;
  - skip silent units (already done);
  - cheaper neuron models for knowledge regions, chosen by detail selection;
  - GPU kernels.

## Known risks

- **Throughput:** a 9 B model's FFNs hold about 6 B weights. Spiking simulation
  needs T timesteps per token, so cost per token is far above llama.cpp. The
  plan converts layers progressively and measures latency at every step.
  Hybrid operation is the default: unconverted layers stay tensor-based.
- **Quantisation:** GGUF weights must be dequantised. Conversion error compounds
  with quantisation error, so it is measured against f32 references.
- **Memory:** about 6 B synapses at f32 is roughly 24 GB per copy. The estate's
  largest nodes have 64–96 GB, so the AARNN region must be sharded across nodes;
  aarnn_rust has authoritative shards.
- **Learning safety:** plasticity is off until stage 6, and then gated by evals
  and snapshots. This is in line with the estate guard rails (Aria governance,
  verify-or-rollback).
