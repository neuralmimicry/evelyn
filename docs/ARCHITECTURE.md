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
| 1 | Generic activation mapping: fit a neuron or population response to any activation (SiLU, GELU, tanh, …) rather than hard-coding, expressed in AARNN-native neuron terms | SwiGLU (true SiLU) rel. error < 5 % on random and real FFN blocks. Stage 0 measured a 27.8 % ReLU-fication gap. | next |
| 2 | Importer for one real FFN layer of a ~9 B open-weights model (candidate: Qwen 3.5 9B, already served by llama.cpp on qc02; also Gemma/Llama 8–9 B), dequantised to f32 | per-layer rel. error < 5 % on activations captured from real prompts | |
| 3 | AARNN knowledge region: instantiate the emitted neuron mesh with AARNN's biomimetic models and automatic detail selection, sharded across nodes; FFN-query API next to AER stimuli; plasticity off by default | AARNN layer output matches stage 2 within 1 % extra error; direct SNN stimuli unaffected | |
| 4 | Evelyn runtime: full model with N converted layers (start with 1, grow); perplexity on a held-out set | ΔPPL vs original < 5 % with 1 layer converted; latency budget recorded; then scale layer by layer | |
| 5 | Gail provider `evelyn/qwen3.5-9b-aarnn` (shadow first) | governed chats 200; quality spot-checks; no regression to other Gail routes | |
| 6 | Continuous learning: AARNN plasticity on the knowledge region, with drift guards and rollback snapshots | no catastrophic-forgetting regressions on a fixed eval set; snapshots restorable | |

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
