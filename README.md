# Evelyn

Evelyn is NeuralMimicry's transformer language system. Its feed-forward knowledge layers run as spiking neuron/synapse populations in the **AARNN** system neural network, which is dynamic and self-evolving, instead of as static weights. Gail serves Evelyn alongside its other LLMs and SNNs.

Evelyn is written entirely in Rust. The staged plan, with a verification gate for each stage, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
cargo test --release -- --nocapture                          # stage 0-2 unit and conversion gates
cargo run --release --bin evelyn-inspect -- model.gguf 0     # architecture, FFN layout, quantisation
cargo run --release --bin evelyn-verify-layer -- model.gguf 0 # stage 2 gate on a real layer
```

Current status: stages 0–4c pass. The Stage 5 OpenAI-compatible shadow
provider is implemented, with its hosted 200-chat, quality, and regression
gate pending. Stage 4c was verified on qwen3:8b across qc04/qc05; see the
stage table and measured latency/quality results in `docs/ARCHITECTURE.md`.

To score increasing numbers of AARNN-served layers across multiple hosts,
create a route manifest as shown in `docs/ARCHITECTURE.md`, then run:

```
cargo run --release --bin evelyn-scale -- model.gguf http://llama-server:8080 eval.txt 256 routes.json 1,2,4
```

The report includes per-layer latency, retries and fallbacks alongside the
perplexity and total runtime latency. It exits non-zero if any sweep changes
perplexity by 5% or more or uses a dense fallback, and reports the fastest
passing multi-host layer count from the measured sweep.

For Stage 5 shadow serving, `evelyn-serve` exposes the runtime at an
OpenAI-compatible `/v1` endpoint. See the architecture document for the route
manifest, tokenizer, API-key, and opt-in Gail profile requirements.

## Container

After CI passes on `main`, the build publishes the amd64/arm64 image as
`ghcr.io/neuralmimicry/evelyn:latest` and a commit-specific tag. The image
contains `evelyn-serve`; mount the GGUF and route manifest at runtime, and pass
the matching tokenizer API URL. Supply `EVELYN_API_KEY` from the deployment's
secret store.

```sh
podman run --rm --name evelyn-qwen35 \
  -p 8080:8080 \
  -e EVELYN_API_KEY \
  -v /srv/models/qwen3.5-9b.gguf:/models/qwen3.5-9b.gguf:ro \
  -v /srv/evelyn/routes.json:/config/routes.json:ro \
  ghcr.io/neuralmimicry/evelyn:latest \
  /models/qwen3.5-9b.gguf http://llama-server:8080 \
  /config/routes.json 0.0.0.0:8080
```

`/healthz` is the container health check. `/v1/models` and chat requests
require the bearer key. Deployments should pin the published digest and supply
model files, route configuration and secrets through their native volume and
secret managers.
