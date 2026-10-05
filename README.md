# Evelyn

Evelyn is NeuralMimicry's transformer language system. Its feed-forward knowledge layers run as spiking neuron/synapse populations in the **AARNN** system neural network, which is dynamic and self-evolving, instead of as static weights. Gail serves Evelyn alongside its other LLMs and SNNs.

Evelyn is written entirely in Rust. The staged plan, with a verification gate for each stage, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
cargo test --release -- --nocapture                          # stage 0-2 unit and conversion gates
cargo run --release --bin evelyn-inspect -- model.gguf 0     # architecture, FFN layout, quantisation
cargo run --release --bin evelyn-verify-layer -- model.gguf 0 # stage 2 gate on a real layer
```

Current status: stages 0, 1 and 2 pass. See the stage table in `docs/ARCHITECTURE.md`.
