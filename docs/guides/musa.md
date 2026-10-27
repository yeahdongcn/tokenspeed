# Running TokenSpeed on MUSA (Moore Threads)

TokenSpeed supports MUSA through the same split it already uses for ROCm and
Ascend: the platform layer decides that Moore Threads' Triton can execute the
portable kernels, and the runtime treats `musa` as one more accelerator device
type. This guide is the source install and the smoke that proves it.

Validated on an MTT S5000 with driver 5.2.0, MUSA 5.2.0, torch/torch_musa
2.11.0, torchada 0.1.95, Triton 3.2.0 and Python 3.10.

## Install

`torch_musa` wheels are built for CPython 3.10, so the environment is Python
3.10 even though TokenSpeed declares 3.11. Every install below therefore passes
`--ignore-requires-python`; nothing on the MUSA path uses a 3.11-only API.

```bash
# 1. kernel package: pure Python on MUSA, no native build and no CUDA toolkit
cd tokenspeed-kernel/python
TOKENSPEED_KERNEL_BACKEND=musa python -m pip install -e . \
    --no-build-isolation --no-deps --ignore-requires-python

# 2. scheduler wheel and the engine's build requirements from the index
python -m pip install "tokenspeed-scheduler" viztracer \
    "nanobind>=2.13.0" "tokenspeed-spdlog" --ignore-requires-python

# 3. the control plane: Rust plus the C++ scheduler it embeds
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
export PATH=/root/.cargo/bin:$PATH
cd tokenspeed-engine
python -m pip install . --no-build-isolation --no-deps --ignore-requires-python

# 4. the Python data plane
cd ../python
python -m pip install -e . --no-build-isolation --no-deps --ignore-requires-python
```

`tokenspeed_kernel/__init__.py` and `tokenspeed/__init__.py` import `torchada`
whenever `torch_musa` is installed, so the CUDA-to-MUSA patches are in place
before anything captures a CUDA symbol. Do not rely on a user-level
`import torchada`: a rank runs inside `ts-host`, which embeds CPython and
starts its own interpreter.

## Platform model

`PlatformInfo.kernel_solutions` lists the kernel source dialects a platform can
execute (`triton`, `torch`, `composite`, ...). A registration written for
another vendor is selectable when its dialect is in that set *and* it claims no
architecture support of its own:

```python
CapabilityRequirement(vendors=frozenset({"amd"}), solution="triton")   # selectable
CapabilityRequirement(vendors=frozenset({"nvidia"}), solution="cute_dsl")  # not
CapabilityRequirement(vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0))
                                                                        # not
```

The reasoning: portable Triton source is the same program on every vendor,
while an arch claim or a vendor-compiler dialect is a statement about one
vendor's hardware. MUSA therefore runs the portable Triton and PyTorch kernels
and nothing else. `TOKENSPEED_KERNEL_VERBOSE=1` logs the ranked selection on
each dispatch; `tokenspeed_kernel.selection.explain_selection()` renders the
same ranking for one operator and signature.

## Launch

The bring-up configuration is eager with the prefill graph disabled; that is
what the smoke below uses. It is not a serving configuration:

```python
from tokenspeed.runtime.entrypoints.engine import Engine

engine = Engine(
    model="Qwen/Qwen3-0.6B",
    dtype="bfloat16",
    attention_backend="triton",
    enforce_eager=True,
    disable_prefill_graph=True,
    attn_tp_size=1,
    gpu_memory_utilization=0.6,
    retraction_snapshot_ratio=0.0,  # the snapshot pool pins host memory
)
print(engine.generate("The capital of France is", {"max_new_tokens": 16}))
engine.shutdown()
```

## Verifying

```bash
python -m pytest tokenspeed-kernel/test/test_musa_platform.py \
    tokenspeed-kernel/test/test_registry.py \
    tokenspeed-kernel/test/test_selection.py -q     # 136 passed
```

A launch is not a pass on its own. Send a request and check the text:
`The capital of France is` → ` Paris. The capital of France is also ...`.

## What is not covered

Greedy and sampled (top-k/top-p) requests work on a dense Qwen3. MoE, MLA,
FP8, GDN/linear attention, multimodal, CUDA graph capture, multi-GPU collectives
(`mccl`) and the HTTP/gRPC gateway have not been validated on MUSA. Kernels for
several of those families are registered and selectable — selection is not
execution.

## Porting notes

Two MUSA facts explain most of the device-type changes in this tree:

- `tensor.is_cuda` is **True** on a torchada-patched MUSA tensor, while
  `tensor.device.type` is `"musa"`. Guards written as
  `device.type == "cuda"` are therefore wrong on MUSA; `is_cuda` guards are not.
- `torch.device("cuda")` resolves to a `musa` device and
  `torch.get_device_module("musa")` works, so generic device-module code needs
  only the extra type.
