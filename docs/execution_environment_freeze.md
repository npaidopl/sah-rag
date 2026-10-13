# Execution Environment and Host Capability Freeze

**Status:** Step 76 freeze
**Host profile ID:** `execution-host-8e2c5a1f3b7d4c9a`

## Scope

Records the actual capability of the execution host, maps each of the
ten architecture slots to one environment status, and freezes the
infrastructure-only pilot gate. No baseline, model, dataset, or
pipeline was executed during collection.

## Host Profile

| Field | Value |
|---|---|
| OS / arch | Windows 10 / AMD64 |
| Python | 3.12.10 |
| CPU | AMD Ryzen 5 5600G |
| Physical CPUs | 6 |
| Logical CPUs | 12 |
| System RAM | 16,508,231,680 bytes (~15.4 GiB) |
| Available RAM | ~6.0 GiB (not frozen; fluctuates) |
| GPU present | AMD Radeon(TM) Graphics (integrated) |
| CUDA-capable GPU | **no** |
| CUDA runtime | not available |
| NVIDIA driver | not present |
| Disk free | 229,171,974,144 bytes (~213 GiB) |
| Disk total | 511,410,073,600 bytes (~476 GiB) |
| Docker CLI | present |
| Docker engine | **not reachable** |
| Git | 2.54.0.windows.1 |

The AMD Radeon integrated GPU is not CUDA-capable and cannot run
PyTorch/vLLM/DeepSpeed/flash-attn workloads. For ML purposes, this
host is effectively CPU-only.

## Collection Safety

The collector used only the Python standard library and read-only
system commands. It did not read serial numbers, usernames, hostnames,
or network addresses. It did not probe the network, load any model,
or read any dataset.

## Network Policy

`ALLOWLISTED_DOWNLOADS` — the default conservative policy. Unrestricted
network access is not granted.

## Hosted-Model and Cost Policy

- Hosted model policy: `APPROVAL_REQUIRED`.
- Maximum approved cost: `EUR 0.00`.
- Secrets available: `false`.

No hosted-model or paid service may be invoked without explicit
approval.

## Environment Status Vocabulary
- EXECUTABLE_ON_CURRENT_HOST
- EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS
- REQUIRES_REMOTE_GPU
- REQUIRES_HOSTED_SERVICE
- REQUIRES_SERVICE_ENVIRONMENT
- REQUIRES_MULTIMODAL_CONTRACT
- NOT_EXECUTABLE

## Architecture Matrix (summary)

| Architecture | Portfolio | Environment | Pilot-eligible |
|---|---|---|---|
| longrag      | READY_FOR_DATA       | EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS | yes |
| raptor       | READY_FOR_DATA       | EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS | yes |
| hipporag2    | READY_FOR_DATA       | REQUIRES_REMOTE_GPU                    | no  |
| lightrag     | READY_FOR_DATA       | EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS | yes |
| pathrag      | READY_FOR_DATA       | EXECUTABLE_ON_CURRENT_HOST             | yes |
| disretrieval | CONDITIONAL_GO       | NOT_EXECUTABLE                         | no  |
| memorag      | CONDITIONAL_GO       | REQUIRES_REMOTE_GPU                    | no  |
| kag          | DEFERRED_WITH_REASON | REQUIRES_SERVICE_ENVIRONMENT           | no  |
| metadata_rag | CONDITIONAL_GO       | NOT_EXECUTABLE                         | no  |
| hikey        | DEFERRED_WITH_REASON | REQUIRES_MULTIMODAL_CONTRACT           | no  |

**4 pilot-eligible** architectures: longrag, raptor, lightrag, pathrag.

## Infrastructure-Pilot Eligibility

Pilot eligibility requires:

1. Portfolio status `READY_FOR_DATA`.
2. Environment status `EXECUTABLE_ON_CURRENT_HOST` or
   `EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS`.
3. Zero mandatory blockers.
4. Generation remains disabled.

## Non-Executable Architectures

- **disretrieval** — foreign execution blocked; no license; clean-room
  implementation not started; requires three isolated CUDA environments.
- **memorag** — no CUDA-capable GPU; memory-model license unresolved;
  retrieval instrumentation not ready.
- **kag** — requires OpenSPG service environment, graph backend, and
  hosted LLM budget.
- **metadata_rag** — project-owned implementation not started.
- **hikey** — multimodal input contract v0.2.0 not designed; no public
  implementation; parser, model licenses, and multimodal GPU unresolved.

## Multimodal Constraints

HiKEY cannot be executed until the cross-baseline envelope is extended
to v0.2.0 with text, table, image, layout, and hierarchy artifacts.
Step 76 records the requirement but does not create v0.2.0.

## Service Constraints

KAG cannot be executed until an approved OpenSPG service environment
and a graph backend (Neo4j or equivalent) are available. The Docker
engine is not reachable on the current host.

## Model-License Constraints

- MemoRAG memory-model weights require a separate license review.
- HiKEY parser and multimodal models require license review.
- DISRetrieval repository has no license at all; foreign execution
  remains blocked regardless of host capability.

## Publishable Execution Gate

**Still disabled.** The blockers recorded in
`baseline_audit/architecture_portfolio_dispositions.yaml` remain in
force. Step 76 does not authorize publishable execution.

## Re-Evaluation Triggers

- A CUDA-capable GPU becomes available.
- The Docker engine becomes reachable.
- Hosted-model policy or cost ceiling changes.
- A specific architecture's blockers are resolved.
- The multimodal contract v0.2.0 is approved.

Each trigger requires a new host profile and a new matrix version.
