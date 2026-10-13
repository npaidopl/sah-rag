$content = @'
# BGE-M3 Tokenizer Adapter

## Status

The project includes an offset-preserving tokenizer adapter for:

```text
BAAI/bge-m3
```

The adapter is implemented independently of the embedding model.
Current chunker version:

```text
0.3.0
```

## Purpose

The tokenizer adapter ensures that chunk-size experiments use the same tokenization family as the selected embedding model.
The adapter provides:
- exact character offsets,
- stable tokenizer identity,
- model maximum length,
- sequence-level special-token count,
- compatibility with the parent-aware child chunker.

## Model and Tokenizer Characteristics

The inspected tokenizer has the following properties:

```text
Model ID: BAAI/bge-m3
Tokenizer class: XLMRobertaTokenizer
Fast tokenizer: true
Vocabulary size: 250002
Model maximum length: 8192
Sequence-level special tokens: 2
```

The tokenizer was loaded without PyTorch and without loading model weights.

## Token Budget Definition

The project defines:

```text
chunk_size = total model-input token budget
```

This budget includes sequence-level special tokens.
For BGE-M3:

```text
special_tokens_count = 2
```

Therefore:

| Model-input budget | Content-token capacity |
|-------------------|----------------------|
| 128 | 126 |
| 256 | 254 |
| 512 | 510 |
| 1024 | 1022 |

The overlap is measured in content tokens.

## Rationale

Using total model-input tokens as the experimental chunk-size definition provides a direct and reproducible relationship between:
- configured chunk size,
- tokenizer output,
- model input,
- memory consumption,
- embedding computation.

A 256-token chunk is guaranteed to require no more than 256 model-input tokens after special tokens are added.

## Offset Mapping

The adapter requires a fast Hugging Face tokenizer with:

```text
return_offsets_mapping = true
```

Content tokens are generated with:

```text
add_special_tokens = false
```

This allows every content token to retain a source-text character span.
Special tokens are not represented as source tokens because they do not correspond to canonical text.

## Chunk Creation

For a configured model-input budget of 256 tokens:

```text
content capacity = 256 - 2 = 254
```

The chunker creates windows over at most 254 content tokens.
When passed to the tokenizer with special tokens enabled:

```text
content tokens + 2 special tokens <= 256
```

## Deterministic Identity

The chunker configuration fingerprint includes:
- chunker configuration,
- tokenizer identity,
- tokenizer revision label,
- special-token count,
- effective content-token capacity.

Therefore, chunks created with different tokenizers or token-budget semantics receive different chunk IDs.
Regex-tokenized and BGE-M3-tokenized artifacts cannot silently collide.

## Tokenizer Identity

The initial tokenizer identity is:

```text
huggingface:BAAI/bge-m3@default:fast
```

A future pinned model revision should replace default before final benchmark execution.

## Initial Validation

Five parsed clerk documents produced:

```text
Validated chunks: 8
Unique chunk IDs: 8
Content-token range: 9–254
Model-input range: 11–256
Over-budget chunks: 0
Chunker version: 0.3.0
```

The eight regex-tokenized chunks and eight BGE-M3-tokenized chunks had:

```text
Shared chunk IDs: 0
```

This confirms that tokenizer identity affects artifact identity.

## Tokenization Differences

Observed examples show that BGE-M3 tokenization differs materially from simple regex tokenization.
Examples include subword splitting of:
- English words,
- Greek legal terminology,
- multilingual text,
- retrieval terminology,
- legal citations.

The regex tokenizer remains useful only for:
- unit tests,
- deterministic development tests,
- dependency-free integration tests.

It must not be used as the final benchmark tokenizer for BGE-M3 embeddings.

## Cache Policy

Tokenizer files are stored under the configured external cache:

```text
D:\rag_work\hf_cache
```

The Git repository does not contain:
- tokenizer files,
- model configuration cache,
- embedding model weights,
- downloaded Hugging Face artifacts.

## Model Weight Policy

Tokenizer preparation does not require:
- PyTorch,
- FlagEmbedding,
- BGE-M3 model weights.

Embedding dependencies and model weights are installed in a later stage.

## Current Limitations

The tokenizer identity currently uses:

```text
@default
```

rather than a pinned repository revision.
Before final experiments, the implementation must record:
- exact model repository revision,
- tokenizer revision,
- transformers version,
- tokenizers version,
- tokenizer-file fingerprints.

## Non-Goals

The tokenizer adapter does not:
- generate embeddings,
- load model weights,
- perform dense retrieval,
- produce learned-sparse weights,
- rerank results,
- modify canonical source text.
'@

$dir = "docs"
if (!(Test-Path -Path $dir)) {
    New-Item -ItemType Directory -Path $dir | Out-Null
}

$content | Set-Content -Path "docs\bge_m3_tokenizer_adapter.md" -Encoding utf8
Write-Host "Successfully generated docs\bge_m3_tokenizer_adapter.md" -ForegroundColor Green