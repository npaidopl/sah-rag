# Deterministic Structural Parser

## Status

The deterministic structural parser is implemented as a reproducible structural baseline.

Current version:
```text
0.3.0
```
It is not the final semantic parser of the proposed architecture.

## Purpose

The parser converts immutable canonical documents into parsed-document artifacts containing one or more semantic-parent candidates.
The parser is deliberately conservative. It creates structural sections only when observable heading evidence is available.
When reliable headings are not detected, the complete canonical document remains one parent.

## Input

The parser consumes `CanonicalDocument` records.
Primary text field:
```text
text_normalized
```
The parser never modifies canonical records.

## Output

The parser produces `ParsedDocument` artifacts containing `SemanticParent` records.
Every parent preserves:
- parent_id
- doc_id
- dataset
- normalized character offsets
- raw offsets when raw and normalized text are identical
- parser identity
- parser version
- configuration fingerprint
- boundary provenance
- heading-detection metadata

## Structural Signals

The current parser supports:
- named headings,
- numbered headings,
- uppercase headings,
- document preambles,
- document boundaries.

Recognized named-heading families include:
- chapter
- section
- article
- part
- title
- appendix
- annex
- exhibit
- schedule
- corresponding Greek structural terms

## Rejection Filters

The parser rejects heading candidates that resemble:
- URLs,
- HTML links,
- equations,
- numeric tables,
- addresses,
- bare year rows,
- ordinary sentences,
- oversized blocks,
- blocks with insufficient alphabetic content.

## Grouping Policy

The parser does not group every fixed number of paragraphs.
Its grouping policy is:
1. If no accepted structural heading exists, preserve the full document as one parent.
2. If structural headings exist, create sections beginning at those headings.
3. Preserve text before the first heading as a preamble.
4. Merge structural groups smaller than the configured minimum size.
5. Preserve exact normalized-text slices and offsets.

## Deterministic Identity

Parent identifiers are generated deterministically from:
- dataset ID,
- canonical document ID,
- parser configuration fingerprint,
- parent index.

Repeated parsing of the same canonical document with the same parser configuration produces the same parent IDs.

## Current Limitations

The parser does not infer semantic boundaries from meaning.
It may under-segment documents that:
- have no explicit headings,
- contain extremely large unbroken blocks,
- use unconventional formatting,
- lose structural formatting during upstream extraction.

It may over-segment documents containing:
- repeated uppercase table labels,
- filing headers,
- financial-statement labels,
- repeated document templates,
- short list entries that resemble numbered sections.

The parser does not repair mojibake or other source-text corruption.
Text-quality handling belongs to a separate audit or canonical-data process.

## Initial Cross-Dataset Profile

A diagnostic sample of 20 documents per dataset produced:

| Dataset | Documents | Parents | Median parents/document |
| :--- | :--- | :--- | :--- |
| clerk | 20 | 20 | 1.0 |
| cuad | 20 | 250 | 5.5 |
| greek_bar_bench | 20 | 20 | 1.0 |
| lara |20 | 337 | 3.0 |
| long_bench_v2 | 20 | 70 | 1.0 |
| loong | 20 | 2803 | 126.5 |
| ruler | 20 | 20 | 1.0 |

*Note: These values are diagnostic observations, not final benchmark results.*

## Scientific Role

The deterministic parser will be evaluated as one parsing condition.
Future parser conditions may include:
- canonical-document-only parsing,
- embedding-assisted semantic parsing,
- hybrid structural-semantic parsing,
- optional dataset-aware structural rule sets.

Parser quality must be evaluated through:
- boundary precision, recall, and F1 where annotations exist,
- segmentation metrics,
- over-segmentation and under-segmentation analysis,
- downstream retrieval performance.

## Non-Goals

The deterministic parser does not attempt to:
- produce universal semantic segmentation,
- use BGE-M3,
- generate child retrieval chunks,
- repair source text,
- summarize sections,
- infer a complete document hierarchy,
- replace dataset-specific official parsers in faithful baselines.