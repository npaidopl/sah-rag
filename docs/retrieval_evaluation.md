# Retrieval Evaluation
 
## Status
 
The project implements typed, checksummed retrieval-evaluation artifacts.
 
Supported metrics include:
- Recall@k
- Precision@k
- reciprocal rank
- nDCG@k
- Hit@k
- macro-averaged aggregate metrics
 
## Evaluation Cases
 
Each evaluation case contains:
- query ID,
- query text,
- retrieval method,
- evaluation level,
- graded relevance judgments,
- ranked retrieved identifiers,
- optional metadata.
 
Evaluation levels may include:

```text
chunk
parent
document
```

Cases from different evaluation levels must not be mixed in one run.

## Relevance Judgments

Each judgment maps one retrievable identifier to a non-negative relevance grade.
Binary relevance uses:

```text
0 = not relevant
1 = relevant
```

Graded relevance may use higher values, for example:

```text
0 = not relevant
1 = partially relevant
2 = highly relevant
```

Recall, Precision, reciprocal rank, and Hit treat every grade greater than zero as relevant.
nDCG preserves graded relevance.

## Recall@k

Recall@k measures the proportion of all relevant items retrieved within the first k positions.

```text
Recall@k = relevant retrieved within k / all relevant items
```

Queries without relevant judgments receive Recall@k of zero.

## Precision@k

Precision@k uses the configured cutoff as a fixed denominator:

```text
Precision@k = relevant retrieved within k / k
```

If fewer than k results are returned, missing ranks are treated as non-relevant.

## Reciprocal Rank

Reciprocal rank is determined by the first relevant result:

```text
RR = 1 / rank of first relevant result
```

If no relevant result occurs within the cutoff, reciprocal rank is zero.
The macro average across queries is mean reciprocal rank.

## nDCG@k

The implementation uses exponential gain:

```text
gain = 2^relevance - 1
```

and logarithmic rank discount.
nDCG@k is the observed DCG divided by the ideal DCG.
Queries without positive relevance receive nDCG of zero.

## Hit@k

Hit@k is binary:

```text
1 = at least one relevant result within k
0 = no relevant result within k
```

## Macro Aggregation

Aggregate metrics are calculated through macro averaging.
Every query contributes equal weight, regardless of:
- number of relevant items,
- document length,
- dataset record size,
- retrieved result count.

Metrics with different:
- retrieval methods,
- evaluation levels,
- cutoffs

cannot be aggregated together.

## Evaluation Artifacts

Each evaluation directory contains:

```text
evaluation/
├── cases.jsonl
├── metrics.jsonl
├── aggregates.jsonl
├── run.json
└── evaluation.manifest.json
```

## Provenance

The evaluation manifest records:
- evaluator identity,
- metric definitions,
- configuration fingerprint,
- source artifact fingerprint,
- evaluation run fingerprint,
- case artifact fingerprint,
- aggregate artifact fingerprint,
- Git revision,
- execution profile,
- record counts.

## Synthetic Validation

The initial synthetic run compared:
- dense retrieval,
- learned-sparse retrieval,
- hybrid RRF retrieval.

It evaluated cutoffs:

```text
1
3
```

The run generated:

```text
Cases: 3
Metric records: 6
Aggregate records: 6
```

The synthetic `hybrid_rrf` condition placed the highest-grade relevant item first.
This is an infrastructure validation only.
It is not evidence that hybrid retrieval is superior on the benchmark datasets.

## Scientific Evaluation Requirements

A scientific result requires:
- benchmark-derived queries,
- benchmark-compatible relevance judgments,
- immutable retrieval predictions,
- fixed retrieval configurations,
- identical query sets across methods,
- explicit handling of queries without positives,
- per-dataset and macro-aggregate reporting,
- confidence intervals or paired significance analysis where appropriate.

## JSON Schemas

The exported schemas are:

```text
schemas/retrieval_evaluation_case.schema.json
schemas/retrieval_metrics.schema.json
schemas/retrieval_aggregate_metrics.schema.json
schemas/retrieval_evaluation_run.schema.json
```

## Non-Goals

The current evaluation layer does not yet:
- map every canonical dataset to relevance judgments,
- evaluate answer generation,
- calculate statistical significance,
- bootstrap confidence intervals,
- select an optimal cutoff,
- make scientific claims from synthetic fixtures.