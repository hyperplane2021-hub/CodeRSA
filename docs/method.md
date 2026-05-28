# CodeRSA Method Notes

CodeRSA is a content-only reranker for code generation. It does not execute
candidate programs during reranking; execution is used only to evaluate accuracy
after candidates have been selected.

## Pipeline

For each task with original instruction `I0`:

1. Sample candidate programs `c1, ..., cn`.
2. Evaluate candidates against benchmark tests and store pass/fail labels.
3. Score Coder and CoderReviewer baselines.
4. For each candidate `ci`, generate a candidate-induced instruction `Ii` by
   asking the model to describe the actual behavior implemented by the code.
5. Compute an L0 score matrix:

   ```text
   L0[i, j] = log P(ci | Ij)
   ```

   Column `0` is the original instruction `I0`; columns `1..n` are generated
   instructions aligned to candidates.

6. Compute local pairwise pragmatic contests.
7. Combine pairwise support with global average support and select the top
   candidate.

## Pairwise Contest

For each pair of candidates `(ci, cj)`, CodeRSA compares:

```text
m_i = L0(ci | I0) - L0(ci | Ij)
m_j = L0(cj | I0) - L0(cj | Ii)
```

If `m_i > m_j`, candidate `ci` wins the contest. If `m_j > m_i`, candidate `cj`
wins. Ties split the point. The pairwise score is the candidate's win rate over
all valid pairwise contests.

## Global Support

The global support score is the mean L0 score for a candidate across the local
instruction neighborhood:

```text
avg_all_l0(ci) = mean_j L0(ci | Ij)
```

## Final Score

The final sweep score is:

```text
score_lambda(ci) = z(pairwise_score(ci)) + lambda * z(avg_all_l0(ci))
```

The scripts sweep `lambda` from `0.0` to `3.0` in increments of `0.1` and report
both `lambda=1` and the best lambda on the run.

## Baselines in This Repo

This repository focuses on:

- Random
- Coder
- CoderReviewer
- Avg-all L0
- Pairwise only
- Pairwise + Avg
- optional lightweight CodeT script

Consensus-WUCS is intentionally not part of this reproduction package.
