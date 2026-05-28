# CodeRSA Method Notes

CodeRSA is a content-only reranker for code generation. It does not execute
candidate programs during reranking; execution is used only to evaluate accuracy
after candidates have been selected.

## Pipeline

For each task with original instruction `I0`:

1. Sample candidate programs and keep `n=10` valid candidates per task.
2. Evaluate candidates against benchmark tests and store pass/fail labels.
3. Score the Coder and CoderReviewer baselines.
4. For each candidate `ci`, generate a candidate-induced instruction `Ii` by
   asking the model to describe the behavior implemented by the code.
5. Compute the L0 score matrix:

   ```text
   L0[i, j] = log P(ci | Ij)
   ```

   Column `0` is the original instruction `I0`; columns `1..n` are generated
   instructions aligned to candidates.

6. Compute local pairwise pragmatic contests.
7. Combine pairwise support with global L0 support and select the top
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

The CodeRSA score used for the main results is the fixed equal-weight
combination:

```text
score(ci) = z(pairwise_score(ci)) + z(avg_all_l0(ci))
```

The reported `CodeRSA` row uses this score directly.

## Baselines

The summary table reports:

- Random
- Coder
- CoderReviewer
- Oracle@10
- Avg-all L0
- Pairwise only
- CodeRSA
