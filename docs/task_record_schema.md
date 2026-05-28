# Task Record Schema

Each stage writes back to per-task JSON files under:

```text
$GORSA_ROOT_DIR/tasks/*.json
```

Important fields:

```text
task_id
text
raw_prompt
context_code
function_prompt
entry_point
test
reference_code
generation_prompt
candidates
candidate_eval
coder_logprobs
reviewer_logprobs
prior_logprobs
additional_instructions
l0_logprobs
results_pairwise_avg
```

## Candidate Fields

Each entry in `candidates` contains:

```text
candidate_id
raw_code
exec_code
scoring_code
original_function_name
```

`exec_code` is used for benchmark evaluation. `scoring_code` masks the first
function name to `f` before language-model scoring.

## Additional Instructions

`additional_instructions` stores:

```text
original
generated
all
raw_generated
```

`all[0]` is the original instruction `I0`. `all[1:]` are candidate-induced
instructions aligned to candidate ids.

## L0 Matrix

`l0_logprobs` is a matrix shaped:

```text
num_candidates x num_instructions
```

The first column is `log P(candidate | original instruction)`. Remaining columns
are `log P(candidate | candidate-induced instruction)`.

## Results

`results_pairwise_avg` contains selections and pass/fail labels for:

```text
coder
coderreviewer
orig_only_l0
avg_all_l0
pairwise_only
pairwise_avg_curve
oracle_any
```

The run-level summary is written to `summary_pairwise_avg.json`.
