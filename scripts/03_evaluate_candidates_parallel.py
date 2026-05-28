import _bootstrap  # noqa: F401

import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

from gorsa_pipeline.core import (
    candidate_passes_mbpp,
    normalize_mbpp_doc,
    read_json,
    task_path,
    write_json,
)
from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config


def make_config(config_dict: dict):
    class Config:
        pass

    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config


def env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in {"0", "false", "False", "no", "NO"}


def prepare_local_eval_root(config) -> tuple[Path, Path, bool]:
    original_root = Path(config.root_dir).resolve()
    if not env_flag("GORSA_EVAL_USE_SCRATCH", "1"):
        return original_root, original_root, False

    scratch_base = Path(os.environ.get("GORSA_EVAL_SCRATCH_ROOT", "/tmp/gorsa_eval_scratch")).resolve()
    scratch_root = scratch_base / original_root.name
    if scratch_root == original_root:
        return original_root, original_root, False

    if env_flag("GORSA_EVAL_RESUME_SCRATCH", "1") and (scratch_root / "tasks").exists():
        config.root_dir = str(scratch_root)
        print("eval original root =", original_root)
        print("eval scratch root =", scratch_root)
        print("resuming existing eval scratch")
        return original_root, scratch_root, True

    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    (scratch_root / "logs" / "eval_task_results").mkdir(parents=True, exist_ok=True)
    shutil.copytree(original_root / "tasks", scratch_root / "tasks")

    run_config = original_root / "run_config.json"
    if run_config.exists():
        shutil.copy2(run_config, scratch_root / "run_config.json")

    existing_results = original_root / "logs" / "eval_task_results"
    if existing_results.exists():
        shutil.copytree(
            existing_results,
            scratch_root / "logs" / "eval_task_results",
            dirs_exist_ok=True,
        )

    config.root_dir = str(scratch_root)
    print("eval original root =", original_root)
    print("eval scratch root =", scratch_root)
    return original_root, scratch_root, True


def writeback_local_eval_root(original_root: Path, scratch_root: Path) -> None:
    original_tasks = original_root / "tasks"
    scratch_tasks = scratch_root / "tasks"
    original_tasks.mkdir(parents=True, exist_ok=True)
    copied_tasks = 0
    for path in scratch_tasks.glob("*.json"):
        shutil.copy2(path, original_tasks / path.name)
        copied_tasks += 1

    original_results = original_root / "logs" / "eval_task_results"
    scratch_results = scratch_root / "logs" / "eval_task_results"
    original_results.mkdir(parents=True, exist_ok=True)
    copied_results = 0
    if scratch_results.exists():
        for path in scratch_results.glob("*.json"):
            shutil.copy2(path, original_results / path.name)
            copied_results += 1

    print("writeback task files =", copied_tasks)
    print("writeback eval result files =", copied_results)


def evaluate_record(config_dict: dict, task_id: str, record: dict) -> tuple[str, int, bool, str | None, list[dict] | None]:
    config = make_config(config_dict)
    candidates = record.get("candidates")
    if candidates is None:
        return task_id, 0, False, "missing candidates", None

    evals = []
    for cand_idx, cand in enumerate(candidates):
        passed, stderr = candidate_passes_mbpp(
            code=cand["exec_code"],
            test_setup_code=record["test_setup_code"],
            test_list=record["test_list"],
            timeout_seconds=config.eval_timeout_seconds,
        )
        evals.append({"candidate_id": cand_idx, "passed": bool(passed), "stderr": stderr})

    return task_id, len(evals), True, None, evals


def evaluate_one_to_file(config_dict: dict, task_id: str, record: dict, result_path: str) -> None:
    result = evaluate_record(config_dict, task_id, record)
    Path(result_path).write_text(json.dumps(result), encoding="utf-8")


def apply_candidate_eval(config, task_id: str, evals: list[dict]) -> tuple[str, int, bool, str | None]:
    path = task_path(config.root_dir, task_id)
    if not Path(path).exists():
        return task_id, 0, False, f"missing task file before write: {path}"

    record = read_json(path)
    if record.get("candidate_eval") is not None and not config.force_rescore:
        return task_id, len(record.get("candidate_eval") or []), False, None

    record["candidate_eval"] = evals
    write_json(record, path)
    return task_id, len(evals), True, None


def build_task_timeout_eval(config, raw_doc: dict, timeout_seconds: int) -> tuple[str, int, bool, str | None, list[dict] | None]:
    doc = normalize_mbpp_doc(raw_doc)
    path = task_path(config.root_dir, doc["task_id"])
    if not Path(path).exists():
        return doc["task_id"], 0, False, f"missing task file after timeout: {path}", None

    record = read_json(path)
    if record.get("candidate_eval") is not None and not config.force_rescore:
        return doc["task_id"], len(record.get("candidate_eval") or []), False, None, None

    candidates = record.get("candidates") or []
    evals = [
        {
            "candidate_id": idx,
            "passed": False,
            "stderr": f"task_eval_timeout_after_{timeout_seconds}s",
        }
        for idx, _cand in enumerate(candidates)
    ]
    return doc["task_id"], len(candidates), True, f"task_eval_timeout_after_{timeout_seconds}s", evals


def main() -> None:
    local_tmp = Path(os.environ.get("GORSA_EVAL_LOCAL_TMPDIR", "/tmp/gorsa_eval_tmp")).resolve()
    local_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(local_tmp)

    config = prepare_config()
    original_root, eval_root, using_scratch = prepare_local_eval_root(config)
    dataset = [dict(raw) for raw in load_dataset_for_config(config)]
    workers = int(os.environ.get("GORSA_EVAL_WORKERS", "4"))
    task_timeout = int(
        os.environ.get(
            "GORSA_EVAL_TASK_TIMEOUT_SECONDS",
            str(max(30, config.eval_timeout_seconds * config.n_candidates + 10)),
        )
    )
    config_dict = config.to_dict()
    result_dir = eval_root / "logs" / "eval_task_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_tmp_dir = local_tmp / "gorsa_eval_task_results" / eval_root.name
    result_tmp_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    timed_out = 0
    errors = []

    ctx = mp.get_context("fork")
    pending = list(dataset)
    active = []

    def start_next() -> None:
        nonlocal skipped
        while pending:
            raw = pending.pop(0)
            doc = normalize_mbpp_doc(raw)
            path = task_path(config.root_dir, doc["task_id"])
            if not Path(path).exists():
                errors.append((doc["task_id"], f"missing task file: {path}"))
                progress.update(1)
                continue

            record = read_json(path)
            if record.get("candidate_eval") is not None and not config.force_rescore:
                skipped += 1
                progress.update(1)
                continue

            result_path = result_tmp_dir / f"{doc['task_id']}.json"
            if result_path.exists():
                result_path.unlink()
            proc = ctx.Process(
                target=evaluate_one_to_file,
                args=(config_dict, doc["task_id"], record, str(result_path)),
            )
            proc.daemon = True
            proc.start()
            active.append(
                {
                    "proc": proc,
                    "raw": raw,
                    "task_id": doc["task_id"],
                    "result_path": result_path,
                    "started": time.monotonic(),
                }
            )
            return

    progress = tqdm(total=len(dataset), desc="Stage 2: evaluate candidates parallel")
    for _ in range(min(workers, len(pending))):
        start_next()

    while active:
        made_progress = False

        for item in list(active):
            proc = item["proc"]
            elapsed = time.monotonic() - item["started"]

            if proc.is_alive() and elapsed <= task_timeout:
                continue

            active.remove(item)
            made_progress = True

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
                if proc.is_alive():
                    proc.kill()
                config_obj = make_config(config_dict)
                task_id, _num, wrote, error, evals = build_task_timeout_eval(
                    config_obj,
                    item["raw"],
                    task_timeout,
                )
                if evals is not None:
                    task_id, _num, wrote, write_error = apply_candidate_eval(config_obj, task_id, evals)
                    error = error or write_error
                timed_out += 1
            else:
                proc.join(timeout=0)
                result_path = item["result_path"]
                if proc.exitcode == 0 and result_path.exists():
                    task_id, _num, wrote, error, evals = json.loads(result_path.read_text(encoding="utf-8"))
                    if evals is not None:
                        config_obj = make_config(config_dict)
                        task_id, _num, wrote, write_error = apply_candidate_eval(config_obj, task_id, evals)
                        error = error or write_error
                    if result_path.exists():
                        shutil.copy2(result_path, result_dir / result_path.name)
                        result_path.unlink()
                else:
                    task_id = item["task_id"]
                    wrote = False
                    error = f"worker_exitcode={proc.exitcode}"

            if error:
                errors.append((task_id, error))
            elif wrote:
                completed += 1
            else:
                skipped += 1

            progress.update(1)
            start_next()

        if not made_progress:
            time.sleep(0.2)

    progress.close()

    print("parallel eval workers =", workers)
    print("per-task timeout seconds =", task_timeout)
    print("tasks evaluated =", completed)
    print("tasks skipped =", skipped)
    print("tasks timed out =", timed_out)
    if errors:
        print("errors =", len(errors))
        for task_id, error in errors[:20]:
            print(f"[ERROR] {task_id}: {error}")
    if using_scratch:
        writeback_local_eval_root(original_root, eval_root)
    print("Stage 2 parallel complete.")
    sys.stdout.flush()
    sys.stderr.flush()
    if env_flag("GORSA_EVAL_FAST_EXIT", "1"):
        os._exit(0)


if __name__ == "__main__":
    main()
