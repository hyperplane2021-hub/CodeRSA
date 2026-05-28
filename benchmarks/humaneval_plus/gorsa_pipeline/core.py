"""Core functions extracted from GoRSA_Human+.ipynb.

This module intentionally keeps the notebook's HumanEval+ pairwise pipeline semantics.
"""
import json
import hashlib
import math
import os
import random
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import ast
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import copy

# ============================
# Generic utilities
# ============================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def maybe_load_json(path: str | Path) -> Optional[dict]:
    path = Path(path)
    if path.exists():
        return read_json(path)
    return None


def logmeanexp(log_values: Sequence[float]) -> float:
    arr = np.asarray(log_values, dtype=np.float64)
    if arr.size == 0:
        return -1e30
    m = float(np.max(arr))
    return float(m + np.log(np.mean(np.exp(arr - m))))


def softmax_np(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    probs /= probs.sum()
    return probs






# ============================
# Config
# ============================

@dataclass
class ReproConfig:
    root_dir: str
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    seed: int = 13
    limit: Optional[int] = None
    n_candidates: int = 10
    candidate_temperature: float = 1.0
    candidate_top_p: float = 1.0
    candidate_max_new_tokens: int = 256
    additional_instruction_temperature: float = 0.7
    additional_instruction_top_p: float = 1.0
    additional_instruction_max_new_tokens: int = 96
    alpha_values: Tuple[float, ...] = tuple(np.round(np.arange(0.0, 1.51, 0.05), 2).tolist())
    score_batch_size: int = 8
    generation_batch_size: int = 8
    eval_timeout_seconds: int = 8
    force_regenerate_candidates: bool = True
    force_rescore: bool = True
    force_recompute_instructions: bool = True
    force_recompute_l0: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alpha_values"] = list(self.alpha_values)
        return d


# ============================
# Dataset and task files
# ============================





# ============================
# Prompts
# ============================

CODE_STOP_WORDS = [
    "\nassert ",
    "\ndef test_",
    "\nimport unittest",
    "\nfrom unittest",
    "\n# write unit test",
    "\n# test",
    "\nif __name__",
    "\nprint(",
    "\nclass ",
    "\n```",
    "\n<|/",
    "Human",
]





def build_coder_prefix(original_instruction: str) -> str:
    return original_instruction.rstrip() + "\n"


def build_reviewer_prefix(code: str) -> str:
    # Matches the appendix example style from the paper.
    return code.rstrip() + "\n\n# Write a docstring for the above function\n"


def build_additional_instruction_prompt(code: str) -> str:
    func_clean = clean_code_generation(code)
    func_clean = mask_first_function_name_ast(func_clean, new_name="f")

    fewshot = textwrap.dedent("""\
    Read each Python function and describe exactly what behavior it implements.

    Rules:
    - Output exactly one sentence.
    - Describe the function's actual input-output behavior only.
    - Do not guess the intended task beyond what the code really does.
    - Mention returned type, deduplication, ordering, and special return values when relevant.
    - Do not write code.
    - Do not write explanations.
    - Do not use markdown.

    Function:
    def f(xs):
        return list(set(xs))
    Description:
    Return a list of the unique elements from the input list.

    Function:
    def f(xs):
        return sorted(set(xs))
    Description:
    Return a sorted list of the distinct elements in the input list.

    Function:
    def f(xs, x):
        for i, v in enumerate(xs):
            if v == x:
                return i
        return -1
    Description:
    Return the index of the first occurrence of x in the list, or -1 if x is not present.

    Function:
    """)

    return fewshot + func_clean + "\nDescription:\n"


def normalize_instruction_text(text: str) -> str:
    text = text.strip()
    text = text.replace("```", "")
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================
# Tokenization / scoring
# ============================

def _build_full_ids(tokenizer, prefix_text: str, suffix_text: str) -> Tuple[List[int], int]:
    prefix_ids = tokenizer(prefix_text, add_special_tokens=True).input_ids
    full_ids = tokenizer(prefix_text + suffix_text, add_special_tokens=True).input_ids
    prefix_len = len(prefix_ids)
    return full_ids, prefix_len


@torch.inference_mode()
def sum_logprobs_texts(
    model,
    tokenizer,
    prefix_texts: Sequence[str],
    suffix_texts: Sequence[str],
    batch_size: int = 8,
    progress_desc: Optional[str] = None,
) -> List[float]:
    assert len(prefix_texts) == len(suffix_texts)

    examples = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for prefix, suffix in zip(prefix_texts, suffix_texts):
        full_ids, prefix_len = _build_full_ids(tokenizer, prefix, suffix)
        if prefix_len >= len(full_ids):
            examples.append((torch.tensor(full_ids, dtype=torch.long), prefix_len, True))
        else:
            examples.append((torch.tensor(full_ids, dtype=torch.long), prefix_len, False))

    all_scores: List[float] = []
    iterator = range(0, len(examples), batch_size)
    if progress_desc:
        iterator = tqdm(iterator, desc=progress_desc)

    device = model.device
    for start in iterator:
        chunk = examples[start : start + batch_size]
        max_len = max(len(item[0]) for item in chunk)

        input_ids = []
        attention_mask = []
        target_mask = []

        for ids, prefix_len, empty_suffix in chunk:
            padded = torch.full((max_len,), pad_id, dtype=torch.long)
            padded[: len(ids)] = ids

            attn = torch.zeros((max_len,), dtype=torch.long)
            attn[: len(ids)] = 1

            tgt = torch.zeros((max_len - 1,), dtype=torch.float32)
            if not empty_suffix:
                start_j = prefix_len - 1
                tgt[start_j : len(ids) - 1] = 1.0

            input_ids.append(padded)
            attention_mask.append(attn)
            target_mask.append(tgt)

        input_ids = torch.stack(input_ids).to(device)
        attention_mask = torch.stack(attention_mask).to(device)
        target_mask = torch.stack(target_mask).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        valid = attention_mask[:, 1:].float() * target_mask

        token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        summed = (token_logprobs * valid).sum(dim=-1)
        all_scores.extend([float(x) for x in summed.detach().cpu().tolist()])

    return all_scores




# ============================
# Generation
# ============================

def stop_at_first(text: str, stop_words: Sequence[str]) -> str:
    if not stop_words:
        return text
    best = len(text)
    for w in stop_words:
        idx = text.find(w)
        if idx != -1:
            best = min(best, idx)
    return text[:best]


def _strip_markdown_fence(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"^```(?:python)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _start_from_first_code_token(text: str) -> str:
    starts = []
    for pat in [r"\bdef\s+\w+\s*\(", r"\bclass\s+\w+", r"\bimport\b", r"\bfrom\b"]:
        m = re.search(pat, text)
        if m:
            starts.append(m.start())
    if starts:
        text = text[min(starts):]
    return text.strip()


def _truncate_to_first_function_block(text: str) -> str:
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*def\s+\w+\s*\(", line):
            start = i
            break

    if start is None:
        return text.strip()

    lines = lines[start:]
    kept = []

    for idx, line in enumerate(lines):
        if idx == 0:
            kept.append(line)
            continue

        stripped = line.strip()

        if stripped == "":
            kept.append(line)
            continue

        indent = len(line) - len(line.lstrip())

        # 一旦回到顶格，并出现新结构/测试/说明，就截断
        if indent == 0:
            if re.match(r"^(def|class|import|from|assert|if\s+__name__|print\s*\()", stripped):
                break
            if stripped.startswith("#"):
                break
            # 顶格普通自然语言也截掉
            break

        kept.append(line)

    return "\n".join(kept).strip()

class _RenameFirstTopLevelFunctionTransformer(ast.NodeTransformer):
    def __init__(self, new_name: str = "f"):
        super().__init__()
        self.new_name = new_name
        self.old_name = None
        self.renamed_first_top_level = False
        self.current_function_stack = []

    def visit_Module(self, node: ast.Module):
        new_body = []
        for stmt in node.body:
            if (
                isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not self.renamed_first_top_level
            ):
                self.old_name = stmt.name
                stmt.name = self.new_name
                self.renamed_first_top_level = True
                stmt = self.visit(stmt)
                new_body.append(stmt)
            else:
                new_body.append(self.visit(stmt))
        node.body = new_body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.current_function_stack.append(node.name)

        # decorator 里如果直接引用了原函数名，也改
        node.decorator_list = [self.visit(dec) for dec in node.decorator_list]

        # 返回注解、参数注解里如果有名字引用，也一起走 visit
        if node.returns is not None:
            node.returns = self.visit(node.returns)

        node.args = self.visit(node.args)
        node.body = [self.visit(stmt) for stmt in node.body]

        self.current_function_stack.pop()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.current_function_stack.append(node.name)

        node.decorator_list = [self.visit(dec) for dec in node.decorator_list]
        if node.returns is not None:
            node.returns = self.visit(node.returns)

        node.args = self.visit(node.args)
        node.body = [self.visit(stmt) for stmt in node.body]

        self.current_function_stack.pop()
        return node

    def visit_arguments(self, node: ast.arguments):
        for arg in node.posonlyargs:
            if arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)
        for arg in node.args:
            if arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)
        if node.vararg and node.vararg.annotation is not None:
            node.vararg.annotation = self.visit(node.vararg.annotation)
        for arg in node.kwonlyargs:
            if arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)
        if node.kwarg and node.kwarg.annotation is not None:
            node.kwarg.annotation = self.visit(node.kwarg.annotation)
        return node

    def visit_Name(self, node: ast.Name):
        # 只改“名字引用”，不改字符串
        if self.old_name is not None and node.id == self.old_name:
            node.id = self.new_name
        return node

    def visit_Attribute(self, node: ast.Attribute):
        # 例如某些极端写法里把函数名挂在属性里，不主动改 attr 本身
        # 但 value 继续递归
        node.value = self.visit(node.value)
        return node

def mask_first_function_name_ast(code: str, new_name: str = "f") -> str:
    """
    Rename the first top-level function to `new_name`, and also rename
    self-recursive calls / direct name references to that function.

    This is AST-based and much safer than regex replacement.
    """
    code = code.strip()
    if not code:
        return code

    tree = ast.parse(code)
    tree = copy.deepcopy(tree)

    transformer = _RenameFirstTopLevelFunctionTransformer(new_name=new_name)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    try:
        new_code = ast.unparse(tree)
    except Exception:
        return code

    return new_code.strip()

def get_first_top_level_function_name(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return stmt.name
    return None

def _remove_comments_and_docstrings(code: str) -> str:
    """
    Remove:
    1) line comments
    2) inline comments
    3) top-level / function-level / class-level docstrings
    Keep normal string literals used in expressions / assignments.
    """
    code = code.replace("\r\n", "\n")

    # ---- step 1: remove comments with tokenize ----
    out = []
    try:
        tokgen = tokenize.generate_tokens(io.StringIO(code).readline)
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0

        for tok in tokgen:
            token_type = tok.type
            token_string = tok.string
            start_line, start_col = tok.start
            end_line, end_col = tok.end

            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out.append(" " * (start_col - last_col))

            # 去掉注释
            if token_type == tokenize.COMMENT:
                pass
            else:
                out.append(token_string)

            prev_toktype = token_type
            last_lineno = end_line
            last_col = end_col

        code = "".join(out)
    except Exception:
        # tokenize 失败时保守回退
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

    # ---- step 2: remove docstrings via AST ----
    try:
        tree = ast.parse(code)

        def strip_docstring(body):
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)

        strip_docstring(tree.body)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                strip_docstring(node.body)

        code = ast.unparse(tree)
    except Exception:
        # 如果 AST 清洗失败，就保留 tokenize 之后的版本
        pass

    return code.strip()



def extract_first_complete_function(code: str) -> str:
    """
    Parse code and return only the first top-level function block.
    If parsing full code fails, try progressively truncating lines.
    """
    code = code.strip()
    if not code:
        return code

    lines = code.splitlines()

    # progressively try prefixes until we get a parsable first function
    for end in range(len(lines), 0, -1):
        chunk = "\n".join(lines[:end]).strip()
        try:
            tree = ast.parse(chunk)
        except Exception:
            continue

        funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not funcs:
            continue

        fn = funcs[0]
        if hasattr(fn, "lineno") and hasattr(fn, "end_lineno"):
            chunk_lines = chunk.splitlines()
            fn_code = "\n".join(chunk_lines[fn.lineno - 1: fn.end_lineno]).strip()
            return fn_code

    return code

def clean_code_generation(text: str) -> str:
    text = _strip_markdown_fence(text)
    text = _start_from_first_code_token(text)
    text = _truncate_to_first_function_block(text)
    text = _remove_comments_and_docstrings(text)
    text = extract_first_complete_function(text)
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        cand = "\n".join(lines[:end]).strip()
        try:
            ast.parse(cand)
            return cand
        except Exception:
            continue

    return text.strip()


@torch.inference_mode()
def sample_n_completions(
    model,
    tokenizer,
    prompt: str,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_words: Sequence[str],
    seed: Optional[int] = None,
) -> List[str]:
    if seed is not None:
        seed_everything(seed)

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    outputs = model.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n,
        max_new_tokens=max_new_tokens,
        max_length=None,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    out: List[str] = []
    for seq in outputs:
        gen_ids = seq[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = stop_at_first(text, stop_words)
        text = clean_code_generation(text)
        out.append(text)

    return out


@torch.inference_mode()
def generate_one_each(
    model,
    tokenizer,
    prompts: Sequence[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_words: Sequence[str],
    batch_size: int = 8,
    seed: Optional[int] = None,
    normalize_instruction: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    if seed is not None:
        seed_everything(seed)

    all_outputs: List[str] = []
    iterator = range(0, len(prompts), batch_size)
    if progress_desc:
        iterator = tqdm(iterator, desc=progress_desc)

    for start in iterator:
        batch_prompts = list(prompts[start : start + batch_size])
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        input_lengths = attention_mask.sum(dim=1).tolist()

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": temperature > 0,
            "num_return_sequences": 1,
            "max_new_tokens": max_new_tokens,
            "max_length": None,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        outputs = model.generate(**generation_kwargs)

        for i, seq in enumerate(outputs):
            gen_ids = seq[int(input_lengths[i]):]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = stop_at_first(text, stop_words)
            text = text.strip()
            if normalize_instruction:
                text = normalize_instruction_text(text)
            all_outputs.append(text)

    return all_outputs


# ============================
# Execution-based evaluation
# ============================

import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Tuple


def candidate_passes_humanevalplus_relaxed(
    code: str,
    context_code: str,
    test_code: str,
    entry_point: str,
    timeout_seconds: int = 8,
) -> Tuple[bool, str]:
    """
    HumanEval+ execution with relaxed function-name matching:
    alias the first top-level function to the required entry_point.
    """
    preamble = textwrap.dedent(
        """\
        import math
        import random
        import re
        import statistics
        import functools
        import itertools
        import collections
        import heapq
        import bisect
        from typing import *
        """
    )

    actual_name = get_first_top_level_function_name(code)

    alias_code = ""
    if actual_name is not None and actual_name != entry_point:
        alias_code = f"{entry_point} = {actual_name}"

    script_parts = [preamble]

    if context_code and context_code.strip():
        script_parts.append(context_code.strip())

    script_parts.append(code.rstrip())

    if alias_code:
        script_parts.append(alias_code)

    script_parts.append(test_code.rstrip())
    script_parts.append(f"check({entry_point})")

    joined = "\n\n".join(script_parts) + "\n"

    with tempfile.TemporaryDirectory() as td:
        file_path = Path(td) / "humaneval_exec.py"
        file_path.write_text(joined, encoding="utf-8")

        try:
            proc = subprocess.run(
                ["python", str(file_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"

        if proc.returncode == 0:
            return True, ""

        stderr = (proc.stderr or proc.stdout or "").strip()
        return False, stderr[:4000]


# ============================
# Clustering
# ============================


# ============================
# RSA / reranking
# ============================


from typing import Optional
from datasets import load_dataset
import re
import ast
import textwrap
from pathlib import Path


def load_humanevalplus_test_split(limit: Optional[int] = None):
    ds = load_dataset("evalplus/humanevalplus", split="test")
    if limit is not None:
        return ds.select(range(min(limit, len(ds))))
    return ds


def _safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))


def task_path(root_dir: str | Path, task_id: str) -> Path:
    return ensure_dir(Path(root_dir) / "tasks") / f"{_safe_task_id(task_id)}.json"


def _split_humaneval_prompt(prompt: str, entry_point: Optional[str] = None):
    lines = prompt.splitlines()
    def_idx = None
    if entry_point:
        target_pat = re.compile(rf"^(async\s+def|def)\s+{re.escape(entry_point)}\s*\(")
        for i, line in enumerate(lines):
            if target_pat.match(line):
                def_idx = i
                break

    if def_idx is None:
        for i, line in enumerate(lines):
            if line.startswith("def ") or line.startswith("async def "):
                def_idx = i
                break

    if def_idx is None:
        return "", prompt

    context_code = "\n".join(lines[:def_idx]).strip()
    function_prompt = "\n".join(lines[def_idx:]).strip()
    return context_code, function_prompt


def _extract_humaneval_instruction_from_prompt(function_prompt: str, entry_point: Optional[str] = None) -> str:
    """
    Extract a clean NL instruction from the FIRST function docstring only.

    Removes:
      - doctest input lines (>>> ...)
      - doctest outputs (True/False/None, lists, numbers, strings, etc.)
      - 'Example:' / 'Examples:' sections
      - trailing blank/comment noise
    """
    try:
        tree = ast.parse(function_prompt)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if entry_point and node.name != entry_point:
                    continue
                doc = ast.get_docstring(node)
                if not doc:
                    continue

                kept = []
                for ln in doc.splitlines():
                    s = ln.strip()

                    if not s:
                        continue

                    # stop once examples start
                    if s.lower().startswith("example"):
                        break

                    # remove doctest prompt lines
                    if s.startswith(">>>"):
                        continue

                    # remove obvious doctest / example outputs
                    if s in {"True", "False", "None"}:
                        continue

                    # remove pure numeric / list / tuple / dict / quoted outputs
                    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
                        continue
                    if re.fullmatch(r"\[.*\]", s):
                        continue
                    if re.fullmatch(r"\(.*\)", s):
                        continue
                    if re.fullmatch(r"\{.*\}", s):
                        continue
                    if re.fullmatch(r"['\"].*['\"]", s):
                        continue

                    # remove inline equality-style examples
                    if "=>" in s or "==" in s:
                        continue

                    kept.append(s)

                text = " ".join(kept).strip()
                text = re.sub(r"\s+", " ", text)

                if text:
                    return text

    except Exception:
        pass

    # fallback: first non-empty non-def line
    lines = function_prompt.splitlines()
    for ln in lines:
        s = ln.strip()
        if s and not s.startswith("def ") and not s.startswith('"""') and not s.startswith("'''"):
            return s

    return function_prompt.strip()
def _extract_first_function_block(function_prompt: str, entry_point: Optional[str] = None) -> str:
    """
    Keep only the first top-level function definition block.
    """
    lines = function_prompt.splitlines()

    start = None
    target_pat = None
    if entry_point:
        target_pat = re.compile(rf"^(async\s+def|def)\s+{re.escape(entry_point)}\s*\(")

    for i, line in enumerate(lines):
        if target_pat is not None:
            if target_pat.match(line):
                start = i
                break
        elif line.startswith("def ") or line.startswith("async def "):
            start = i
            break

    if start is None:
        return function_prompt.strip()

    kept = [lines[start]]

    for line in lines[start + 1:]:
        # keep indented lines / blank lines as part of first function block
        if line.strip() == "":
            kept.append(line)
            continue

        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break

        kept.append(line)

    return "\n".join(kept).rstrip()

def normalize_humanevalplus_doc(doc: dict) -> dict:
    raw_prompt = str(doc["prompt"]).rstrip()
    entry_point = str(doc["entry_point"])
    context_code, function_prompt = _split_humaneval_prompt(raw_prompt, entry_point=entry_point)
    function_prompt = _extract_first_function_block(function_prompt, entry_point=entry_point)
    nl_text = _extract_humaneval_instruction_from_prompt(function_prompt, entry_point=entry_point)

    reference_code = raw_prompt + str(doc.get("canonical_solution", "") or "")

    out = {
        "task_id": str(doc["task_id"]),
        "text": nl_text,
        "raw_prompt": raw_prompt,
        "context_code": context_code,
        "function_prompt": function_prompt,
        "entry_point": entry_point,
        "test": str(doc["test"]),
        "reference_code": reference_code,
        "test_list": [],
        "test_setup_code": "",
        "challenge_test_list": [str(doc["test"])],
    }
    return out



def extract_first_doctest(function_prompt: str) -> str:
    """
    Extract the first doctest block from the first function prompt.

    Returns something like:
        >>> func(args)
        result

    Stops before:
      - next doctest
      - blank line
      - docstring terminator
    """
    lines = function_prompt.splitlines()

    for i, line in enumerate(lines):
        s = line.strip()

        if s.startswith(">>>"):
            block = [s]
            j = i + 1

            while j < len(lines):
                nxt = lines[j].strip()

                if not nxt:
                    break
                if nxt.startswith(">>>"):
                    break
                if nxt in {'"""', "'''"}:
                    break

                block.append(nxt)
                j += 1

            return "\n".join(block)

    return ""

import textwrap

def build_humanevalplus_generation_prompt(doc: dict) -> str:
    """
    Plain-text generation prompt for HumanEval+:
    instruction + one doctest block
    """
    description = doc["text"].strip()
    test_example = extract_first_doctest(doc["function_prompt"]).strip()

    body = description
    if test_example:
        body = body + "\n" + test_example

    body = textwrap.indent(body, "        ")

    return (
        "        You are an expert Python programmer.\n"
        "        Write exactly one Python function that solves the task.\n"
        "        Output only Python code.\n"
        "        Do not write tests.\n"
        "        Do not write example usage.\n"
        "        Do not write explanations.\n"
        "        Do not write markdown fences.\n\n"
        '        """\n'
        f"{body}\n"
        '        """\n'
    )


def initialize_task_files_humanevalplus(config: ReproConfig, dataset) -> None:
    ensure_dir(config.root_dir)
    write_json({"config": config.to_dict()}, Path(config.root_dir) / "run_config.json")

    for raw in dataset:
        doc = normalize_humanevalplus_doc(raw)
        path = task_path(config.root_dir, doc["task_id"])
        if path.exists():
            continue

        prompt = build_humanevalplus_generation_prompt(doc)

        record = {
            "task_id": doc["task_id"],
            "text": doc["text"],
            "raw_prompt": doc["raw_prompt"],
            "context_code": doc["context_code"],
            "function_prompt": doc["function_prompt"],
            "entry_point": doc["entry_point"],
            "test": doc["test"],
            "test_list": doc["test_list"],
            "test_setup_code": doc["test_setup_code"],
            "challenge_test_list": doc["challenge_test_list"],
            "reference_code": doc["reference_code"],
            "generation_prompt": prompt,
            "candidates": None,
            "candidate_eval": None,
            "coder_logprobs": None,
            "reviewer_logprobs": None,
            "prior_logprobs": None,
            "additional_instructions": None,
            "l0_logprobs": None,
            "results_pairwise_avg": None,
        }
        write_json(record, path)


# ============================
# Candidate extraction overrides from the pairwise notebook stage
# ============================

OVERSAMPLE_FACTOR = 3

MIN_OVERSAMPLE = 50

def stable_int_from_task_id(task_id: str, mod: int = 10**8):
    h = hashlib.md5(str(task_id).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod

INLINE_CONTINUATION_MARKERS = [
    "Human:",
    "Assistant:",
    "User:",
    "Question:",
    "Answer:",
    "Instruction:",
    "Problem:",
    "Prompt:",
    "Task:",
    "Premise:",
    "Create a Python function",
    "How can you modify",
    "Given a number",
    "Please rewrite",
    "Please note",
    "You're right",
    "What If",
    "I've written a function",
    "Generate the response as code",
    "The implementation remains the same",
    "Answer in Python code",
    "Here is the completed function",
    "```python",
    "```",
]

def truncate_after_bracket_letter(line: str) -> str:
    """
    If ')', ']', or '}' is immediately followed by a letter
    (no space), treat that as inline prose continuation.
    """

    m = re.search(r'([)\]\}])[A-Za-z]', line)
    if m:
        return line[:m.start(1) + 1]

    return line

def strip_inline_continuation(line: str) -> str:
    line = truncate_after_bracket_letter(line)

    cut = len(line)
    for marker in INLINE_CONTINUATION_MARKERS:
        idx = line.find(marker)
        if idx != -1:
            cut = min(cut, idx)

    return line[:cut].rstrip()

def try_extract_first_function_from_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    text = re.sub(r"^```python\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    lines = [strip_inline_continuation(line) for line in text.splitlines()]

    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    for end in range(len(lines), 0, -1):
        chunk = "\n".join(lines[:end]).strip()
        if not chunk:
            continue

        try:
            tree = ast.parse(chunk)
        except Exception:
            continue

        funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not funcs:
            continue

        fn = funcs[0]

        if hasattr(fn, "lineno") and hasattr(fn, "end_lineno"):
            chunk_lines = chunk.splitlines()
            fn_code = "\n".join(chunk_lines[fn.lineno - 1 : fn.end_lineno]).strip()
            return fn_code

    return ""

# ============================
# Additional-instruction postprocessing
# ============================

def align_humaneval_instruction_style(text: str) -> str:
    """
    Light normalization for HumanEval-style behavioral descriptions.

    For HumanEval+, we keep the generated behavioral description as-is instead of forcing a "Write a function..." prefix.
    We only:
      - normalize whitespace
      - remove prompt artifacts
      - ensure proper capitalization
      - ensure trailing punctuation
    """
    text = text.strip()
    if not text:
        return text

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # remove prompt artifacts sometimes produced by LLM
    text = re.sub(r"^assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^description:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("###", "")
    text = text.replace("```", "")

    # ensure first letter uppercase
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # ensure trailing punctuation
    if text[-1] not in ".!?":
        text += "."

    return text

def postprocess_generated_instruction(text: str) -> str:
    text = text.strip()
    if re.search(r"\bDescription\s*:", text, flags=re.IGNORECASE):
        text = re.split(r"\bDescription\s*:", text, flags=re.IGNORECASE)[-1].strip()
    if re.search(r"\bFunction\s*:", text, flags=re.IGNORECASE):
        text = re.split(r"\bFunction\s*:", text, flags=re.IGNORECASE)[0].strip()
    text = re.sub(r"^\s*(return|if|elif|else|for|while|def|class|pass|yield|import|from)\b.*?\bDescription\s*:\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*assistant\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Description:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[-:`'\"),.\]}]+", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    m = re.search(r"(.+?[.!?])(\s|$)", text)
    if m:
        text = m.group(1).strip()

    text = align_humaneval_instruction_style(text)
    return text

def clean_generated_description(text: str) -> str:
    text = text.strip()

    # 去掉模板残留
    text = text.replace("###Description end###", "")
    text = text.replace("###Description start###", "")
    text = text.replace("###Function start###", "")
    text = text.replace("###Function end###", "")
    text = text.replace("###", "")

    # 去掉示例污染
    text = text.replace("next_power_of_2", "")

    # 去掉反引号
    text = text.replace("`", "")

    # 压缩空格
    text = re.sub(r"\s+", " ", text).strip()

    # 如果句子被截断在奇怪地方（比如最后是介词），简单裁剪
    bad_endings = [
        " of",
        " in",
        " for",
        " with",
        " and",
        " or",
        " the",
        " a",
        " an"
    ]

    for ending in bad_endings:
        if text.endswith(ending):
            text = text[: -len(ending)].strip()

    return text

# ============================
# Pairwise + avg reranking
# ============================

from typing import Dict, List, Optional, Sequence, Tuple, Any
import numpy as np


def zscore_np(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr))
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def get_instruction_pack(record: dict) -> Tuple[str, List[Optional[str]], List[str]]:
    add_inst = record.get("additional_instructions")
    if add_inst is None:
        raise ValueError("missing additional_instructions")

    if isinstance(add_inst, dict):
        original = add_inst.get("original")
        generated_aligned = add_inst.get("generated")
        all_instructions = add_inst.get("all")
        if original is None and all_instructions:
            original = all_instructions[0]
        if generated_aligned is None and all_instructions:
            generated_aligned = all_instructions[1:]
    elif isinstance(add_inst, list):
        all_instructions = add_inst
        original = all_instructions[0]
        generated_aligned = all_instructions[1:]
    else:
        raise ValueError(f"unexpected additional_instructions type: {type(add_inst)}")

    if not all_instructions:
        raise ValueError("empty additional_instructions")

    if original is None:
        original = all_instructions[0]

    return original, list(generated_aligned), list(all_instructions)


def build_candidate_desc_col_map(
    generated_aligned: List[Optional[str]],
    all_instructions: List[str],
    n_candidates: int,
) -> Dict[int, int]:
    from collections import defaultdict

    positions = defaultdict(list)
    for col_idx, txt in enumerate(all_instructions[1:], start=1):
        positions[txt].append(col_idx)

    used = defaultdict(int)
    out = {}

    for cand_idx in range(min(n_candidates, len(generated_aligned))):
        txt = generated_aligned[cand_idx]
        if txt is None:
            continue
        pos_list = positions.get(txt, [])
        k = used[txt]
        if k < len(pos_list):
            out[cand_idx] = pos_list[k]
            used[txt] += 1

    return out


def compute_pairwise_proxy_scores(
    record: dict,
    tie_margin: float = 0.0,
) -> Dict[str, Any]:
    l0 = np.asarray(record["l0_logprobs"], dtype=np.float64)
    candidates = record["candidates"]
    n_candidates = len(candidates)

    original, generated_aligned, all_instructions = get_instruction_pack(record)
    desc_col_map = build_candidate_desc_col_map(
        generated_aligned=generated_aligned,
        all_instructions=all_instructions,
        n_candidates=n_candidates,
    )

    orig_scores = np.asarray(l0[:, 0], dtype=np.float64)

    wins = np.zeros(n_candidates, dtype=np.float64)
    comps = np.zeros(n_candidates, dtype=np.float64)

    for i in range(n_candidates):
        col_i = desc_col_map.get(i)
        if col_i is None:
            continue

        for j in range(i + 1, n_candidates):
            col_j = desc_col_map.get(j)
            if col_j is None:
                continue

            m_i = float(orig_scores[i] - l0[i, col_j])
            m_j = float(orig_scores[j] - l0[j, col_i])

            if m_i > m_j + tie_margin:
                wins[i] += 1.0
            elif m_j > m_i + tie_margin:
                wins[j] += 1.0
            else:
                wins[i] += 0.5
                wins[j] += 0.5

            comps[i] += 1.0
            comps[j] += 1.0

    pairwise_scores = np.zeros(n_candidates, dtype=np.float64)
    valid_mask = comps > 0
    pairwise_scores[valid_mask] = wins[valid_mask] / comps[valid_mask]

    return {
        "pairwise_scores": pairwise_scores,
        "pairwise_wins": wins,
        "pairwise_comparisons": comps,
        "desc_col_map": desc_col_map,
    }


def compute_task_results_pairwise_avg(
    record: dict,
    lambda_values: Sequence[float],
    tie_margin: float = 0.0,
) -> dict:
    candidates = record["candidates"]
    candidate_eval = record["candidate_eval"]
    coder_logprobs = np.asarray(record["coder_logprobs"], dtype=np.float64)
    reviewer_logprobs = np.asarray(record["reviewer_logprobs"], dtype=np.float64)
    l0 = np.asarray(record["l0_logprobs"], dtype=np.float64)

    coder_idx = int(np.argmax(coder_logprobs))
    cr_idx = int(np.argmax(coder_logprobs + reviewer_logprobs))
    oracle_any = any(x["passed"] for x in candidate_eval)

    orig_only_scores = np.asarray(l0[:, 0], dtype=np.float64)
    avg_all_scores = np.asarray(np.mean(l0, axis=1), dtype=np.float64)

    pair_pack = compute_pairwise_proxy_scores(record, tie_margin=tie_margin)
    pairwise_scores = np.asarray(pair_pack["pairwise_scores"], dtype=np.float64)

    pairwise_z = zscore_np(pairwise_scores)
    avg_all_z = zscore_np(avg_all_scores)

    orig_only_idx = int(np.argmax(orig_only_scores))
    avg_all_idx = int(np.argmax(avg_all_scores))
    pairwise_only_idx = int(np.argmax(pairwise_scores))

    pairwise_avg_curve = []
    for lam in lambda_values:
        mixed_scores = pairwise_z + float(lam) * avg_all_z
        idx = int(np.argmax(mixed_scores))
        pairwise_avg_curve.append(
            {
                "lambda": float(lam),
                "selected_idx": idx,
                "selected_passed": bool(candidate_eval[idx]["passed"]),
                "selected_score": float(mixed_scores[idx]),
                "all_scores": [float(x) for x in mixed_scores.tolist()],
            }
        )

    return {
        "oracle_any": bool(oracle_any),

        "coder": {
            "selected_idx": coder_idx,
            "selected_passed": bool(candidate_eval[coder_idx]["passed"]),
            "selected_score": float(coder_logprobs[coder_idx]),
            "all_scores": [float(x) for x in coder_logprobs.tolist()],
        },

        "coderreviewer": {
            "selected_idx": cr_idx,
            "selected_passed": bool(candidate_eval[cr_idx]["passed"]),
            "selected_score": float((coder_logprobs + reviewer_logprobs)[cr_idx]),
            "all_scores": [float(x) for x in (coder_logprobs + reviewer_logprobs).tolist()],
        },

        "orig_only_l0": {
            "selected_idx": orig_only_idx,
            "selected_passed": bool(candidate_eval[orig_only_idx]["passed"]),
            "selected_score": float(orig_only_scores[orig_only_idx]),
            "all_scores": [float(x) for x in orig_only_scores.tolist()],
        },

        "avg_all_l0": {
            "selected_idx": avg_all_idx,
            "selected_passed": bool(candidate_eval[avg_all_idx]["passed"]),
            "selected_score": float(avg_all_scores[avg_all_idx]),
            "all_scores": [float(x) for x in avg_all_scores.tolist()],
        },

        "pairwise_only": {
            "selected_idx": pairwise_only_idx,
            "selected_passed": bool(candidate_eval[pairwise_only_idx]["passed"]),
            "selected_score": float(pairwise_scores[pairwise_only_idx]),
            "all_scores": [float(x) for x in pairwise_scores.tolist()],
        },

        "pairwise_avg_curve": pairwise_avg_curve,
    }


def summarize_run_pairwise_avg(root_dir: str | Path, lambda_values: Sequence[float]) -> dict:
    task_dir = Path(root_dir) / "tasks"
    records = []
    for path in sorted(task_dir.glob("*.json")):
        rec = read_json(path)
        if rec.get("results_pairwise_avg") is not None:
            records.append(rec)

    total = len(records)
    if total == 0:
        raise RuntimeError("No completed pairwise+avg task records found.")

    summary = {
        "num_tasks": total,
        "oracle10": sum(bool(r["results_pairwise_avg"]["oracle_any"]) for r in records) / total,
        "random_acc": (
            sum(
                bool(ev["passed"])
                for r in records
                for ev in r.get("candidate_eval", [])
            )
            / max(1, sum(len(r.get("candidate_eval", [])) for r in records))
        ),
        "coder_acc": sum(bool(r["results_pairwise_avg"]["coder"]["selected_passed"]) for r in records) / total,
        "coderreviewer_acc": sum(bool(r["results_pairwise_avg"]["coderreviewer"]["selected_passed"]) for r in records) / total,
        "orig_only_l0_acc": sum(bool(r["results_pairwise_avg"]["orig_only_l0"]["selected_passed"]) for r in records) / total,
        "avg_all_l0_acc": sum(bool(r["results_pairwise_avg"]["avg_all_l0"]["selected_passed"]) for r in records) / total,
        "pairwise_only_acc": sum(bool(r["results_pairwise_avg"]["pairwise_only"]["selected_passed"]) for r in records) / total,
        "pairwise_avg_curve": [],
    }

    for lam in lambda_values:
        lam = float(lam)
        acc = sum(
            next(x for x in r["results_pairwise_avg"]["pairwise_avg_curve"] if float(x["lambda"]) == lam)["selected_passed"]
            for r in records
        ) / total
        summary["pairwise_avg_curve"].append({"lambda": lam, "accuracy": float(acc)})

    best = max(summary["pairwise_avg_curve"], key=lambda x: x["accuracy"])
    summary["pairwise_avg_best"] = best
    summary["pairwise_avg_lambda1"] = next(
        x for x in summary["pairwise_avg_curve"] if float(x["lambda"]) == 1.0
    )
    return summary
