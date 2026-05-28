"""Core utilities for the BigCodeBench CodeRSA pipeline."""
import json
import hashlib
import random
import re
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import ast
import numpy as np
import torch
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
    score_batch_size: int = 8
    generation_batch_size: int = 8
    eval_timeout_seconds: int = 8
    force_regenerate_candidates: bool = True
    force_rescore: bool = True
    force_recompute_instructions: bool = True
    force_recompute_l0: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


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
    try:
        func_clean = mask_first_function_name_ast(code, new_name="f")
    except Exception:
        func_clean = clean_code_generation(code)
        try:
            func_clean = mask_first_function_name_ast(func_clean, new_name="f")
        except Exception:
            func_clean = func_clean.strip() or code.strip()

    fewshot = textwrap.dedent("""\
    Read the Python function and write a precise behavioral specification.

    Rules:
    - Output two to four concise sentences.
    - Describe only behavior that is explicitly supported by the code.
    - Preserve observable details that tests may check: exact return type or shape, ordering, filtering, case sensitivity, labels/titles, file paths, network/file side effects, randomness/seed behavior, and exact exception or error-message behavior.
    - Mention implementation details only when they change observable behavior.
    - Do not guess intended behavior beyond what the code actually does.
    - Do not return code.
    - Do not use markdown.

    Function:
    def f(groups):
        result = {}
        for key, values in groups.items():
            values = [x for x in values if x is not None]
            if not values:
                result[key] = None
            elif len(values) == 1:
                result[key] = values[0]
            else:
                result[key] = sum(values) / len(values)
        return result
    Description:
    Return a dictionary with the same keys as the input mapping. For each key, ignore None values; return None if none remain, the sole remaining value if exactly one remains, or the arithmetic mean if two or more remain.

    Function:
    def f(path):
        with open(path, "r", encoding="utf-8") as fh:
            words = re.findall(r"[A-Za-z]+", fh.read().lower())
        counter = collections.Counter(w for w in words if len(w) >= 4)
        return counter.most_common(5)
    Description:
    Read the UTF-8 text file and count only alphabetic words after lowercasing the entire file. Ignore words shorter than four characters, and return up to five (word, count) pairs ordered by descending frequency.

    Function:
    def f(items):
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    Description:
    Return a list containing the first occurrence of each distinct input item. Preserve the original order.

    Function:
    def f(url):
        response = urllib.request.urlopen(url)
        data = response.read().decode("utf-8")
        words = re.findall(r"\\b\\w+\\b", data)
        counts = collections.Counter(words)
        top = counts.most_common(10)
        fig, ax = plt.subplots()
        ax.bar([w for w, _ in top], [c for _, c in top])
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
        ax.set_title("Top 10 Words")
        return counts, ax
    Description:
    Fetch UTF-8 text from the URL, count regex word tokens without lowercasing or removing stopwords, and return the full Counter together with a matplotlib Axes. Plot the ten most common words as a vertical bar chart with x-axis label "Words", y-axis label "Frequency", and title "Top 10 Words".

    Function:
    def f(url, download_path):
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            if r.headers.get("Content-Type") != "application/zip":
                return "Error: The URL does not point to a ZIP file."
            with tempfile.TemporaryFile(suffix=".zip") as tmp:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        tmp.write(chunk)
                tmp.seek(0)
                with zipfile.ZipFile(tmp) as zf:
                    zf.extractall(download_path)
            return download_path
        except zipfile.BadZipFile:
            return "Error: The downloaded file is not a valid ZIP file."
        except requests.RequestException:
            return "Error: Unable to download the file from the provided URL."
        except Exception as exc:
            return f"Error: {exc}"
    Description:
    Download the URL with streaming requests and require the response Content-Type to be exactly "application/zip". Extract the ZIP contents into download_path and return download_path on success; return the specific ZIP-type, corrupt-ZIP, download-failure, or generic error string shown by the code.

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
# Task records
# ============================


def _safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))


def task_path(root_dir: str | Path, task_id: str) -> Path:
    return ensure_dir(Path(root_dir) / "tasks") / f"{_safe_task_id(task_id)}.json"


# ============================
# Candidate extraction helpers
# ============================

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

def align_behavior_description_text(text: str) -> str:
    """
    Light normalization for generated behavioral descriptions.

    The generated text is kept as a behavior description rather than converted
    into a task template.
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

    sentences = re.findall(r"[^.!?]+[.!?]", text)
    if sentences:
        text = " ".join(s.strip() for s in sentences[:3]).strip()
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)

    text = align_behavior_description_text(text)
    return text


def is_unusable_generated_instruction(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return True
    return bool(re.fullmatch(r"no description (available|provided)\.?", text, flags=re.IGNORECASE))

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
# CodeRSA reranking
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

    codersa_scores = pairwise_z + avg_all_z
    codersa_idx = int(np.argmax(codersa_scores))

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

        "codersa": {
            "selected_idx": codersa_idx,
            "selected_passed": bool(candidate_eval[codersa_idx]["passed"]),
            "selected_score": float(codersa_scores[codersa_idx]),
            "all_scores": [float(x) for x in codersa_scores.tolist()],
        },
    }


def summarize_run_pairwise_avg(root_dir: str | Path) -> dict:
    task_dir = Path(root_dir) / "tasks"
    records = []
    for path in sorted(task_dir.glob("*.json")):
        rec = read_json(path)
        if rec.get("results_pairwise_avg") is not None:
            records.append(rec)

    total = len(records)
    if total == 0:
        raise RuntimeError("No completed CodeRSA task records found.")

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
        "codersa_acc": sum(bool(r["results_pairwise_avg"]["codersa"]["selected_passed"]) for r in records) / total,
    }

    summary["codersa"] = {"accuracy": summary["codersa_acc"]}
    return summary
