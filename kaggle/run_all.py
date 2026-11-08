"""Run everything on a Kaggle GPU notebook, in order, resumable.

    python kaggle/run_all.py prep    # data, literature index, extractions, generations, test-list hashes
    python kaggle/run_all.py gate    # write the hashes into a PREREG addendum, commit and push it
    python kaggle/run_all.py eval    # every evaluation (only after the gate commit exists)
    python kaggle/run_all.py all     # prep -> gate -> eval
    python kaggle/run_all.py push    # commit and push results

Each step leaves a marker in data/cache/steps/, so a stopped session resumes where it stopped.
The pre-registration rule (R2) is enforced in code: `eval` refuses to start unless PREREG.md contains the
current test-list hashes in a committed state.
"""
import datetime
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _d in ("E2", "E4", "E6"):
    sys.path.insert(0, str(ROOT / _d))
os.chdir(ROOT)
os.environ.setdefault("EMB_DISK_CACHE", "0")
os.environ.setdefault("LLM_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

from common.io import DATA, read_jsonl, sha256_list, write_json  # noqa: E402

STEPS = DATA / "steps"
STEPS.mkdir(parents=True, exist_ok=True)
HF = "https://huggingface.co/datasets"
LOG = ROOT / "kaggle" / "run_log.md"


def log(msg):
    line = f"[{datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def step(name):
    def deco(fn):
        def run():
            if (STEPS / f"{name}.done").exists():
                print(f"-- skip {name} (done)")
                return
            t0 = time.time()
            log(f"start {name}")
            fn()
            (STEPS / f"{name}.done").write_text("ok")
            log(f"done  {name} ({(time.time() - t0) / 60:.1f} min)")
        run.__name__ = name
        return run
    return deco


def in_dir(d, fn, *a):
    old = os.getcwd()
    sys.path.insert(0, str(ROOT / d))
    os.chdir(ROOT / d)
    try:
        return fn(*a)
    finally:
        os.chdir(old)


# ======================================================================= prep

@step("download")
def download():
    d = DATA / "hf"
    (d / "arxiv").mkdir(parents=True, exist_ok=True)
    sh(f"curl -sL -o {d}/iclr_reviews.parquet {HF}/smallari/openreview-iclr-peer-reviews/resolve/main/data/raw-00000-of-00001.parquet")
    sh(f"curl -sL -o {d}/iclr2026.parquet {HF}/davidheineman/iclr-2026/resolve/main/data/main-00000-of-00001.parquet")
    sh(f"curl -sL -o {d}/njb_pointwise.parquet {HF}/allenai/novelty-judge-bench/resolve/main/data/human-only_pointwise.parquet")
    for i in range(11):
        sh(f"curl -sL -o {d}/arxiv/train-{i:05d}-of-00011.parquet "
           f"{HF}/librarian-bots/arxiv-metadata-snapshot/resolve/main/data/train-{i:05d}-of-00011.parquet")


@step("convert")
def convert():
    sys.path.insert(0, str(ROOT / "data"))
    import convert_hf
    convert_hf.iclr(); convert_hf.njb(); convert_hf.arxiv(); convert_hf.hypobench()
    sh(f"rm -rf {DATA}/hf/arxiv")  # free 2.7 GB


@step("index")
def index():
    from main.data import LitIndex
    LitIndex.build(start="2019-01-01", max_papers=int(os.environ.get("INDEX_MAX", "300000")))


@step("e4_cards")
def e4_cards():
    import run_e4
    in_dir("E4", run_e4.cmd_cards)


@step("e2_prep")
def e2_prep():
    import run_e2
    in_dir("E2", run_e2.cmd_pools)
    in_dir("E2", run_e2.cmd_claims, None)     # all 100 focal papers per field
    in_dir("E2", run_e2.cmd_calibrate)


@step("e6_prep")
def e6_prep():
    import run_e6
    in_dir("E6", run_e6.cmd_build_lit)
    in_dir("E6", run_e6.cmd_build_data)
    in_dir("E6", run_e6.cmd_generate)


@step("m_data")
def m_data():
    from main import data as md
    md.build_iclr()
    md.build_uptake()
    md.build_groups()


def current_hashes():
    h = {}
    for F in ("cs.CL", "cs.CV", "cs.RO"):
        f = ROOT / "E2" / "focal" / f"{F}.jsonl"
        if f.exists():
            ids = [r["id"] for r in read_jsonl(f)]
            h[f"E2_focal_{F}"] = {"n": len(ids), "sha256": sha256_list(ids)}
    f = ROOT / "data" / "novelty_judge_bench.jsonl"
    if f.exists():
        ids = [r["id"] for r in read_jsonl(f)]
        h["E2_human_check"] = {"n": len(ids), "sha256": sha256_list(ids)}
    f = ROOT / "E4" / "cards.jsonl"
    if f.exists():
        ids = [r["paper_id"] for r in read_jsonl(f) if r["year"] == 2026]
        h["E4_test_2026"] = {"n": len(ids), "sha256": sha256_list(ids)}
    for tb in ("literature", "data"):
        f = ROOT / "E6" / f"testbed_{tb}.jsonl"
        if f.exists():
            ids = [r["input_set_id"] for r in read_jsonl(f)]
            h[f"E6_{tb}"] = {"n": len(ids), "sha256": sha256_list(ids)}
    for k in ("iclr", "uptake"):
        f = ROOT / "main" / f"items_{k}.jsonl"
        if f.exists():
            ids = [r["id"] for r in read_jsonl(f) if r["split"] == "test"]
            h[f"M_{k}_test"] = {"n": len(ids), "sha256": sha256_list(ids)}
    return h


def prep():
    for s in (download, convert, index, e4_cards, e2_prep, e6_prep, m_data):
        s()
    h = current_hashes()
    write_json(ROOT / "kaggle" / "hashes.json", h)
    # id manifests (no text) so the test lists can be checked against the hashes
    man = {}
    for k in ("iclr", "uptake"):
        f = ROOT / "main" / f"items_{k}.jsonl"
        if f.exists():
            man[k] = [{"id": r["id"], "split": r["split"], "t": r["t"]} for r in read_jsonl(f)]
    write_json(ROOT / "main" / "item_manifest.json", man)
    log("prep finished; hashes in kaggle/hashes.json")


# ======================================================================= gate

def git(cmd):
    sh(f"git -C {ROOT} {cmd}")


def gate():
    h = json.loads((ROOT / "kaggle" / "hashes.json").read_text())
    pre = (ROOT / "PREREG.md").read_text(encoding="utf-8")
    if all(v["sha256"] in pre for v in h.values()):
        log("gate: hashes already in PREREG.md")
    else:
        today = datetime.date.today().isoformat()
        add = [f"\n### Addendum {today}: test lists for E2, E4, E6 and M (before any evaluation)",
               "Frozen before evaluation; produced by `kaggle/run_all.py prep` on a Kaggle T4.",
               "Compute: Qwen2.5-1.5B-Instruct (extraction temperature 0, generation 0.7), Qwen3-Embedding-0.6B, "
               "Qwen3-Reranker-0.6B; literature index = 300k arXiv papers (2019+, uniform sample, seed 0).",
               "ICLR inputs R are retrieved (10 nearest arXiv papers before the submission deadline), because the public "
               "mirrors carry no reference lists. E1-based tests (I1, S1, S2, S3) wait for the E1 items and get their "
               "own addendum.", ""]
        add += [f"* `{k}`: n = {v['n']}, sha256 `{v['sha256']}`" for k, v in h.items()]
        with open(ROOT / "PREREG.md", "a", encoding="utf-8") as f:
            f.write("\n".join(add) + "\n")
        git("add PREREG.md kaggle/hashes.json main/item_manifest.json E2/pool_manifest.json kaggle/run_log.md")
        git('commit -m "PREREG addendum: test-list hashes for E2, E4, E6, M (before evaluation)" '
            '-m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"')
        push()
    (STEPS / "gate.done").write_text("ok")


def gate_ok():
    h = json.loads((ROOT / "kaggle" / "hashes.json").read_text())
    pre = (ROOT / "PREREG.md").read_text(encoding="utf-8")
    dirty = subprocess.run(f"git -C {ROOT} diff --quiet HEAD -- PREREG.md", shell=True).returncode != 0
    return all(v["sha256"] in pre for v in h.values()) and not dirty


# ======================================================================= eval

def safe(name, fn, *a):
    if (STEPS / f"{name}.done").exists():
        print(f"-- skip {name} (done)"); return
    log(f"start {name}")
    try:
        fn(*a)
        (STEPS / f"{name}.done").write_text("ok")
        log(f"done  {name}")
    except Exception as e:  # a failed step is logged and the next one runs; nothing is invented
        log(f"FAILED {name}: {e!r}")
        traceback.print_exc()


def evaluate():
    if not gate_ok():
        sys.exit("PREREG.md does not contain the committed test-list hashes: run `gate` first (rule R2).")
    import run_e2, run_e4, run_e6
    from main import run as mr
    from main import battery as B
    from main.context import Ctx
    safe("e2_axioms", in_dir, "E2", run_e2.cmd_axioms, None)
    os.environ["E2_HUMAN_CLAIMS"] = "llm"
    safe("e2_human", in_dir, "E2", run_e2.cmd_human)
    safe("e4_ceiling", in_dir, "E4", run_e4.cmd_ceiling)
    safe("e4_views", in_dir, "E4", run_e4.cmd_views)
    safe("e4_reading", in_dir, "E4", run_e4.cmd_reading)
    safe("e4_summary", in_dir, "E4", run_e4.cmd_summary)
    safe("e6_evaluate", in_dir, "E6", run_e6.cmd_evaluate)
    safe("e6_sample20", in_dir, "E6", run_e6.cmd_sample20)
    safe("m_fit", mr.cmd_fit)
    safe("m_core", lambda: mr.run_tests(B.CORE, Ctx(), "test"))
    safe("m_extended", lambda: mr.run_tests(B.EXTENDED, Ctx(), "test"))
    safe("m_timing", mr.cmd_timing)


def push():
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        log("no GITHUB_TOKEN: results stay in /kaggle/working (download them from the Output tab)")
        return
    url = subprocess.run(f"git -C {ROOT} remote get-url origin", shell=True, capture_output=True, text=True).stdout.strip()
    url = url.split("@")[-1].replace("https://", "")
    r = subprocess.run(f"git -C {ROOT} push https://x-access-token:{tok}@{url} HEAD:main", shell=True,
                       capture_output=True, text=True)
    log("pushed to GitHub" if r.returncode == 0 else
        "push FAILED (token is probably read-only: give it Contents = Read and write); results kept locally")


def commit_results():
    paths = ("E2/results E2/calib.json E2/pool_manifest.json E4/results E4/cards.jsonl E6/generations.jsonl "
             "E6/results E6/error_analysis_samples.md main/results main/item_manifest.json kaggle/run_log.md "
             "kaggle/hashes.json")
    git(f"add -A {paths}")
    r = subprocess.run(f'git -C {ROOT} commit -m "Results from the Kaggle run" '
                       f'-m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"', shell=True)
    if r.returncode == 0:
        push()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "gate":
        gate()
    elif cmd == "eval":
        evaluate(); commit_results()
    elif cmd == "push":
        commit_results()
    elif cmd == "all":
        prep(); gate(); evaluate(); commit_results()
