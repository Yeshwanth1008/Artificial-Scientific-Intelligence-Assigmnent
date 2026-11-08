"""Main Problem runner -- one command per table of REPORT.md.

    python -m main.run hashes                 # sha256 of the (sub-sampled) test lists -> paste into PREREG.md
    python -m main.run fit                    # facet means + both quality heads (train; alpha chosen on select)
    python -m main.run select                 # every core test on the SELECTION year (allowed before freezing)
    python -m main.run test core              # Table 1: core battery on the test set (run once)
    python -m main.run test extended          # Table 2: extended battery
    python -m main.run test <TestName>        # a single test
    python -m main.run example "<question>" <t>  # print every readout for one question
    python -m main.run timing                 # seconds per question on this machine (excluding retrieval)
"""
import json
import sys
import time

import numpy as np

from common.io import sha256_list, write_json
from . import battery as B
from .context import HERE, Ctx

RES = HERE / "results"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def cmd_hashes():
    ctx = Ctx(load_state=False)
    out = {f"{k}_test": {"n": len(ctx.split(k, "test")), "sha256": sha256_list([r["id"] for r in ctx.split(k, "test")])}
           for k in ("iclr", "uptake")}
    print(json.dumps(out, indent=2))


def cmd_fit():
    ctx = Ctx(load_state=False)
    m = ctx.model
    tr = ctx.split("iclr", "train")
    m.fit_facet_means([r["facets"] for r in tr] + [r["facets"] for r in ctx.split("uptake", "train")])
    ctx._phi = {}  # scope depends on the facet means: drop any cache built before them
    for kind, target, head in (("iclr", "contribution", "q_review"), ("uptake", "log_uptake", "q_uptake")):
        get = (lambda r: r["scores"][target]) if kind == "iclr" else (lambda r: r[target])
        tr = [r for r in ctx.split(kind, "train") if not np.isnan(get(r))]
        se = [r for r in ctx.split(kind, "select") if not np.isnan(get(r))]
        m.fit_head(head, [ctx.phi_item(r) for r in tr], np.array([get(r) for r in tr]),
                   [ctx.phi_item(r) for r in se], np.array([get(r) for r in se]))
        print(head, "alpha =", m.heads[head][2])
    m.save()
    ctx.flush()


class _SelectCtx(Ctx):
    """Runs tests with 'select' in place of 'test' -- for development before the test set is opened."""
    def split(self, kind, split):
        return super().split(kind, "select" if split == "test" else split)


def run_tests(names, ctx, tag):
    for n in names:
        t0 = time.time()
        try:
            r = getattr(B, n)(ctx)
        except Exception as e:  # a test that cannot run is listed with the reason, never faked
            r = {"error": repr(e)}
        r["seconds"] = round(time.time() - t0, 1)
        write_json(RES / tag / f"{n}.json", _jsonable(r))
        print(f"{n:22s} pass={r.get('pass')}  ({r['seconds']}s)" + (f"  {r.get('not_run') or r.get('error')}"
                                                                     if ("not_run" in r or "error" in r) else ""))
    ctx.flush()
    write_json(RES / tag / "_not_run.json", B.NOT_RUN_REASONS)


def cmd_example(q, t):
    ctx = Ctx()
    phi = ctx.model.encode(q, None, t)
    print(json.dumps(_jsonable(ctx.model.readouts(phi)), indent=2))


def cmd_timing(n=30):
    ctx = Ctx()
    rows = ctx.split("iclr", "select")[:n]
    t0 = time.time()
    for r in rows:
        ctx.model.encode(r["q"], r["R"], r["t"], facets=None)
    print(f"{(time.time() - t0) / n:.2f} s per question (incl. facet extraction and novelty retrieval)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "hashes":
        cmd_hashes()
    elif cmd == "fit":
        cmd_fit()
    elif cmd == "select":
        run_tests(B.CORE, _SelectCtx(), "select")
    elif cmd == "test":
        which = sys.argv[2]
        names = B.CORE if which == "core" else B.EXTENDED if which == "extended" else [which]
        run_tests(names, Ctx(), "test")
    elif cmd == "example":
        cmd_example(sys.argv[2], sys.argv[3])
    elif cmd == "timing":
        cmd_timing()
