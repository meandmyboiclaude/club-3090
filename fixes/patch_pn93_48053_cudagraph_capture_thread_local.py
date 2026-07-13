#!/usr/bin/env python3
"""PN93 — capture_error_mode="thread_local" on all cudagraph capture paths.

Backport of vllm#48053 to nightly-9e57de71 (dev1060), 4 files:

  - vllm/compilation/cuda_graph.py           (CUDAGraphWrapper.__call__)
  - vllm/compilation/breakable_cudagraph.py  (_begin_segment capture_begin)
  - vllm/v1/worker/gpu/cudagraph_utils.py    (MRV2 capture loop)
  - vllm/v1/worker/gpu_ubatch_wrapper.py     (_capture_ubatch_thread)

torch's default capture_error_mode="global" invalidates an in-progress
capture when ANY thread in the process issues CUDA work — KV-offload
daemons, out-of-tree plugins moving weights on side streams, and our own
VRAM-guardian/telemetry helpers can all poison a capture and crash boot
with "operation would make the legacy stream depend on a capturing
blocking stream" / cudaErrorStreamCaptureUnjoined-class failures.
"thread_local" scopes capture-invalidating checks to the capturing thread;
kernels issued to the capture stream are still recorded (stream-capture
status follows the stream, not the issuing thread's error mode).

Directly relevant to us: PIECEWISE cudagraph capture on TQ3+MTP with the
VRAM guardian polling in a background thread during warmup.

Anchors verified byte-exact in-image 2026-07-13 (genesis N13 patches the
gc/empty_cache lambda region of cuda_graph.py ABOVE this anchor — no
overlap; verified combined). Retire when upstream passes
capture_error_mode on these sites — this patcher self-retires per file.
"""
import pathlib
import sys

LOG = "[pn93-cudagraph-capture-thread-local]"
VLLM = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "# PN93:"

COMMENT = (
    '{i}# PN93: vllm#48053 backport — thread_local: CUDA work issued by\n'
    "{i}# helper threads (KV offloaders, guardian/telemetry pollers, plugins\n"
    "{i}# on side streams) must not invalidate this thread's capture.\n"
)

# (target, OLD, NEW) — NEW built per-site to keep upstream indentation exact.
SITES = (
    (
        VLLM / "compilation/cuda_graph.py",
        (
            "                with torch.cuda.graph(\n"
            "                    cudagraph,\n"
            "                    pool=self.graph_pool,\n"
            "                    stream=current_stream(),\n"
            "                ):\n"
        ),
        (
            COMMENT.format(i="                ")
            + "                with torch.cuda.graph(\n"
            "                    cudagraph,\n"
            "                    pool=self.graph_pool,\n"
            "                    stream=current_stream(),\n"
            '                    capture_error_mode="thread_local",\n'
            "                ):\n"
        ),
    ),
    (
        VLLM / "compilation/breakable_cudagraph.py",
        (
            "        if self.pool is not None:\n"
            "            g.capture_begin(pool=self.pool)\n"
            "        else:\n"
            "            g.capture_begin()\n"
        ),
        (
            COMMENT.format(i="        ")
            + "        if self.pool is not None:\n"
            '            g.capture_begin(pool=self.pool, capture_error_mode="thread_local")\n'
            "        else:\n"
            '            g.capture_begin(capture_error_mode="thread_local")\n'
        ),
    ),
    (
        VLLM / "v1/worker/gpu/cudagraph_utils.py",
        "                        with torch.cuda.graph(graph, self.pool):\n",
        (
            COMMENT.format(i="                        ")
            + "                        with torch.cuda.graph(\n"
            '                            graph, self.pool, capture_error_mode="thread_local"\n'
            "                        ):\n"
        ),
    ),
    (
        VLLM / "v1/worker/gpu_ubatch_wrapper.py",
        (
            "            with torch.cuda.graph(\n"
            "                cudagraph_metadata.cudagraph,\n"
            "                stream=compute_stream,\n"
            "                pool=self.graph_pool,\n"
            "            ):\n"
        ),
        (
            COMMENT.format(i="            ")
            + "            # (The ubatch threads' own kernels are still captured:\n"
            "            # stream-capture status follows the stream, not the issuing\n"
            "            # thread's error mode.)\n"
            "            with torch.cuda.graph(\n"
            "                cudagraph_metadata.cudagraph,\n"
            "                stream=compute_stream,\n"
            "                pool=self.graph_pool,\n"
            '                capture_error_mode="thread_local",\n'
            "            ):\n"
        ),
    ),
)


def main() -> int:
    rc = 0
    for target, old, new in SITES:
        if not target.exists():
            print(f"{LOG} FATAL: {target} not present", file=sys.stderr)
            return 1
        text = target.read_text()
        if MARKER in text:
            print(f"{LOG} {target.name}: already applied (idempotent)")
            continue
        if "capture_error_mode" in text:
            print(
                f"{LOG} {target.name}: upstream drift — capture_error_mode "
                f"already present, self-retire (no-op)"
            )
            continue
        if old not in text:
            print(
                f"{LOG} FATAL: anchor-not-found in {target.name} — upstream "
                f"refactor; re-derive before boot (helper-thread CUDA work can "
                f"poison cudagraph capture on TQ3+MTP warmup)",
                file=sys.stderr,
            )
            rc = 1
            continue
        if text.count(old) != 1:
            print(
                f"{LOG} FATAL: ambiguous anchor in {target.name} "
                f"({text.count(old)} matches)",
                file=sys.stderr,
            )
            rc = 1
            continue
        target.write_text(text.replace(old, new, 1))
        print(
            f"{LOG} {target.name}: applied — capture_error_mode='thread_local' "
            f"(vllm#48053 backport)"
        )
    return rc


sys.exit(main())
