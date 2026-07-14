#!/usr/bin/env python3
"""Apply Cliff 2b fused kernel patch to the running vllm install.

Run AFTER Genesis patches (which modify chunk.py) and BEFORE vllm serve.
Idempotent — safe to re-run.
"""
import os
import shutil
import site
import sys


def find_vllm_root():
    """Find the installed vllm package directory."""
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = os.path.join(p, "vllm")
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "__init__.py")):
            return candidate
    try:
        import vllm
        return os.path.dirname(vllm.__file__)
    except ImportError:
        return None


def apply_patch():
    vllm_root = find_vllm_root()
    if vllm_root is None:
        print("[cliff2b] ERROR: cannot find vllm package", file=sys.stderr)
        sys.exit(1)

    ops_dir = os.path.join(vllm_root, "model_executor", "layers", "fla", "ops")
    if not os.path.isdir(ops_dir):
        print(f"[cliff2b] SKIP: {ops_dir} does not exist (no FLA ops)", file=sys.stderr)
        return

    # 1. Copy chunk_fused.py
    patch_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(patch_dir, "chunk_fused.py")
    dst = os.path.join(ops_dir, "chunk_fused.py")
    shutil.copy2(src, dst)
    print(f"[cliff2b] chunk_fused.py -> {dst}")

    # 2. Patch chunk.py (add import + dispatch) if not already patched
    chunk_py = os.path.join(ops_dir, "chunk.py")
    with open(chunk_py, "r") as f:
        content = f.read()

    if "_USE_FUSED" in content:
        print("[cliff2b] chunk.py already patched, skipping")
        return

    # Add imports after last existing import line
    import_block = (
        "\nimport os\n"
        "from .chunk_fused import chunk_gated_delta_rule_fwd_fused\n"
        "\n_USE_FUSED = os.environ.get(\"VLLM_FLA_GDN_FUSED\", \"1\").strip() != \"0\"\n"
    )
    # Insert after 'from .wy_fast import recompute_w_u_fwd'
    marker = "from .wy_fast import recompute_w_u_fwd"
    if marker in content:
        content = content.replace(marker, marker + import_block, 1)
    else:
        print("[cliff2b] WARN: could not find import marker, prepending", file=sys.stderr)
        content = import_block + content

    # Replace the two-kernel path with conditional dispatch
    # [2026-07-14 re-anchor for dev1060] Live chunk.py adds PN354's
    # `**_GENESIS_PN354_KW` to both calls and `core_attn_out=core_attn_out`
    # to chunk_fwd_o (buffer reuse). Anchor must match verbatim or the patch
    # silently falls through to its import-only branch (observed: 0 fused refs).
    old_vanilla = (
        "    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(\n"
        "        k=k,\n"
        "        w=w,\n"
        "        u=u,\n"
        "        g=g,\n"
        "        initial_state=initial_state,\n"
        "        output_final_state=output_final_state,\n"
        "        cu_seqlens=cu_seqlens,\n"
        "        chunk_indices=chunk_indices,\n"
        "        chunk_offsets=chunk_offsets,\n"
        "        **_GENESIS_PN354_KW,\n"
        "    )\n"
        "    o = chunk_fwd_o(\n"
        "        q=q,\n"
        "        k=k,\n"
        "        v=v_new,\n"
        "        h=h,\n"
        "        g=g,\n"
        "        scale=scale,\n"
        "        cu_seqlens=cu_seqlens,\n"
        "        chunk_indices=chunk_indices,\n"
        "        core_attn_out=core_attn_out,\n"
        "        **_GENESIS_PN354_KW,\n"
        "    )\n"
        "    if SUPPRESS_LEVEL < 3:\n"
        "        return g, o, A, final_state, None, None, None\n"
        "    elif SUPPRESS_LEVEL >= 3:\n"
        "        return g, o, A, final_state, w, h, v_new"
    )

    new_dispatch = (
        "    if _USE_FUSED:\n"
        "        o, final_state = chunk_gated_delta_rule_fwd_fused(\n"
        "            q=q,\n"
        "            k=k,\n"
        "            v=u,\n"
        "            w=w,\n"
        "            g=g,\n"
        "            gk=None,\n"
        "            scale=scale,\n"
        "            initial_state=initial_state,\n"
        "            output_final_state=output_final_state,\n"
        "            o_buf=core_attn_out,\n"
        "            cu_seqlens=cu_seqlens,\n"
        "            chunk_offsets=chunk_offsets,\n"
        "            use_exp2=_GENESIS_PN354_USE_EXP2,\n"
        "        )\n"
        "        if SUPPRESS_LEVEL < 3:\n"
        "            return g, o, A, final_state, None, None, None\n"
        "        return g, o, A, final_state, w, None, None\n"
        "    else:\n"
        "        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(\n"
        "            k=k,\n"
        "            w=w,\n"
        "            u=u,\n"
        "            g=g,\n"
        "            initial_state=initial_state,\n"
        "            output_final_state=output_final_state,\n"
        "            cu_seqlens=cu_seqlens,\n"
        "            chunk_indices=chunk_indices,\n"
        "            chunk_offsets=chunk_offsets,\n"
        "            **_GENESIS_PN354_KW,\n"
        "        )\n"
        "        o = chunk_fwd_o(\n"
        "            q=q,\n"
        "            k=k,\n"
        "            v=v_new,\n"
        "            h=h,\n"
        "            g=g,\n"
        "            scale=scale,\n"
        "            cu_seqlens=cu_seqlens,\n"
        "            chunk_indices=chunk_indices,\n"
        "            core_attn_out=core_attn_out,\n"
        "            **_GENESIS_PN354_KW,\n"
        "        )\n"
        "        if SUPPRESS_LEVEL < 3:\n"
        "            return g, o, A, final_state, None, None, None\n"
        "        return g, o, A, final_state, w, h, v_new"
    )

    if old_vanilla in content:
        content = content.replace(old_vanilla, new_dispatch, 1)
        print("[cliff2b] chunk.py patched: fused dispatch added")
    else:
        print("[cliff2b] WARN: vanilla two-kernel block not found (Genesis may have modified it)", file=sys.stderr)
        print("[cliff2b] WARN: chunk.py NOT patched — fused kernel available but not wired", file=sys.stderr)
        # Still write the imports so chunk_fused is at least importable
        pass

    with open(chunk_py, "w") as f:
        f.write(content)

    # 3. Clear pycache
    pycache = os.path.join(ops_dir, "__pycache__")
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)
        print("[cliff2b] __pycache__ cleared")


if __name__ == "__main__":
    apply_patch()
