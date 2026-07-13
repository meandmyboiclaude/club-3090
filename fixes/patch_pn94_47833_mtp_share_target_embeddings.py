#!/usr/bin/env python3
"""PN94 — MTP drafts always share target embeddings; embedding-width guard
narrowed to EAGLE drafts only.

Combined backport of vllm#47833 AND vllm#47953 (same semantic change; both
applied here as one hunk pair, per their shared root issue #47794) to
nightly-9e57de71 (dev1060), in
vllm/v1/spec_decode/llm_base_proposer.py::_maybe_share_embeddings.

Upstream history: PR #43957 added a width guard so EAGLE drafts shipping
their own differently-sized embed_tokens keep them. But the guard also
covered MTP drafts, whose projection is built for the TARGET embedding
width (e.g. Gemma4 MTP pre_projection takes 2 * backbone_hidden_size, draft
1024 vs target 2816) — a width mismatch then wrongly kept separate
embeddings and crashed init (#47794). #47833/#47953 restrict the width
guard to EAGLE drafts (detected via the has_own_embed_tokens attribute,
which only EAGLE draft models define); MTP drafts now always share the
target embedding.

For our method='mtp' on this pin: Qwen3_5MTP defines no
has_own_embed_tokens (verified in-image) -> MTP branch -> always share.
Target and MTP hidden sizes are equal on Qwopus3.6-27B, so today's runtime
behavior is unchanged; this removes the width/isinstance guard as a latent
failure mode and aligns us with upstream semantics.

Anchors verified byte-exact in-image 2026-07-13 (genesis P94/N9/N35 patch
other regions of this file — no overlap). Retire when upstream restricts
the width guard to EAGLE (is_eagle_model / hasattr-conjunct present) —
this patcher then self-retires.
"""
import pathlib
import sys

LOG = "[pn94-mtp-share-target-embeddings]"
TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py"
)
MARKER = "# PN94:"

SUB1_OLD = (
    "            share_embeddings = False\n"
    '            if hasattr(self.model, "has_own_embed_tokens"):\n'
)
SUB1_NEW = (
    "            share_embeddings = False\n"
    "            # PN94: vllm#47833 + #47953 — record the drafter family; the\n"
    "            # embedding-width guard below must only apply to EAGLE drafts.\n"
    '            is_eagle_model = hasattr(self.model, "has_own_embed_tokens")\n'
    "            if is_eagle_model:\n"
)

SUB2_OLD = (
    "            if share_embeddings:\n"
    "                draft_embed = self.model.model.embed_tokens\n"
    "                # Only share when both models use the same embedding width.\n"
    "                # Guard with isinstance so non-Tensor weights (e.g. in tests)\n"
    "                # are not affected — mirrors the weight-equality check above.\n"
)
SUB2_NEW = (
    "            if share_embeddings and is_eagle_model:\n"
    "                # PN94: vllm#47833 + #47953 — EAGLE draft models may ship\n"
    "                # their own embedding of a different width than the target\n"
    "                # (e.g. Eagle3MiniMaxM2), in which case the two must stay\n"
    "                # separate (see PR #43957). MTP draft models are excluded:\n"
    "                # their projection is built for the *target* embedding\n"
    "                # width, so they must always share the target embedding\n"
    "                # regardless of the draft checkpoint's own embed_tokens\n"
    "                # width (see issue #47794 — Gemma4 MTP).\n"
    "                draft_embed = self.model.model.embed_tokens\n"
    "                # Guard with isinstance so non-Tensor weights (e.g. in tests)\n"
    "                # are not affected — mirrors the weight-equality check above.\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"{LOG} FATAL: {TARGET} not present", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"{LOG} already applied (idempotent)")
        return 0

    # Self-retire: upstream landed either PR's form of the narrowing.
    if "def _maybe_share_embeddings" in text:
        body = text.split("def _maybe_share_embeddings", 1)[1][:8000]
        if (
            "is_eagle_model" in body
            or 'share_embeddings and hasattr(self.model, "has_own_embed_tokens")'
            in body
        ):
            print(
                f"{LOG} upstream drift: EAGLE-only width guard already present "
                f"— self-retire (no-op)"
            )
            return 0

    for name, old in (("family-record", SUB1_OLD), ("width-guard-narrowing", SUB2_OLD)):
        if old not in text:
            print(
                f"{LOG} FATAL: anchor-not-found ({name}) — upstream refactor; "
                f"re-derive before boot (MTP embed sharing falls back to the "
                f"width guard)",
                file=sys.stderr,
            )
            return 1
        if text.count(old) != 1:
            print(
                f"{LOG} FATAL: ambiguous anchor ({name}, {text.count(old)} "
                f"matches)",
                file=sys.stderr,
            )
            return 1

    text = text.replace(SUB1_OLD, SUB1_NEW, 1).replace(SUB2_OLD, SUB2_NEW, 1)
    TARGET.write_text(text)
    print(
        f"{LOG} applied: MTP drafts always share target embeddings; width "
        f"guard narrowed to EAGLE (vllm#47833 + #47953 backport)"
    )
    return 0


sys.exit(main())
