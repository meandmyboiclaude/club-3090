# Club-3090 vLLM Image — Build + Distribution

The vLLM image workflow (`.github/workflows/build-vllm-image.yml`) builds on
GitHub-hosted Ubuntu, applies vendored overlays via Dockerfile, pushes a dated
nightly tag to GHCR, then promotes the `:latest` and `:nightly-stable` aliases
to point at the just-built dated tag. No self-hosted runner required.

The release-pinned tag `:club-vX.Y.Z` is created automatically when the workflow
runs on a Git tag push (e.g. `v0.7.0`).

## Tag conventions

| Tag | Mutable? | What it represents | Recommended for |
|---|---|---|---|
| `nightly-YYYYMMDD-clubNNNN` | Immutable | A specific build of upstream nightly + our overlays | Reproducibility; pin in `VLLM_IMAGE` to lock to a known state |
| `:latest` | Mutable | The most-recent dated nightly | Users who want bleeding-edge; expect occasional breakage |
| `:nightly-stable` | Mutable | Same target as `:latest` today; reserved for future divergence | Same as `:latest` for now |
| `:club-vX.Y.Z` | Immutable (per release) | Built on the v0.7.0 tag push and never moves | Users who want verified releases; this is the recommended path |

## Why no smoke-gating?

The workflow originally tried to gate `:latest` promotion on a self-hosted GPU
runner running `verify-full.sh` + a 3-prompt smoke. That added:

- A dependency on infra (the runner) we don't maintain.
- A permission requirement (`administration: read` to list runners) the default `GITHUB_TOKEN` lacks.
- A failure mode where `:latest` never gets promoted if no runner is registered.

We dropped smoke-gating in v0.7.1 (issue #135) because:

1. The build step itself catches the most common failure modes (missing patch
   source, invalid Dockerfile, overlay path mismatches).
2. Users who want verified images use `:club-vX.Y.Z` release tags, not `:latest`.
3. The Docker Hub convention is that `:latest` = "most recent, no guarantees".

If a self-hosted GPU runner ever becomes available, smoke-gating can be layered
on top of this workflow as a separate post-build job — the simpler design today
doesn't preclude it.

## Using the GHCR image

The pre-built GHCR image is **opt-in**. Default launches use the upstream vLLM
nightly SHA resolved from `scripts/lib/profiles/engines/<engine-id>.yml →
install.spec` (the standard `vllm/vllm-openai:nightly-<sha>` ref).

To use the verified release image:

```bash
VLLM_IMAGE=ghcr.io/noonghunna/vllm-club3090:club-v0.7.0 \
  bash scripts/launch.sh --variant vllm/dual
```

To use the bleeding-edge `:latest` (rebuilt on every overlay change + weekly):

```bash
VLLM_IMAGE=ghcr.io/noonghunna/vllm-club3090:latest \
  bash scripts/launch.sh --variant vllm/dual
```

`VLLM_IMAGE` is a full image-reference override. The launcher still exports
`VLLM_NIGHTLY_SHA` from the matching EngineProfile, but Docker Compose uses
`VLLM_IMAGE` first when present.

## Workflow triggers

The workflow runs on:

- **`workflow_dispatch`** — manual `gh workflow run build-vllm-image.yml` for ad-hoc rebuilds
- **`schedule`** — weekly, Sunday 00:00 UTC
- **`push`** to master when `.github/workflows/build-vllm-image.yml`, `docker/vllm-club3090/**`, or `models/*/vllm/patches/**` change
- **`push`** of any tag matching `v0.7.*`, `v0.[8-9].*`, or `v[1-9]*` — adds the `:club-vX.Y.Z` release tag

## Workflow permissions

`GITHUB_TOKEN` with default `contents: read` + `packages: write` is sufficient.
No `administration: read` is needed (we removed the runner detection that
required it). No personal access tokens required.

## Manual `:latest` bootstrap (recovery)

If `:latest` ever gets out of sync with the most-recent dated nightly (e.g.
during this workflow's redesign migration), you can manually re-tag a dated
nightly as `:latest` without re-building:

```bash
docker login ghcr.io --username "${GITHUB_USERNAME}" --password-stdin <<< "${GITHUB_TOKEN}"

docker buildx imagetools create \
  -t ghcr.io/noonghunna/vllm-club3090:latest \
  -t ghcr.io/noonghunna/vllm-club3090:nightly-stable \
  ghcr.io/noonghunna/vllm-club3090:nightly-YYYYMMDD-clubXXXX
```

Replace `nightly-YYYYMMDD-clubXXXX` with the dated tag you want to bless as
`:latest`. List recent tags via:

```bash
gh api -H 'Accept: application/vnd.github+json' \
  /users/noonghunna/packages/container/vllm-club3090/versions | \
  jq -r '.[] | .metadata.container.tags[]' | head -20
```

(This requires `read:packages` scope on the token. Without it, list tags via the
GHCR web UI: https://github.com/noonghunna/club-3090/pkgs/container/vllm-club3090.)

## Retention

The scheduled workflow keeps:

- `:latest`
- `:nightly-stable`
- every `:club-v*` release tag
- dated `nightly-YYYYMMDD-clubNNNN` tags from the last four weeks

Older dated nightly versions are deleted by the retention job (runs only on the
weekly schedule + `workflow_dispatch`, not on every push).
