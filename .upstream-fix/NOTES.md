# Qwen3.5-VL `encode_image_slice` SIGSEGV on sm_87 — investigation notes

Branch: `fix/qwen3-vl-encode-image-slice-sm87`
Investigation date: 2026-04-29
Investigator: opus 4.7 1M agent on `edge-builder-1`

## Crash signature (ours)

- Build: `b8977` / commit `b1d5f5b44918c816478bdff60c07ca90529405f5` (sync : ggml)
- Hardware: Jetson Orin Nano + AGX, both Ampere `sm_87`, CUDA backend
- Model: Qwen3.5-VL 0.8B IQ4_XS + mmproj-F16
- Workload: vision encode of a single still frame (frigate-style snapshot)
- Crash: `rc=-11` (SIGSEGV) in `encode_image_slice`
- Reproducible: 100%, fast (~7 s/attempt × 3 retries)
- Workaround that **works**: `--no-mmproj-offload=1` (CPU mmproj path, ~150 s/clip)

## Root cause (high confidence)

This is **the same class of crash** documented in upstream issues:
- ggml-org/llama.cpp#17881 (Qwen3-VL crashes llama-server when ecoding image slice) — closed stale, unfixed
- ggml-org/llama.cpp#18181 (fit params does not take vision encoders into account) — open, the canonical tracker
- ggml-org/llama.cpp#21750 (server crash with large images, Qwen3.5/Qwen3-VL) — Vulkan-flavored variant

Both bisects in the threads land on the same area:
- #17881 (scottjg, RTX 4090): bisect points at PR ggml-org/llama.cpp#16653 (auto fit-params for context size)
- #18181: same auto-fit feature does not deduct mmproj memory from the budget

### Mechanism

1. `llama-server` startup runs `llama_params_fit` to autosize context against free VRAM,
   leaving a default 1024 MiB headroom (`--fit-target 1024`).
2. `mtmd` (mmproj) memory is **not** subtracted from the budget — `libcommon` doesn't link
   against `mtmd`, so the fit logic doesn't know about it.
3. KV cache + activations end up sized to consume nearly all VRAM.
4. When the vision encoder fires on first image, it tries to allocate the per-slice
   tensor and `cudaMalloc` returns failure (silent in release builds).
5. `ggml_vbuffer_tensor_alloc` then dereferences a null buffer → SIGSEGV.
   Stack trace from #17881:
   ```
   Thread 1 "llama-server" received signal SIGSEGV
   0x... in ggml_vbuffer_tensor_alloc (buf=0x0, tensor=...) at ggml/src/ggml-alloc.c:446
   ```

`sm_87` (Ampere Jetson) is not architecture-specific to the bug; the bug appears on RTX
4090 (`sm_89`) and 5090 (`sm_120`) too. We trip it on Jetson because Orin's unified
8/64 GiB pool gets squeezed harder than discrete cards: the fit logic gives most of it
to KV cache and the mmproj forward pass then has nothing left.

## Upstream candidate fix

**PR ggml-org/llama.cpp#21489 — "mtmd: fit_params now take into account mmproj"** by @ngxson.
- Status: OPEN, last activity 2026-04-11
- Author branch: `ngxson/llama.cpp` `xsn/mmproj_fit_params`
- HEAD commit: `e6386c7560c072ffba2faf06adb39a838a4d5e96`
- Adds `mtmd_get_memory_usage()` API; subtracts mmproj weight buffer + compute buffer
  from `fit_params_target[0]` before context creation.
- Diff is ~129/72 across 5 files in `tools/mtmd/` and `tools/server/`.

The PR has 4 conflicts against current master (master is 312 commits ahead of the PR
base). Conflicts are mostly mechanical:
- `tools/mtmd/clip.cpp`: 1 hunk
- `tools/mtmd/mtmd.cpp`: 2 hunks
- `tools/server/server-context.cpp`: 1 hunk — upstream master added
  `mparams.media_marker = get_media_marker();` line that the PR's parent doesn't have

The patch is preserved at `.upstream-fix/pr-21489-mmproj-fit-params.patch` in this branch
for the next agent that wants to rebase + cherry-pick.

## Why I did NOT cherry-pick onto this branch

1. The cherry-pick has 4 small conflicts that need either rebasing the PR or hand-merging.
2. I cannot build CUDA aarch64 here on the dev host. Per the task constraints I cannot
   claim a fix without a build + test.
3. The deeper-bump agent on the AGX is already rebuilding past `b8977`. If PR #21489
   lands upstream in their build window, they get it for free. If it doesn't, they
   already have the disposable build environment to apply this patch.

## Recommendation for the next person picking this up

If you need to actually deploy a fix (instead of waiting for upstream merge):

1. On AGX, a worktree at `/data/llama-build/upstream-investigate/` (or a fresh clone):
   ```
   git clone https://github.com/mihai-chiorean/turbo3-cuda.git
   cd turbo3-cuda
   git remote add upstream https://github.com/ggml-org/llama.cpp.git
   git fetch upstream master
   git checkout -b try/pr-21489 upstream/master
   git remote add ngxson https://github.com/ngxson/llama.cpp.git
   git fetch ngxson xsn/mmproj_fit_params
   git cherry-pick e6386c7560c072ffba2faf06adb39a838a4d5e96
   # resolve 4 trivial conflicts (mostly: keep new HEAD members, add PR's logic around them)
   ```
2. Build with the aarch64+CUDA recipe at
   `~/workspace/samples/deepstream-vision/docs/llama-cpp-jetson-build.md`.
3. Test with our exact failing config (Qwen3.5-VL 0.8B IQ4_XS + mmproj-F16, no
   `--no-mmproj-offload`). Expected: server should log `[mtmd] estimated memory usage of
   mmproj is X MiB` on startup and the image encode should complete instead of crashing.

If you DON'T want to deploy a fix and are OK with the workaround:
- Keep `--no-mmproj-offload=1`. The 150 s/clip CPU vision encode is a real cost but it is
  stable, and the upstream fix is in a single open PR with no merge ETA.
- Alternative cheap mitigation without a code fix: pass a much higher `--fit-target`
  (try `--fit-target 4096` or `--fit-target 6144`) so the auto-fit leaves enough VRAM
  headroom for the mmproj forward pass. This is what other users in #18181 confirmed
  works as a workaround. We have not validated this on Jetson Orin yet.

## What's surprising / non-obvious

The crash *looks* like a CUDA backend bug (SIGSEGV inside compute) but it is actually a
**libcommon planning bug** that happens to deref-null inside the CUDA allocator path.
Our `--no-mmproj-offload` workaround works not because of any CPU-vs-GPU vision-encoder
correctness difference, but because routing mmproj through CPU buffers entirely avoids
the VRAM allocation that would fail. The `sm_87` in the bug name is misleading — it's
not architecture-specific, we just happen to run on Jetson Ampere.
