# Session 27 — QUALITY: Norm Correction, Validation Infrastructure, Feature Parity

## READ FIRST (MANDATORY)

1. **Read `AGENTS.md`** — contains ALL rules, architecture, dead ends, file locations, benchmarking protocol. Follow it EXACTLY.
2. **Read this entire prompt** before starting any work.
3. Create branch **`session/27-quality`** from `release/cuda-optimized`.
4. The Obsidian vault is at `/mnt/c/vaults/forge/` — search it for any context you need.
5. GPU is an RTX 5090 32GB (SM120). CUDA 12.8. WSL2 Ubuntu 24.04.

---

## CONTEXT — WHERE WE ARE AFTER S26

### Session 26 Delivered
- SM120 D=256 generation bug FIXED — LUT disabled for D=256 (NVIDIA codegen bug NVBUG 5218000/5288270)
- turbo4/turbo1.5 vec_dot Q_reg bug FIXED — now uses q8_1 Q correctly
- Block-128 storage VALIDATED and committed — turbo3 3.125 bpv (5.12x), turbo2 2.125 bpv (7.53x)
- bpv numbers corrected in README
- ALL 4 turbo types beat q8_0 at both short AND 32K
- Pushed to `release/cuda-optimized`

### Current S26 Numbers (RTX 5090, Qwen 3.5 27B Q6_K, block-128)

| Type | bpv | Short | 32K | PPL 512 |
|------|:---:|:-----:|:---:|:-------:|
| q8_0 | 8.5 | 64.06 | 54.01 | 6.759 |
| turbo4 | 4.25 | 65.33 | 58.03 | 6.825 |
| turbo3 | 3.125 | 65.14 | 56.28 | 6.852 |
| turbo2 | 2.125 | 64.13 | 58.48 | 7.080 |
| turbo1.5 | 2.00 | 64.00 | 55.62 | 7.312 |

### NIAH Quality (3090 Ti, SM86)
- turbo3: 86.4% (57/66) — **beats q8_0** (84.8%, 56/66)
- turbo2: 78.8% (52/66)
- Multi-key: 100% for q8_0/turbo3/turbo2 through 32K

### S26 Discoveries & Lessons
- **SM120 NVIDIA codegen bug**: NVBUG 5218000/5288270. LUT at D=256 with 168 registers triggers miscompilation. Tested CUDA 12.8-13.2 — all affected. Workaround: LUT disabled for D=256.
- **Benchmark variance**: Warmup 1 rep (discard), then measure 5 reps. GPU boost clock fluctuates — take best of stable runs.
- **Qwen 3.5 thinking models**: Need `max_tokens: 200+` for generation tests.
- **PPL alone is insufficient**: S26 proved PPL can be perfect while generation produces garbage (different code paths). Always test generation too.

### The Feature Parity Gap (TheTom vs Us)
TheTom's repo: `/home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/`

**What he has that we don't:**
1. **Norm correction** (spiritbuun) — -1.17% PPL. **Task 1.**
2. **50-chunk wikitext-103 validation** — gold standard quality proof. **Task 2.**
3. **Skip rate measurement** — direct sparse V skip %. **Task 3.**
4. **Quality gate script** — automated pre-commit check. **Task 4.**
5. KL divergence, cross-format sparse V — deferred to S28.

---

## TASK 1: Norm Correction (spiritbuun) — THE BIG QUALITY WIN

### Background
TheTom reports **-1.17% PPL on CUDA** with norm correction. This rescales the reconstructed vector to exactly match the original's magnitude.

**IMPORTANT CHECK FIRST**: HyperionMS2040's block-128 SET_ROWS fix (committed in S26) MAY already include norm correction. Read `set-rows.cu` and look for:
```cuda
const float recon_norm = sqrtf(s_recon_sq);
const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
```
If present → norm correction is DONE. Just verify PPL improvement vs the pre-S26 baseline.

### What Norm Correction Does
```
grp_norm   = ||x_rotated||            (original rotated vector norm)
recon_norm = ||centroids[indices]||    (reconstructed centroid vector norm)
corrected_norm = grp_norm / recon_norm (rescaled to match original magnitude)
```

### Where To Implement (if not already present)
**File**: `ggml/src/ggml-cuda/set-rows.cu`

For each turbo type's SET_ROWS kernel, after quantizing to centroid indices:
1. Each thread has `c = CENTROIDS[idx]`
2. Compute `recon_sq = c * c`
3. Warp reduce sum (same `__shfl_xor` pattern as grp_norm)
4. `recon_norm = sqrtf(s_recon_sq)`
5. `corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm`
6. Store `corrected_norm` as block norm

### Testing Protocol
1. PPL ctx=512 turbo3 BEFORE change → baseline
2. Implement norm correction
3. PPL ctx=512 and ctx=2048 → expect improvement
4. If improves: verify turbo2, turbo4, turbo1.5
5. Speed regression check (SET_ROWS not in decode path — should be zero)
6. Commit

---

## TASK 2: Download Wikitext-103 and Run 50-Chunk Validation

### Why
TheTom's gold standard: 50 chunks of wikitext-103 at 32K context (CI ±0.021). We used 8 chunks of wikitext-2 up to 16K.

### Steps
1. Download:
   ```bash
   cd /home/erol/ai/turboquant
   python3 -c "
   from huggingface_hub import hf_hub_download
   hf_hub_download(repo_id='Salesforce/wikitext', filename='wikitext-103-raw-v1/wiki.test.raw',
                   repo_type='dataset', local_dir='.')
   "
   ```

2. Run (start 32K overnight, shorter contexts during the day):
   ```bash
   MODEL=/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf
   WIKI103=/home/erol/ai/turboquant/wikitext-103-raw-v1/wiki.test.raw

   for CTX in 512 2048 8192 32768; do
     for TYPE in turbo3 turbo2 turbo1.5 q8_0; do
       echo "=== $TYPE ctx=$CTX ==="
       ./build/bin/llama-perplexity -m $MODEL -f $WIKI103 -c $CTX \
         -ctk $TYPE -ctv $TYPE -fa on --chunks 50 -ngl 99 --no-mmap 2>&1 | grep "Final"
     done
   done
   ```

3. **Critical check**: Does the PPL delta between turbo types and q8_0 GROW with context length? If turbo3-q8_0 delta at 32K is larger than at 512, the sparse V threshold may be too aggressive at long context.

### Runtime
~3 hours per type at 32K. 4 types × 4 contexts = 16 runs. The 32K runs should go overnight.

---

## TASK 3: Port Skip Rate Measurement

### What
Direct measurement of sparse V skip percentage per layer per threshold.

### How
1. Copy TheTom's script:
   ```bash
   cp /home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/scripts/measure_skip_rate.py quality-tests/
   ```

2. Run at all thresholds:
   ```bash
   for T in 1e-6 5e-3 1e-2; do
     echo "=== threshold=$T ==="
     /home/erol/miniconda3/envs/tq/bin/python quality-tests/measure_skip_rate.py \
       --threshold $T --contexts 512,2048,4096,8192
   done
   ```

3. TheTom's reference (1e-6, Qwen3-1.7B): 9.1% at 512, 20.7% at 2K, 28.4% at 4K. Our 5e-3/1e-2 should be higher.

---

## TASK 4: Create Quality Gate Script

### What
Automated pre-merge check: PPL + speed + generation.

### Create `quality-tests/quality-gate.sh`:
```bash
#!/bin/bash
set -e
MODEL=${MODEL:-/home/erol/ai/turboquant/models/opus-v2-Q6_K.gguf}
WIKI=$(find /home/erol/ai/turboquant -name "wiki.test.raw" 2>/dev/null | head -1)

echo "=== TurboQuant Quality Gate ==="

# 1. PPL (turbo3 ctx=512 < 6.89)
PPL=$(./build/bin/llama-perplexity -m $MODEL -f $WIKI -c 512 -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99 --no-mmap 2>&1 | grep "Final estimate" | awk '{print $4}')
echo "PPL: $PPL"
(( $(echo "$PPL > 6.89" | bc -l) )) && echo "FAIL: PPL" && exit 1

# 2. Speed (warmup + measure, turbo3 short > 55 tok/s)
./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 1 -p 0 -n 128 -mmp 0 > /dev/null 2>&1
SPEED=$(./build/bin/llama-bench -m $MODEL -fa 1 -ctk turbo3 -ctv turbo3 -d 0 -ngl 99 -t 1 -r 3 -p 0 -n 128 -mmp 0 2>&1 | grep "tg128" | awk '{print $(NF-2)}')
echo "Speed: $SPEED tok/s"
(( $(echo "$SPEED < 55.0" | bc -l) )) && echo "FAIL: speed" && exit 1

# 3. Generation (Qwen 9B D=256 turbo3 — must produce non-empty content)
./build/bin/llama-server -m /home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf \
  -ctk turbo3 -ctv turbo3 -fa on -ngl 99 -c 4096 --port 8091 --no-mmap --log-disable &
sleep 30
CONTENT=$(curl -s http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200,"temperature":0}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:50])" 2>/dev/null)
pkill -f "llama-server.*8091" 2>/dev/null; sleep 3
echo "Generation: '$CONTENT'"
[ -z "$CONTENT" ] && echo "FAIL: empty generation" && exit 1

echo "=== PASS ==="
```

---

## TASK 5: Re-run NIAH on 5090

### Why
S25 NIAH data from 5090 is INVALID (collected before D=256 fix). Need fresh data.

### How
```bash
# For each type: start server, run NIAH, kill, next
for TYPE in q8_0 turbo3 turbo2 turbo1.5; do
  echo "=== NIAH $TYPE ==="
  ./build/bin/llama-server -m /home/erol/ai/turboquant/models/Qwen3.5-9B-Q8_0.gguf \
    -ctk $TYPE -ctv $TYPE -fa on -ngl 99 -c 65536 --port 8090 --no-mmap --log-disable &
  sleep 30
  python3 quality-tests/niah_test.py --port 8090 --label ${TYPE}_5090 \
    --contexts "4096,8192,16384,32768" --depths "10,25,50,75,90" --reps 1
  pkill -f "llama-server.*8090"; sleep 20
done
```

turbo1.5 should now produce real results (Q_reg bug fixed in S26).

---

## TASK 6: Update Documentation and Vault

### AGENTS.md
- Add S26 final results (block-128, both bug fixes, all types beat q8_0)
- Add S27 results (norm correction, 50-chunk, skip rates)
- Update bpv in Q Format table (turbo3 3.125, turbo2 2.125)
- Add SM120 D=256 codegen bug to Known Issues

### README.md
- Add norm correction PPL story (if applicable)
- Add quality validation section (50-chunk wikitext-103, NIAH)

### Vault
- `01 Sessions/Session 27.md`
- `03 Benchmarks/Benchmark Hub.md`
- `00 Dashboard/Project Status.md`

---

## THETOM'S REFERENCE REPO
```
/home/erol/ai/turboquant/research/llama-cpp-turboquant/.trash/research/repos/TheTom-turboquant_plus/
```
| File | What It Has |
|------|------------|
| `docs/quality-benchmarks.md` | Norm correction PPL data |
| `docs/threshold-ablation.md` | Threshold sweep (1e-4 to 1e-8) |
| `docs/long-context-sparse-v-validation.md` | 50-chunk methodology |
| `scripts/measure_skip_rate.py` | Skip rate script |
| `scripts/turbo-quality-gate.sh` | His quality gate |
| `scripts/niah_test.py` | His NIAH (44KB comprehensive) |

---

## MODELS

| Model | Path | Use For |
|-------|------|---------|
| Qwen 3.5 27B Q6_K | `models/opus-v2-Q6_K.gguf` | PPL, speed |
| Qwen 3.5 9B Q8_0 | `models/Qwen3.5-9B-Q8_0.gguf` | NIAH, generation |
| Llama-3.3-8B Q6_K | `models/allura-forge_Llama-3.3-8B-Instruct-Q6_K.gguf` | NIAH D=128 |

---

## WHAT NOT TO DO
- Do NOT change VEC kernel code structure (168 regs ceiling, S25)
- Do NOT re-enable LUT for D=256 on SM120 (NVIDIA bug unfixed)
- Do NOT run benchmarks in background (Rule 11)
- Do NOT use max_tokens < 200 with Qwen 3.5
- Do NOT skip generation check — PPL alone is insufficient (S26 lesson)
- Do NOT trust first rep of llama-bench — always warmup then measure

---

## SUCCESS CRITERIA

- [ ] Norm correction checked/implemented — PPL improved or unchanged
- [ ] 50-chunk wikitext-103 PPL recorded (all types, ctx=512/2048/8192/32768)
- [ ] Skip rate measured at 1e-6, 5e-3, 1e-2
- [ ] Quality gate script created and passing
- [ ] NIAH re-run on 5090 (all types including turbo1.5)
- [ ] All docs and vault updated
- [ ] Pushed to `release/cuda-optimized`

---

## ESTIMATED EFFORT

| Task | Time |
|------|------|
| Norm correction | 2-3 hours |
| Wikitext-103 50-chunk | 1 hour setup + 12 hours overnight |
| Skip rate measurement | 1-2 hours |
| Quality gate | 1 hour |
| NIAH 5090 re-run | 2-3 hours |
| Documentation | 1 hour |
| **Total** | **~10 hours active + 12 hours overnight** |
