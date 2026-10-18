#!/bin/bash
# B4 ATTENTION-SHARE GATE window: profile a LATE (pos>=100k) chunk of p131k with the
# finer mla_proj/mla_attn and kda_proj/kda_attn class breakdown. Default-off enhancement.
# Method (pinned): chunk at position >=100k of p131k; profile separate proj/attn halves.
set -eo pipefail

restore_serve() {
  cd /Users/studio/tf-speed-1016
  nohup env GLM_LOAD_LIMIT_GB=190 ./zig-out/native/bin/tensorfold-native serve \
    /Users/studio/Models/GLM-5.3-Flash-oQ4e-mtp --name GLM-5.3-Flash-oQ4e \
    --host 127.0.0.1 --port 8008 --context 262144 --no-update-check \
    > /Users/studio/tf-lead/serve-b4-restore.log 2>&1 < /dev/null &
  disown
  sleep 45
  echo "--- RESTORE HEALTH (retry x3) ---"
  for i in 1 2 3; do
    H=$(curl -s -m 8 http://127.0.0.1:8008/health | head -c 200); echo "attempt $i: $H"
    case "$H" in *'"status": "ok"'*) break;; esac
    sleep 20
  done
}
trap restore_serve EXIT

BR=/Users/studio/tf-speed-1016
LOG=/Users/studio/tf-lead/b4-window.log
REC=/Users/studio/tf-lead/b4-prompt-profile.log
PROMPT=/Users/studio/tf-zig-meas/prompts/long.json
MODEL=/Users/studio/Models/GLM-5.3-Flash-oQ4e-mtp

exec > "$LOG" 2>&1
echo "=== [1] stop resident serve (LoadLimit GLM_LOAD_LIMIT_GB=215 fallback) ==="
sudo launchctl bootout system /Library/LaunchDaemons/local.tensorfold.glm8008.plist 2>&1 || true
pkill -f "tensorfold-native serve" 2>&1 || true
sleep 8
memory_pressure -Q | tail -1

echo "=== [2] pre-window HEAD bits ==="
ls -la $BR/zig-out/bin/tf-glm-run 2>&1 | tail -1
md5sum $BR/zig-out/bin/tf-glm-run

echo "=== [3] B4 LATE-chunk profile at pos=100000 (chunk rows=16384, p143000) ==="
echo "--- [3a] LATE chunk profile (pos=100000) ---"
GLM_LOAD_LIMIT_GB=190 GLM_CAP=262144 GLM_MAX=4 GLM_DEPTHS=0 GLM_RUNS=1 \
  GLM_PROMPT_PROFILE=10 GLM_PROMPT_PROFILE_POS=100000 GLM_PROMPT_PROFILE_ROWS=16384 \
  $BR/zig-out/bin/tf-glm-run $MODEL $PROMPT > $REC 2>&1
PROFILE_EXIT=$?
echo "PROFILE-EXIT=$PROFILE_EXIT"
tail -50 $REC

echo "=== [4] extract per-class numbers (mask, ms, %) ==="
python3 <<'EOF'
import re
with open('/Users/studio/tf-lead/b4-prompt-profile.log') as f:
    text = f.read()
rows = []
for line in text.splitlines():
    m = re.match(r'prompt profile \d+ rows @ pos \d+: (\S+)\s+\s*([\d.]+) ms \(min ([\d.]+), max ([\d.]+)\)(?:\s+class\s+([\d.]+) ms\s+([\d.]+)%)?', line)
    if m:
        rows.append({
            'name': m.group(1),
            'med_ms': float(m.group(2)),
            'min_ms': float(m.group(3)),
            'max_ms': float(m.group(4)),
            'class_ms': float(m.group(5)) if m.group(5) else None,
            'class_pct': float(m.group(6)) if m.group(6) else None,
        })
# Print the table
print("name    med(ms)   min(ms)   max(ms)   class(ms)  class(%)")
for r in rows:
    if r['class_ms'] is None:
        print(f"  {r['name']:12s} {r['med_ms']:8.2f}  {r['min_ms']:8.2f}  {r['max_ms']:8.2f}")
    else:
        print(f"  {r['name']:12s} {r['med_ms']:8.2f}  {r['min_ms']:8.2f}  {r['max_ms']:8.2f}   {r['class_ms']:8.2f}    {r['class_pct']:5.1f}")
EOF

echo "=== [5] restore handled by EXIT trap ==="
