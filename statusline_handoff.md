# Claude Code 自訂狀態列 — 交接安裝說明

> 把這整份檔案貼給 / 給另一台 server 上的 Claude Code，請它照做即可。

請幫我在這台機器設定 Claude Code 的自訂狀態列（status line），需求如下。

## 1. 建立腳本 `~/.claude/statusline-command.sh`

內容完全照這份（逐字，不要改動）：

```bash
#!/bin/bash
input=$(cat)

cwd=$(echo "$input"    | jq -r '.cwd')
model=$(echo "$input"  | jq -r '.model.display_name')
used=$(echo "$input"   | jq -r '.context_window.used_percentage // empty')
effort=$(echo "$input" | jq -r '.effort.level // empty')
cost=$(echo "$input"   | jq -r '.cost.total_cost_usd // empty')
lim5h=$(echo "$input"  | jq -r '.rate_limits.five_hour.used_percentage // empty')
lim7d=$(echo "$input"  | jq -r '.rate_limits.seven_day.used_percentage // empty')
rst5h=$(echo "$input"  | jq -r '.rate_limits.five_hour.resets_at // empty')
rst7d=$(echo "$input"  | jq -r '.rate_limits.seven_day.resets_at // empty')

# --- ANSI colors (actual ESC bytes via $'...') ---
R=$'\033[0m'; BOLD=$'\033[1m'; DIM=$'\033[2m'
CYAN=$'\033[36m'; MAG=$'\033[35m'
GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'

# color by usage: green <50, yellow 50-80, red >=80
pctcol() {
  local p=$1
  if   [ "$p" -ge 80 ]; then printf '%s' "$RED"
  elif [ "$p" -ge 50 ]; then printf '%s' "$YEL"
  else                       printf '%s' "$GRN"; fi
}

# countdown to reset: "↻3h" / "↻12m" / "↻now"
fmt_reset() {
  local ts=$1 now secs
  [ -z "$ts" ] && return
  now=$(date +%s)
  secs=$(( ${ts%.*} - now ))
  [ "$secs" -le 0 ] && { printf '↻now'; return; }
  if [ "$secs" -ge 3600 ]; then printf '↻%dh' $(( secs / 3600 ))
  else                          printf '↻%dm' $(( secs / 60 )); fi
}

# one rate-limit segment: "5h:24% ↻3h" colored by usage
fmt_lim() {
  local p=$1 ts=$2 label=$3 pi rs
  [ -z "$p" ] && return
  pi=$(printf '%.0f' "$p")
  rs=$(fmt_reset "$ts")
  [ -n "$rs" ] && rs=" $rs"
  printf '%s%s:%s%%%s%s' "$(pctcol "$pi")" "$label" "$pi" "$rs" "$R"
}

header="${DIM}[$(whoami)@$(hostname -s) $(basename "$cwd")]${R}"
mdl="${BOLD}${CYAN}${model}${R}"

eff=""
[ -n "$effort" ] && eff=" | ${MAG}effort=${effort}${R}"

ctx=""
if [ -n "$used" ]; then
  ui=$(printf '%.0f' "$used")
  ctx=" | $(pctcol "$ui")ctx:${ui}%${R}"
fi

usd=""
[ -n "$cost" ] && usd=" | ${DIM}\$$(printf '%.2f' "$cost")${R}"

seg5h=$(fmt_lim "$lim5h" "$rst5h" "5h")
seg7d=$(fmt_lim "$lim7d" "$rst7d" "7d")
lim="$seg5h"
[ -n "$seg7d" ] && { [ -n "$lim" ] && lim="$lim · $seg7d" || lim="$seg7d"; }
[ -n "$lim" ] && lim=" | $lim"

printf '%s %s%s%s%s%s' "$header" "$mdl" "$eff" "$ctx" "$usd" "$lim"
```

## 2. 把 `statusLine` 設定「合併」進 `~/.claude/settings.json`

**重要：請保留檔案裡所有既有的 key，只新增/覆寫 `statusLine` 這一個欄位，不要整檔重寫成只剩這段。** 若檔案不存在就建立成只含這個 key 的 JSON。

要加入的欄位：

```json
"statusLine": {
  "type": "command",
  "command": "bash ~/.claude/statusline-command.sh"
}
```

建議用 `jq` 合併以免破壞既有內容：

```bash
[ -f ~/.claude/settings.json ] || echo '{}' > ~/.claude/settings.json
tmp=$(jq '.statusLine = {type:"command", command:"bash \(env.HOME)/.claude/statusline-command.sh"}' ~/.claude/settings.json) \
  && printf '%s\n' "$tmp" > ~/.claude/settings.json
```

## 3. 前置需求與驗證

- 這支腳本依賴 `jq`：先確認 `command -v jq`，沒有就安裝（或回報我）。
- 設定完用這筆假資料測試，應印出帶顏色的一行，格式為
  `[user@host dir] <model> | effort=… | ctx:NN% | $cost | 5h:NN% ↻Xh · 7d:NN% ↻Xh`：

```bash
echo '{"cwd":"/tmp/Demo","model":{"display_name":"Opus 4.8"},"context_window":{"used_percentage":65},"effort":{"level":"high"},"cost":{"total_cost_usd":0.42},"rate_limits":{"five_hour":{"used_percentage":85,"resets_at":'"$(($(date +%s)+7200))"'},"seven_day":{"used_percentage":33,"resets_at":'"$(($(date +%s)+129600))"'}}}' | bash ~/.claude/statusline-command.sh; echo
```

- 缺欄位（無訂閱額度 / 無 effort / 無花費）時，對應區段應自動省略，不報錯。

設定完成後請回報測試輸出。

---

## 顯示說明

狀態列會顯示：`使用者@主機 目錄`、模型、effort、context 用量 %、本 session 花費、以及訂閱制 5 小時 / 7 天額度用量（含重置倒數 `↻Xh`），並依用量上色（綠 <50% / 黃 50–80% / 紅 ≥80%）。每個欄位在缺資料時自動隱藏。
