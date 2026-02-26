#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# check_latest.sh — HexaGrid file freshness check
#
# Run this before any session that edits dashboard, API, or config files.
# Compares /mnt/project/ (project snapshot) vs /mnt/user-data/uploads/
# (files explicitly attached in conversation) and flags which is newer.
#
# Usage:
#   bash check_latest.sh
#   bash check_latest.sh --file index.html   # check specific file only
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR="/mnt/project"
UPLOAD_DIR="/mnt/user-data/uploads"
TARGET_FILE="${1:-}"   # optional: --file <name>

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

echo -e "${BLD}${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "${BLD}  HexaGrid — File Freshness Check${RST}"
echo -e "${BLD}${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo ""

# ── 1. Show all project files with timestamps ─────────────────────────────
echo -e "${BLD}Project snapshot  (/mnt/project/)${RST}"
if [[ -d "$PROJECT_DIR" ]]; then
    ls -lht "$PROJECT_DIR" | grep -v '^total' | awk '{printf "  %-40s %s %s %s\n", $9, $6, $7, $8}'
else
    echo -e "  ${RED}Directory not found${RST}"
fi

echo ""

# ── 2. Show all uploaded files with timestamps ────────────────────────────
echo -e "${BLD}Uploaded files    (/mnt/user-data/uploads/)${RST}"
if [[ -d "$UPLOAD_DIR" ]] && [[ -n "$(ls -A $UPLOAD_DIR 2>/dev/null)" ]]; then
    ls -lht "$UPLOAD_DIR" | grep -v '^total' | awk '{printf "  %-40s %s %s %s\n", $9, $6, $7, $8}'
else
    echo -e "  ${YEL}No uploaded files in this session${RST}"
fi

echo ""

# ── 3. Cross-check: flag files that exist in both locations ───────────────
echo -e "${BLD}Cross-check (files present in both locations)${RST}"
FOUND=0
for proj_file in "$PROJECT_DIR"/*; do
    fname=$(basename "$proj_file")
    upload_file="$UPLOAD_DIR/$fname"
    if [[ -f "$upload_file" ]]; then
        FOUND=1
        proj_ts=$(stat -c %Y "$proj_file")
        up_ts=$(stat -c %Y "$upload_file")
        proj_dt=$(stat -c %y "$proj_file" | cut -d'.' -f1)
        up_dt=$(stat -c %y "$upload_file" | cut -d'.' -f1)

        if [[ $up_ts -gt $proj_ts ]]; then
            echo -e "  ${GRN}UPLOAD IS NEWER${RST}  ${BLD}$fname${RST}"
            echo -e "    project: $proj_dt"
            echo -e "    upload:  ${GRN}$up_dt${RST}  <-- USE THIS"
        elif [[ $proj_ts -gt $up_ts ]]; then
            echo -e "  ${YEL}PROJECT IS NEWER${RST} ${BLD}$fname${RST}"
            echo -e "    project: ${YEL}$proj_dt${RST}  <-- USE THIS"
            echo -e "    upload:  $up_dt"
        else
            echo -e "  ${CYN}SAME TIMESTAMP${RST}  ${BLD}$fname${RST}  ($proj_dt)"
        fi
        echo ""
    fi
done

if [[ $FOUND -eq 0 ]]; then
    echo -e "  ${CYN}No filename overlap — project snapshot is the only source.${RST}"
    echo -e "  To provide a newer version, attach the file in the conversation."
fi

echo ""

# ── 4. Recommendation ─────────────────────────────────────────────────────
echo -e "${BLD}${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "${BLD}Rule: always use the file with the LATER timestamp.${RST}"
echo -e "If /mnt/user-data/uploads/ has a newer copy, use that."
echo -e "If only /mnt/project/ exists, use that."
echo -e "When in doubt: ${YEL}attach the file in the conversation${RST} — uploads"
echo -e "always reflect what is currently running on your machine."
echo -e "${BLD}${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
