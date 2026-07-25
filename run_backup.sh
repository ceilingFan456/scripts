#!/bin/bash
#
# Sandbox backup runner (see backup.sh for the original manual commands).
#
# Uploads one of three sources to Azure Blob with azcopy:
#   home  -> /home/t-qimhuang  ->  <container>/backup_<MM_DD_YY>/home_backup
#   disk1 -> /datadisk         ->  <container>/backup_<MM_DD_YY>/disk1_backup
#   disk2 -> /data2            ->  <container>/backup_<MM_DD_YY>/disk2_backup
#
# The SAS token is NEVER stored in this script. It is read at runtime from
# $SAS_FILE (default ~/.azcopy_sas, chmod 600). Store it once with:
#
#   read -rs SAS && printf '%s' "$SAS" > ~/.azcopy_sas && chmod 600 ~/.azcopy_sas && unset SAS
#
# (paste only the query string, i.e. everything AFTER the "?", then press Enter)
#
# Usage:
#   ./run_backup.sh home
#   ./run_backup.sh disk1
#   ./run_backup.sh disk2
#
# Env overrides:
#   SAS_FILE      path to the token file       (default ~/.azcopy_sas)
#   BACKUP_DATE   destination folder suffix    (default today as MM_DD_YY)

set -euo pipefail

TARGET="${1:-}"
SAS_FILE="${SAS_FILE:-$HOME/.azcopy_sas}"
BACKUP_DATE="${BACKUP_DATE:-$(date +%m_%d_%y)}"
DEST_ROOT="https://singaporeteamstorage.blob.core.windows.net/shared/qiming/backup/backup_${BACKUP_DATE}"
LOG_DIR="$HOME/backup_logs/backup_${BACKUP_DATE}"

case "$TARGET" in
  home)  SRC="/home/t-qimhuang/*"; DEST="home_backup"  ;;
  disk1) SRC="/datadisk/*";        DEST="disk1_backup" ;;
  disk2) SRC="/data2/*";           DEST="disk2_backup" ;;
  *) echo "usage: $0 {home|disk1|disk2}" >&2; exit 2 ;;
esac

# Never upload the SAS token file itself as part of the home backup.
EXTRA=()
if [[ "$TARGET" == "home" ]]; then
  EXTRA+=(--exclude-path ".azcopy_sas;backup_logs")
fi

if [[ ! -r "$SAS_FILE" ]]; then
  echo "ERROR: SAS token file not found: $SAS_FILE" >&2
  echo "Create it with: read -rs SAS && printf '%s' \"\$SAS\" > $SAS_FILE && chmod 600 $SAS_FILE && unset SAS" >&2
  exit 1
fi

# Strip a leading '?' and any trailing newline the user may have pasted.
SAS="$(tr -d '\r\n' < "$SAS_FILE")"
SAS="${SAS#\?}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${TARGET}.log"

{
  echo "=== $(date -Is) starting backup: $TARGET ==="
  echo "source     : $SRC"
  echo "destination: ${DEST_ROOT}/${DEST}"
} | tee -a "$LOG"

set +e
sudo -n env AZCOPY_LOG_LOCATION="$LOG_DIR/azcopy_logs" \
  azcopy copy "$SRC" "${DEST_ROOT}/${DEST}?${SAS}" \
    --recursive --put-md5 "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
STATUS="${PIPESTATUS[0]}"
set -e

echo "=== $(date -Is) finished backup: $TARGET (exit $STATUS) ===" | tee -a "$LOG"
exit "$STATUS"
