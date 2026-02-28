#!/bin/sh
set -e

MINIO_BUCKET="${MINIO_BUCKET:-plm}"
MINIO_PREFIX="${MINIO_PREFIX:-synthetic_data/}"
MINIO_SEED_DIR="${MINIO_SEED_DIR:-/seed/synthetic_data}"

if [ "${MINIO_PREFIX%/}" = "$MINIO_PREFIX" ]; then
    MINIO_PREFIX="${MINIO_PREFIX}/"
fi

minio "$@" &
minio_pid=$!

trap 'kill "$minio_pid"' INT TERM

attempts=0
max_attempts=20
while [ $attempts -lt $max_attempts ]; do
    attempts=$((attempts + 1))
    if mc alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if [ $attempts -ge $max_attempts ]; then
    echo "MinIO not ready after $max_attempts attempts."
    wait "$minio_pid"
    exit 1
fi

# Fix accidental double-nesting (synthetic_data/synthetic_data)
if mc ls "local/$MINIO_BUCKET/$MINIO_PREFIX" >/dev/null 2>&1; then
    if mc ls "local/$MINIO_BUCKET/${MINIO_PREFIX}synthetic_data" >/dev/null 2>&1; then
        echo "Fixing nested synthetic_data folder..."
        mc cp --recursive "local/$MINIO_BUCKET/${MINIO_PREFIX}synthetic_data/." \
            "local/$MINIO_BUCKET/$MINIO_PREFIX" >/dev/null 2>&1 || true
        mc rm --recursive --force "local/$MINIO_BUCKET/${MINIO_PREFIX}synthetic_data" >/dev/null 2>&1 || true
    fi
fi

if [ ! -f /data/.seeded ]; then
    mc mb --ignore-existing "local/$MINIO_BUCKET" >/dev/null
    if [ -d "$MINIO_SEED_DIR" ]; then
        mc cp --recursive "$MINIO_SEED_DIR"/. "local/$MINIO_BUCKET" >/dev/null
    fi
    touch /data/.seeded
fi

wait "$minio_pid"
