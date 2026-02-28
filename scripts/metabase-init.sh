#!/bin/sh
set -e

METABASE_URL="${METABASE_URL:-http://metabase:3000}"
MB_ADMIN_EMAIL="${MB_ADMIN_EMAIL:-admin@example.com}"
MB_ADMIN_PASSWORD="${MB_ADMIN_PASSWORD:-admin1234}"
MB_ADMIN_FIRST_NAME="${MB_ADMIN_FIRST_NAME:-Admin}"
MB_ADMIN_LAST_NAME="${MB_ADMIN_LAST_NAME:-User}"
MB_SITE_NAME="${MB_SITE_NAME:-PLM Analytics}"

METABASE_DB_NAME="${METABASE_DB_NAME:-plm}"
CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-clickhouse}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-8123}"
CLICKHOUSE_DB="${CLICKHOUSE_DB:-plm}"
CLICKHOUSE_USER="${CLICKHOUSE_USER:-default}"
CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-}"
CLICKHOUSE_SSL="${CLICKHOUSE_SSL:-false}"

echo "Waiting for Metabase..."
attempts=0
max_attempts=30
while [ $attempts -lt $max_attempts ]; do
    attempts=$((attempts + 1))
    if curl -fsS "$METABASE_URL/api/health" | jq -e '.status=="ok"' >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if [ $attempts -ge $max_attempts ]; then
    echo "Metabase not ready after $max_attempts attempts."
    exit 1
fi

props_json="$(curl -fsS "$METABASE_URL/api/session/properties")"
setup_token="$(echo "$props_json" | jq -r '."setup-token"')"
has_user_setup="$(echo "$props_json" | jq -r '."has-user-setup"')"

session_id=""
if [ "$has_user_setup" != "true" ] && [ -n "$setup_token" ] && [ "$setup_token" != "null" ]; then
    echo "Running first-time Metabase setup..."
    setup_payload="$(jq -n \
        --arg token "$setup_token" \
        --arg email "$MB_ADMIN_EMAIL" \
        --arg password "$MB_ADMIN_PASSWORD" \
        --arg first_name "$MB_ADMIN_FIRST_NAME" \
        --arg last_name "$MB_ADMIN_LAST_NAME" \
        --arg site_name "$MB_SITE_NAME" \
        '{
            token: $token,
            user: {
                email: $email,
                password: $password,
                first_name: $first_name,
                last_name: $last_name
            },
            prefs: {
                site_name: $site_name
            }
        }'
    )"

    setup_resp="$(curl -sS -X POST "$METABASE_URL/api/setup" \
        -H "Content-Type: application/json" \
        -d "$setup_payload")"
    session_id="$(echo "$setup_resp" | jq -r '.id')"
    if [ -z "$session_id" ] || [ "$session_id" = "null" ]; then
        echo "Metabase setup failed. Response:"
        echo "$setup_resp"
        exit 1
    fi
else
    echo "Metabase already set up. Creating session..."
    session_id="$(curl -fsS -X POST "$METABASE_URL/api/session" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"$MB_ADMIN_EMAIL\",\"password\":\"$MB_ADMIN_PASSWORD\"}" | jq -r '.id')"
fi

if [ -z "$session_id" ] || [ "$session_id" = "null" ]; then
    echo "Failed to obtain Metabase session."
    exit 1
fi

existing_id="$(curl -fsS -H "X-Metabase-Session: $session_id" "$METABASE_URL/api/database" \
    | jq -r --arg name "$METABASE_DB_NAME" '.data[]? | select(.name==$name) | .id' | head -n1)"

if [ -n "$existing_id" ] && [ "$existing_id" != "null" ]; then
    echo "Metabase database '$METABASE_DB_NAME' already exists (id=$existing_id)."
    exit 0
fi

echo "Adding ClickHouse database '$METABASE_DB_NAME'..."
db_payload="$(jq -n \
    --arg name "$METABASE_DB_NAME" \
    --arg host "$CLICKHOUSE_HOST" \
    --argjson port "$CLICKHOUSE_PORT" \
    --arg db "$CLICKHOUSE_DB" \
    --arg user "$CLICKHOUSE_USER" \
    --arg password "$CLICKHOUSE_PASSWORD" \
    --argjson ssl "$CLICKHOUSE_SSL" \
    '{
        engine: "clickhouse",
        name: $name,
        details: {
            host: $host,
            port: $port,
            db: $db,
            user: $user,
            password: $password,
            ssl: $ssl
        }
    }'
)"

curl -fsS -X POST "$METABASE_URL/api/database" \
    -H "Content-Type: application/json" \
    -H "X-Metabase-Session: $session_id" \
    -d "$db_payload" >/dev/null

echo "Metabase setup complete."
