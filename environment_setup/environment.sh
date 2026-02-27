#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/compose.yaml"
COMPOSE="docker compose -f $COMPOSE_FILE"

function load_openai_key() {
    if [ -f "$SCRIPT_DIR/.openapi_env" ]; then
        export OPENAI_API_KEY="$(grep -E '^OPENAI_API_KEY=' "$SCRIPT_DIR/.openapi_env" | head -n 1 | cut -d '=' -f2- | tr -d '\r\n')"
    fi
}

function start() {
    echo "Starting containers..."
    load_openai_key
    $COMPOSE up -d --build
    echo "Containers started."
}

function stop() {
    echo "Stopping containers..."
    $COMPOSE down
    echo "Containers stopped."
}

function restart() {
    echo "Restarting containers..."
    load_openai_key
    $COMPOSE down
    $COMPOSE up -d --build
    echo "Containers restarted."
}

function logs() {
    load_openai_key
    $COMPOSE logs -f
}

function remove_data() {
    echo "WARNING:"
    echo "This will remove:"
    echo "- Containers"
    echo "- Named volumes (ClickHouse + MinIO data)"
    echo ""
    read -p "Type YES to continue: " confirm

    if [ "$confirm" = "YES" ]; then
        echo "Stopping services and removing data volumes..."
        $COMPOSE down -v --remove-orphans
        echo "Data removal complete."
    else
        echo "Cleanup cancelled."
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    remove-data)
        remove_data
        ;;
    *)
        echo "Usage: ./environment.sh {start|stop|restart|logs|remove-data}"
        exit 1
        ;;
esac
