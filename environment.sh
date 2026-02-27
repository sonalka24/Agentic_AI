#!/bin/bash

set -euo pipefail

COMPOSE_FILE="compose.yaml"
COMPOSE="docker compose -f $COMPOSE_FILE"

function start() {
    echo "Starting containers..."
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
    $COMPOSE down
    $COMPOSE up -d --build
    echo "Containers restarted."
}

function logs() {
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
