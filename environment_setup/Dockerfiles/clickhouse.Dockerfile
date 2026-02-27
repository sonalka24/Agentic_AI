FROM clickhouse/clickhouse-server:24.3

COPY environment_setup/sql/clickhouse-init.sql /docker-entrypoint-initdb.d/00-init.sql
