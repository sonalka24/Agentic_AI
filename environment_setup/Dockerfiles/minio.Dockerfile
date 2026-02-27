FROM minio/mc:latest AS mc

FROM quay.io/minio/minio:latest

COPY --from=mc /usr/bin/mc /usr/bin/mc
COPY environment_setup/synthetic_data/ /seed/synthetic_data/
COPY environment_setup/scripts/minio-entrypoint.sh /usr/local/bin/minio-entrypoint.sh

RUN chmod +x /usr/local/bin/minio-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/minio-entrypoint.sh"]
CMD ["server", "/data", "--console-address", ":9001"]
