FROM alpine:3.19

RUN apk add --no-cache curl jq

COPY scripts/metabase-init.sh /usr/local/bin/metabase-init.sh
RUN chmod +x /usr/local/bin/metabase-init.sh

ENTRYPOINT ["/usr/local/bin/metabase-init.sh"]
