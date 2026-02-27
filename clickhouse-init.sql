CREATE DATABASE IF NOT EXISTS lh_gold;

CREATE TABLE IF NOT EXISTS lh_gold.products (
  product_id String,
  name String,
  category String,
  lifecycle_stage String,
  status String,
  owner String,
  created_at DateTime,
  updated_at DateTime
)
ENGINE = MergeTree
ORDER BY (product_id);

CREATE TABLE IF NOT EXISTS lh_gold.lifecycle_events (
  event_id String,
  product_id String,
  event_type String,
  event_ts DateTime,
  details String
)
ENGINE = MergeTree
ORDER BY (product_id, event_ts);

INSERT INTO lh_gold.products (product_id, name, category, lifecycle_stage, status, owner, created_at, updated_at) VALUES
  ('P-1001', 'Aquila Sensor', 'IoT', 'Design', 'Active', 'eng-team', now() - INTERVAL 90 DAY, now() - INTERVAL 7 DAY),
  ('P-1002', 'Orion Hub', 'Edge', 'Prototype', 'Active', 'r&d', now() - INTERVAL 60 DAY, now() - INTERVAL 2 DAY),
  ('P-1003', 'Nimbus Gateway', 'Networking', 'Production', 'Active', 'ops', now() - INTERVAL 365 DAY, now() - INTERVAL 1 DAY);

INSERT INTO lh_gold.lifecycle_events (event_id, product_id, event_type, event_ts, details) VALUES
  ('E-2001', 'P-1001', 'spec_created', now() - INTERVAL 85 DAY, 'Initial specs drafted'),
  ('E-2002', 'P-1001', 'design_review', now() - INTERVAL 70 DAY, 'Design review passed'),
  ('E-2003', 'P-1002', 'prototype_built', now() - INTERVAL 30 DAY, 'Prototype build v1'),
  ('E-2004', 'P-1003', 'release', now() - INTERVAL 300 DAY, 'GA release 1.0'),
  ('E-2005', 'P-1003', 'patch', now() - INTERVAL 10 DAY, 'Patch 1.0.3');
