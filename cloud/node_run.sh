#!/bin/bash
# Executed on the EC2 node through SSM: one federated run = 1 server + K client containers.
set -u; BUCKET=$1 RUN=$2 ARM=$3 K=$4 SEED=$5 Z=$6 RATIO=$7
E="-e BUCKET=$BUCKET -e RUN=$RUN -e ARM=$ARM -e K=$K -e SEED=$SEED -e Z=$Z -e RATIO=$RATIO"
for j in $(seq 0 $((K-1))); do docker run -d --rm --name c$j --memory 160m $E -e ROLE=client -e CID=$j securefl >/dev/null; done
docker run -d --name srv --memory 400m $E -e ROLE=server securefl >/dev/null
: > /tmp/stats.txt
while [ "$(docker inspect -f '{{.State.Running}}' srv 2>/dev/null)" = "true" ]; do
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.NetIO}}' >> /tmp/stats.txt; sleep 2; done
code=$(docker inspect -f '{{.State.ExitCode}}' srv); docker logs srv 2>&1 | tail -3; docker rm srv >/dev/null
aws s3 cp /tmp/stats.txt s3://$BUCKET/stats/$RUN.csv --quiet --region eu-west-1 || true
echo "exit=$code"
