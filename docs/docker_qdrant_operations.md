# Docker Qdrant Operations

## Scope

This document describes the Dockerized Qdrant service used by the Dry Retrieval Architecture.

The deployment is intended for local development, reproducible experiments, and comparative retrieval research.

## Deployment Files

```text
docker-compose.qdrant.yml
scripts/ingest_clerk_qdrant_server.py
scripts/validate_clerk_qdrant_server_parity.py
```

## Service Identity

```text
Compose project: rag-architecture
Container: rag-qdrant
Image: qdrant/qdrant:v1.19.1
REST endpoint: http://127.0.0.1:6333
gRPC endpoint: 127.0.0.1:6334
Dashboard: http://127.0.0.1:6333/dashboard
```

Ports are bound to the local loopback interface.

The default deployment must not be exposed to an untrusted network without authentication, TLS, and an explicit security review.

## Persistent Volumes

The deployment uses Docker named volumes:

```text
rag_architecture_qdrant_storage
rag_architecture_qdrant_snapshots
```

The named-volume approach avoids treating `qdrant-client` local-mode storage as server storage. The local-mode storage directory must never be mounted directly as `/qdrant/storage`.

## Start the Service

```powershell
docker compose `
-f ".\docker-compose.qdrant.yml" `
up `
-d
```

## Check Service Status

```powershell
docker compose `
-f ".\docker-compose.qdrant.yml" `
ps
```

## Check Logs

```powershell
docker compose `
-f ".\docker-compose.qdrant.yml" `
logs `
--tail 100 `
qdrant
```

## Check the REST API

```powershell
Invoke-RestMethod `
-Uri "http://127.0.0.1:6333/" `
-Method Get |
ConvertTo-Json `
-Depth 10
```

## Open the Dashboard

```powershell
Start-Process `
"http://127.0.0.1:6333/dashboard"
```

## Stop the Service

```powershell
docker compose `
-f ".\docker-compose.qdrant.yml" `
stop
```

Stopping the service preserves named volumes.

## Restart the Service

```powershell
docker compose `
-f ".\docker-compose.qdrant.yml" `
restart
```

## Remove the Container Without Deleting Data

```powershell
docker compose `
-f ".\docker-compose.qdrant.yml" `
down
```

The named volumes remain unless the `--volumes` option is used.

## Destructive Reset

The following operation deletes the Qdrant named volumes and all indexed points:

```powershell
docker compose `
-f ".\docker-compose.qdrant.yml" `
down `
--volumes
```

Do not run it unless a full rebuild is intended. The collection can be reconstructed from the checksummed embedding catalog and embedding shards.

## Ingest the Clerk Collection

Start the service first. Then run:

```powershell
python .\scripts\ingest_clerk_qdrant_server.py `
--qdrant-url "http://127.0.0.1:6333" `
--batch-size 128 `
--recreate
```

Expected result:

```text
Collection: rag_clerk_d984acfa6aae
Points before: 0
Points after: 5317
Documents: 2756
Collection status: green
```

## Idempotent Reingestion

```powershell
python .\scripts\ingest_clerk_qdrant_server.py `
--qdrant-url "http://127.0.0.1:6333" `
--batch-size 128
```

Expected result:

```text
Points before: 5317
Points after: 5317
```

Deterministic point IDs prevent duplicate collection growth.

## Validate Local-to-Server Parity

```powershell
python .\scripts\validate_clerk_qdrant_server_parity.py
```

The storage-level requirements are:

```text
Local point count: 5317
Server point count: 5317
Missing server IDs: 0
Unexpected server IDs: 0
Payload mismatches: 0
Exact storage parity: true
```

Retrieval-ranking parity is reported separately because local-mode and server-mode search implementations may have different execution paths.

## Validate Persistence

Stop and start the service:

```powershell
docker compose `
-f ".\docker-compose.qdrant.yml" `
stop

docker compose `
-f ".\docker-compose.qdrant.yml" `
start

Start-Sleep -Seconds 5
```

Then verify the exact point count:

```powershell
python -c "from qdrant_client import QdrantClient; c=QdrantClient(url='http://127.0.0.1:6333'); print(c.count(collection_name='rag_clerk_d984acfa6aae',exact=True).count); c.close()"
```

Expected result:

```text
5317
```

## Rebuild Principle

The authoritative sources are:

```text
embedding run catalog
embedding shard headers
embedding records
embedding shard manifests
source child chunks
```

The Qdrant collection is a derived, reconstructible index. The Docker volume is persistent operational state, not the primary scientific artifact.

## Future Experiment Collections

Comparative experiments should use deterministic collection names:

```text
rag_{dataset}_{architecture}_{configuration_hash}
```

Each collection manifest must record:

```text
dataset ID
architecture ID
architecture version
configuration fingerprint
embedding model and revision
chunking configuration
point count
document count
Qdrant image version
code revision
completion status
```

## Parallel Execution Safety

Multiple workers must not write incompatible configurations into the same collection.

Each experiment cell should use either:

* a unique deterministic collection, or
* an isolated temporary Qdrant service.

Parallel Colab workers should preferably produce immutable embedding artifacts for later centralized ingestion.
