# SecureFL-IDS: Federated Intrusion Detection with Differentially Private, Sparsified Updates on AWS

**Nemi Ishwarlal Vikani (24303046)**  
MSc in Cloud Computing, National College of Ireland

SecureFL-IDS is a federated intrusion detection system evaluated on the official UNSW-NB15 training and testing partitions (175,341 training rows and 82,332 test rows) and deployed on AWS (`eu-west-1`) using EC2, S3, CloudWatch, Systems Manager, and Docker, provisioned with Terraform. Each client trains a NumPy MLP locally with per-example gradient clipping and Gaussian noise (DP-SGD accounted via Rényi Differential Privacy at $\delta = 10^{-5}$) and uploads only the largest $k = 10\%$ of coordinates in float16 with error feedback.

## Project Structure

- `securefl/` - Core package (`data.py` preprocessing and Dirichlet non-IID partitioning, `model.py` MLP and vectorised per-example clipped gradients, `fl.py` top-$k$ sparsification with error feedback, RDP accounting, and FedAvg loop)
- `scripts/` - Experiment matrix (`matrix.py`), local runner (`run_local.py`, `run_all.sh`), AWS campaign driver (`cloud_campaign.py`), statistical analysis and plot generator (`analyse.py`, `make_macros.py`), and independent verification suite (`verify.py`)
- `cloud/` - Container runtime (`Dockerfile`, `cloud_node.py`, `node_run.sh`)
- `terraform/` - AWS infrastructure (`main.tf`: S3 parameter bus, EC2 `t3.small` worker, IAM instance profile for SSM/S3/CloudWatch, CloudWatch log group `/securefl-ids/runs` and dashboard `SecureFL-IDS`)
- `tests/` - Unit tests (`test_core.py`)
- `data/SHA256SUMS` - SHA-256 checksums for the official UNSW-NB15 CSV partitions
- `results/` - 320 local run JSONs (`results/local/`), 18 live AWS cloud run JSONs (`results/cloud/`), and aggregated summary (`results/summary.json`)
- `report_assets/` - Generated LaTeX tables, macros, and 300 dpi figures

## Quick Start

```bash
./setup.sh
. .venv/bin/activate
OMP_NUM_THREADS=1 python -m pytest -q tests
python scripts/verify.py
```

## Reproducing the Local Campaign (320 Runs)

```bash
JOBS=6 bash scripts/run_all.sh
python scripts/verify.py
```

## Reproducing the Live AWS Campaign

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-west-1

cd terraform && terraform init && terraform apply && cd ..
python scripts/cloud_campaign.py setup
NSEEDS=5 python scripts/cloud_campaign.py campaign
REPORT_DIR=report_assets python scripts/analyse.py
```
