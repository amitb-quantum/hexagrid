#!/usr/bin/env python3
"""
tpu_collector_agent.py — HexaGrid TPU Telemetry Collector
==========================================================
Collects TPU metrics from GCP (Cloud Monitoring API) or AWS (neuron-monitor)
and pushes them to the HexaGrid API every POLL_INTERVAL seconds.

Providers:
    gcp  — Google Cloud TPU v4/v5e/v5p via Cloud Monitoring API
    aws  — AWS Trainium/Inferentia via neuron-monitor daemon JSON

Environment variables:
    HEXAGRID_ENDPOINT    HexaGrid API base URL (default: http://localhost:8000)
    HEXAGRID_TOKEN       API key from bootstrap output
    TPU_PROVIDER         gcp | aws (default: auto-detect)
    CLUSTER_ID           Logical cluster name (default: hostname)
    POLL_INTERVAL_S      Seconds between polls (default: 30)

    # GCP only:
    GCP_PROJECT_ID       GCP project ID
    GCP_ZONE             Zone e.g. us-central2-b
    GCP_TPU_NODE         TPU node name

    # AWS only:
    (no extra config needed — reads from neuron-monitor on localhost:8888)

Usage:
    python tpu_collector_agent.py
"""

import json
import logging
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] hg-tpu-agent %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("hg-tpu-agent")

# ── Config ────────────────────────────────────────────────────────────────────
HEXAGRID_BASE    = os.environ.get("HEXAGRID_ENDPOINT", "http://localhost:8000").rstrip("/")
HEXAGRID_TOKEN   = os.environ.get("HEXAGRID_TOKEN", "")
TPU_PROVIDER     = os.environ.get("TPU_PROVIDER", "auto")
CLUSTER_ID       = os.environ.get("CLUSTER_ID", socket.gethostname())
POLL_INTERVAL    = int(os.environ.get("POLL_INTERVAL_S", "30"))
INGEST_URL       = f"{HEXAGRID_BASE}/api/v1/telemetry/tpu/"

# GCP
GCP_PROJECT      = os.environ.get("GCP_PROJECT_ID", "")
GCP_ZONE         = os.environ.get("GCP_ZONE", "")
GCP_TPU_NODE     = os.environ.get("GCP_TPU_NODE", "")

# AWS Neuron Monitor
NEURON_MONITOR   = os.environ.get("NEURON_MONITOR_URL", "http://localhost:8888")


# ══════════════════════════════════════════════════════════════════════════════
#  GCP BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def _gcp_get_token():
    """Get GCP access token from metadata server (works on GCE/GKE/Cloud Run)."""
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"}
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read())["access_token"]
    except Exception as e:
        log.error("GCP metadata token fetch failed: %s", e)
        return None


def _gcp_query_metric(token, project, metric_type, node_filter, minutes=2):
    """Query a single Cloud Monitoring metric and return time series."""
    end   = datetime.now(timezone.utc)
    start = end.replace(minute=end.minute - minutes) if end.minute >= minutes else end
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str   = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    filter_str = (
        f'metric.type="{metric_type}" '
        f'AND resource.labels.node_id="{node_filter}"'
    )
    url = (
        f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries"
        f"?filter={urllib.parse.quote(filter_str)}"
        f"&interval.startTime={start_str}&interval.endTime={end_str}"
        f"&view=FULL"
    )
    try:
        import urllib.parse
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read()).get("timeSeries", [])
    except Exception as e:
        log.warning("GCP metric query failed (%s): %s", metric_type, e)
        return []


def collect_gcp():
    """Collect GCP Cloud TPU metrics via Cloud Monitoring API."""
    if not GCP_PROJECT or not GCP_ZONE or not GCP_TPU_NODE:
        log.warning("GCP_PROJECT_ID, GCP_ZONE, GCP_TPU_NODE must be set for GCP provider")
        return None

    token = _gcp_get_token()
    if not token:
        return None

    # GCP TPU Cloud Monitoring metric types
    METRICS = {
        "matrix_util_pct":  "tpu.googleapis.com/container/accelerator/duty_cycle",
        "memory_util_pct":  "tpu.googleapis.com/container/accelerator/memory_used",
        "hbm_used_mib":     "tpu.googleapis.com/container/accelerator/memory_used",
        "hbm_total_mib":    "tpu.googleapis.com/container/accelerator/memory_total",
    }

    # For each chip, build a payload — GCP exposes per-chip via chip_id label
    chip_data = {}
    for field, metric in METRICS.items():
        series = _gcp_query_metric(token, GCP_PROJECT, metric, GCP_TPU_NODE)
        for ts in series:
            chip_id = ts.get("resource", {}).get("labels", {}).get("chip_id", "chip-0")
            points  = ts.get("points", [])
            if not points:
                continue
            val = points[0].get("value", {}).get("doubleValue") or \
                  points[0].get("value", {}).get("int64Value")
            if val is None:
                continue
            if chip_id not in chip_data:
                chip_data[chip_id] = {"chip_id": chip_id, "chip_type": "tpu-v5e"}
            # Convert bytes → MiB for memory metrics
            if "mib" in field and val > 1024 * 1024:
                val = val / (1024 * 1024)
            chip_data[chip_id][field] = round(float(val), 2)

    if not chip_data:
        # No live metrics — push a heartbeat with a single synthetic chip
        chip_data["chip-0"] = {"chip_id": "chip-0", "chip_type": "tpu-v5e",
                                "matrix_util_pct": None}

    return {
        "node_id":          GCP_TPU_NODE,
        "cluster_id":       CLUSTER_ID,
        "provider":         "gcp",
        "zone":             GCP_ZONE,
        "accelerator_type": os.environ.get("GCP_ACCELERATOR_TYPE", "v5e-8"),
        "chips":            list(chip_data.values()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AWS BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def collect_aws():
    """
    Collect AWS Trainium/Inferentia metrics via neuron-monitor.
    neuron-monitor must be running: neuron-monitor &
    It exposes JSON on http://localhost:8888 by default.
    """
    try:
        with urllib.request.urlopen(NEURON_MONITOR, timeout=5) as r:
            data = json.loads(r.read())
    except Exception as e:
        log.warning("neuron-monitor not reachable at %s: %s", NEURON_MONITOR, e)
        return None

    # Parse neuron-monitor JSON structure
    # See: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html
    chips = []
    neuron_data = data.get("neuron_runtime_data", [{}])[0]
    report      = neuron_data.get("report", {})

    # Per-NeuronCore utilization
    nc_util = report.get("neuroncore_counters", {}).get("neuroncores_in_use", {})
    hw_stats = report.get("memory_info", {})
    errors   = report.get("error_counters", {})

    # Build per-chip entries (Trainium1: 2 NeuronCores/chip, Inferentia2: 2/chip)
    nc_ids = list(nc_util.keys()) if nc_util else ["nc0"]
    for i, nc_id in enumerate(nc_ids):
        nc = nc_util.get(nc_id, {})
        util = nc.get("neuroncore_utilization", 0) * 100  # 0-1 → percentage

        # Memory from aggregate (neuron-monitor reports per-runtime, not per-core)
        mem_used  = hw_stats.get("neuron_runtime_used_bytes", {}).get("host", 0) / (1024*1024)
        mem_total = hw_stats.get("loaded_models", {}).get("total_size", 0) / (1024*1024)

        total_errors = sum(errors.values()) if errors else 0

        chips.append({
            "chip_id":            nc_id,
            "chip_type":          "trainium" if "trn" in socket.gethostname().lower() else "inferentia2",
            "neuroncore_util_pct": round(util, 2),
            "matrix_util_pct":    round(util, 2),    # best proxy available
            "hbm_used_mib":       round(mem_used, 1),
            "hbm_total_mib":      round(mem_total, 1) if mem_total else None,
            "runtime_errors":     total_errors,
        })

    if not chips:
        chips = [{"chip_id": "nc0", "chip_type": "trainium", "matrix_util_pct": None}]

    # Instance type from EC2 metadata
    instance_type = None
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type", 
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            instance_type = r.read().decode()
    except Exception:
        pass

    return {
        "node_id":          socket.gethostname(),
        "cluster_id":       CLUSTER_ID,
        "provider":         "aws",
        "zone":             os.environ.get("AWS_REGION", "unknown"),
        "accelerator_type": instance_type or "trn1",
        "instance_type":    instance_type,
        "chips":            chips,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-DETECT
# ══════════════════════════════════════════════════════════════════════════════

def detect_provider():
    """Auto-detect TPU provider from environment."""
    # Check for GCP metadata server
    try:
        urllib.request.urlopen(
            "http://metadata.google.internal/computeMetadata/v1/",
            timeout=1
        )
        return "gcp"
    except Exception:
        pass
    # Check for neuron-monitor
    try:
        urllib.request.urlopen(NEURON_MONITOR, timeout=1)
        return "aws"
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PUSH
# ══════════════════════════════════════════════════════════════════════════════

def push(payload: dict) -> bool:
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "X-API-Key":    HEXAGRID_TOKEN,
    }
    try:
        req = urllib.request.Request(INGEST_URL, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            result = json.loads(r.read())
            log.info("Push OK — chips=%d ts=%s", result.get("chips_recorded", 0), result.get("ts", ""))
            return True
    except urllib.error.HTTPError as e:
        log.warning("Push HTTP error: %d %s", e.code, e.reason)
    except Exception as e:
        log.warning("Push failed: %s", e)
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    provider = TPU_PROVIDER
    if provider == "auto":
        provider = detect_provider()
        if not provider:
            log.error("Could not auto-detect TPU provider. "
                      "Set TPU_PROVIDER=gcp or TPU_PROVIDER=aws explicitly.")
            sys.exit(1)
        log.info("Auto-detected provider: %s", provider)

    log.info("HexaGrid TPU collector starting — provider=%s cluster=%s interval=%ds",
             provider, CLUSTER_ID, POLL_INTERVAL)

    backoff = POLL_INTERVAL
    while True:
        try:
            payload = collect_gcp() if provider == "gcp" else collect_aws()
            if payload:
                ok = push(payload)
                backoff = POLL_INTERVAL if ok else min(backoff * 2, 300)
            else:
                log.warning("Collection returned no data — will retry")
                backoff = min(backoff * 2, 300)
        except Exception as e:
            log.error("Collector error: %s", e)
            backoff = min(backoff * 2, 300)
        time.sleep(backoff)


if __name__ == "__main__":
    main()
