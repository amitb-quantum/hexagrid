"""
Energia Phase 9 — Multi-Site Orchestrator
==========================================
Fleet-level workload routing across multiple virtual data center sites.
Each site is associated with a real ISO grid region, giving it live
price and carbon data from Phases 6 & 8.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                  SiteOrchestrator                   │
  │                                                     │
  │  Site A (CAISO)   Site B (ERCOT)   Site C (NYISO)  │
  │  Virginia DC      Texas DC         New York DC      │
  │                                                     │
  │  Scores each site on: price, carbon, capacity,      │
  │  PUE, SLA urgency — then routes the workload.       │
  └─────────────────────────────────────────────────────┘

Features:
  - Configurable fleet of named sites (persist in SQLite)
  - Real-time composite scoring: price + carbon + capacity
  - Workload routing decisions with full reasoning
  - Routing history log (last 200 decisions)
  - Fleet health summary
  - Cost and carbon savings vs single-site baseline

Usage:
  from site_orchestrator import SiteOrchestrator
  orch = SiteOrchestrator()
  decision = orch.route_workload(job_type='llm_training', duration_min=60, gpu_count=8)
"""

import os, json, sqlite3, time, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import requests
from dotenv import load_dotenv

load_dotenv()

LOG_DIR  = Path(__file__).parent.parent / "logs" / "fleet"
DB_PATH  = Path(__file__).parent.parent / "cache" / "fleet.db"
LOCAL_API = os.getenv("ENERGIA_API_BASE", "http://localhost:8000/api/v1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("site_orchestrator")

# ── Default fleet configuration ───────────────────────────────────────────────
# Each site maps to a real ISO region for live price/carbon data
DEFAULT_SITES = [
    {
        "site_id":    "site-ca",
        "name":       "California DC",
        "location":   "San Jose, CA",
        "iso":        "CAISO",
        "num_racks":  8,
        "capacity_pct": 65.0,   # current utilization %
        "pue":        1.35,
        "tier":       3,
        "online":     True,
    },
    {
        "site_id":    "site-tx",
        "name":       "Texas DC",
        "location":   "Austin, TX",
        "iso":        "ERCOT",
        "num_racks":  12,
        "capacity_pct": 50.0,
        "pue":        1.42,
        "tier":       3,
        "online":     True,
    },
    {
        "site_id":    "site-ny",
        "name":       "New York DC",
        "location":   "Newark, NJ",
        "iso":        "NYISO",
        "num_racks":  6,
        "capacity_pct": 80.0,
        "pue":        1.28,
        "tier":       4,
        "online":     True,
    },
    {
        "site_id":    "site-ne",
        "name":       "New England DC",
        "location":   "Boston, MA",
        "iso":        "ISONE",
        "num_racks":  4,
        "capacity_pct": 40.0,
        "pue":        1.31,
        "tier":       3,
        "online":     True,
    },
    {
        "site_id":    "site-va",
        "name":       "Virginia DC",
        "location":   "Ashburn, VA",
        "iso":        "PJM",
        "num_racks":  16,
        "capacity_pct": 55.0,
        "pue":        1.38,
        "tier":       4,
        "online":     True,
    },
]

# Job type profiles — GPU count and typical duration
JOB_PROFILES = {
    "llm_training":      {"gpu_min": 8,  "label": "LLM Training",         "sla_hours": 24},
    "llm_inference":     {"gpu_min": 2,  "label": "LLM Inference",        "sla_hours": 1},
    "diffusion":         {"gpu_min": 4,  "label": "Diffusion Model",      "sla_hours": 8},
    "rl_training":       {"gpu_min": 4,  "label": "RL Training",          "sla_hours": 12},
    "embedding":         {"gpu_min": 1,  "label": "Embedding Batch",      "sla_hours": 4},
    "batch_inference":   {"gpu_min": 4,  "label": "Batch Inference",      "sla_hours": 6},
    "data_processing":   {"gpu_min": 2,  "label": "Data Processing",      "sla_hours": 8},
    "custom":            {"gpu_min": 1,  "label": "Custom Workload",      "sla_hours": 12},
}


@dataclass
class SiteScore:
    site_id:          str
    name:             str
    iso:              str
    location:         str
    price_usd_kwh:    float
    carbon_gco2:      float
    carbon_label:     str
    carbon_color:     str
    capacity_avail:   float      # % headroom available
    pue:              float
    price_score:      float      # 0–100, lower price = higher score
    carbon_score:     float      # 0–100, lower carbon = higher score
    capacity_score:   float      # 0–100, more headroom = higher score
    pue_score:        float      # 0–100, lower PUE = higher score
    composite_score:  float      # weighted total, higher = better
    rank:             int
    recommended:      bool
    reasoning:        str
    estimated_cost_usd: float    # cost for this job at this site
    estimated_carbon_g: float    # gCO2 for this job at this site


class SiteOrchestrator:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._init_db()

    # ── Database ──────────────────────────────────────────────────────────────
    def _init_db(self):
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sites (
                site_id      TEXT PRIMARY KEY,
                config       TEXT NOT NULL,
                updated_at   REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS routing_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    REAL NOT NULL,
                job_type     TEXT,
                gpu_count    INTEGER,
                duration_min INTEGER,
                chosen_site  TEXT,
                score        REAL,
                reasoning    TEXT,
                all_scores   TEXT
            );
        """)
        self._db.commit()
        self._ensure_default_sites()

    def _ensure_default_sites(self):
        for site in DEFAULT_SITES:
            existing = self._db.execute(
                "SELECT site_id FROM sites WHERE site_id=?", (site["site_id"],)
            ).fetchone()
            if not existing:
                self._db.execute(
                    "INSERT INTO sites (site_id, config, updated_at) VALUES (?,?,?)",
                    (site["site_id"], json.dumps(site), time.time())
                )
        self._db.commit()

    def get_sites(self) -> list[dict]:
        rows = self._db.execute("SELECT config FROM sites ORDER BY site_id").fetchall()
        return [json.loads(r[0]) for r in rows]

    def update_site(self, site_id: str, updates: dict) -> bool:
        row = self._db.execute(
            "SELECT config FROM sites WHERE site_id=?", (site_id,)
        ).fetchone()
        if not row:
            return False
        config = json.loads(row[0])
        config.update(updates)
        self._db.execute(
            "UPDATE sites SET config=?, updated_at=? WHERE site_id=?",
            (json.dumps(config), time.time(), site_id)
        )
        self._db.commit()
        return True

    def _log_decision(self, job_type, gpu_count, duration_min, chosen: SiteScore, all_scores: list):
        self._db.execute(
            """INSERT INTO routing_log
               (timestamp, job_type, gpu_count, duration_min, chosen_site, score, reasoning, all_scores)
               VALUES (?,?,?,?,?,?,?,?)""",
            (time.time(), job_type, gpu_count, duration_min,
             chosen.site_id, chosen.composite_score,
             chosen.reasoning,
             json.dumps([asdict(s) for s in all_scores]))
        )
        self._db.commit()

    def get_routing_history(self, limit: int = 50) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM routing_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        cols = ["id","timestamp","job_type","gpu_count","duration_min",
                "chosen_site","score","reasoning","all_scores"]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["all_scores"] = json.loads(d["all_scores"])
            d["datetime"] = datetime.fromtimestamp(d["timestamp"], tz=timezone.utc).isoformat()
            result.append(d)
        return result

    # ── Live data fetching ────────────────────────────────────────────────────
    def _fetch_carbon_snapshot(self) -> dict:
        try:
            r = requests.get(f"{LOCAL_API}/carbon/snapshot", timeout=8)
            if r.ok:
                return r.json().get("zones", {})
        except Exception as e:
            logger.warning(f"Carbon snapshot fetch failed: {e}")
        return {}

    def _fetch_price(self) -> float:
        """Fetch live CAISO price from pricefeed."""
        try:
            r = requests.get(f"{LOCAL_API}/pricefeed?horizon_min=5", timeout=8)
            if r.ok:
                return r.json().get("current_price_usd_kwh", 0.055)
        except Exception as e:
            logger.warning(f"Pricefeed fetch failed: {e}")
        return 0.055

    # Regional price fallbacks (EIA 2024 averages)
    REGIONAL_PRICES = {
        "CAISO": 0.0620,
        "ERCOT": 0.0410,
        "NYISO": 0.0580,
        "ISONE": 0.0550,
        "PJM":   0.0480,
    }

    # ── Core routing logic ────────────────────────────────────────────────────
    def route_workload(
        self,
        job_type:     str   = "custom",
        duration_min: int   = 60,
        gpu_count:    int   = 4,
        cost_weight:  float = 0.40,
        carbon_weight:float = 0.30,
        # capacity and PUE share the remaining 0.30
        priority:     str   = "balanced",   # balanced | cost | carbon | capacity
    ) -> dict:
        """
        Score all online sites and return ranked list + recommended site.

        Scoring weights (adjustable via priority):
          balanced:  cost=0.40, carbon=0.30, capacity=0.20, pue=0.10
          cost:      cost=0.70, carbon=0.10, capacity=0.15, pue=0.05
          carbon:    cost=0.10, carbon=0.70, capacity=0.15, pue=0.05
          capacity:  cost=0.20, carbon=0.20, capacity=0.50, pue=0.10
        """
        WEIGHTS = {
            "balanced": {"cost": 0.40, "carbon": 0.30, "capacity": 0.20, "pue": 0.10},
            "cost":     {"cost": 0.70, "carbon": 0.10, "capacity": 0.15, "pue": 0.05},
            "carbon":   {"cost": 0.10, "carbon": 0.70, "capacity": 0.15, "pue": 0.05},
            "capacity": {"cost": 0.20, "carbon": 0.20, "capacity": 0.50, "pue": 0.10},
        }
        w = WEIGHTS.get(priority, WEIGHTS["balanced"])

        sites        = [s for s in self.get_sites() if s.get("online", True)]
        carbon_data  = self._fetch_carbon_snapshot()
        live_price   = self._fetch_price()

        # Build raw metrics for each site
        raw = []
        for site in sites:
            iso       = site["iso"]
            cdata     = carbon_data.get(iso, {})
            price     = live_price if iso == "CAISO" else self.REGIONAL_PRICES.get(iso, 0.055)
            carbon    = cdata.get("carbon_intensity", self._fallback_carbon(iso))
            capacity_avail = 100.0 - site.get("capacity_pct", 50.0)
            pue       = site.get("pue", 1.40)

            # Estimated job cost: power_kw × duration_hours × price × PUE
            gpu_kw    = gpu_count * 0.7          # ~700W per GPU at load
            hours     = duration_min / 60.0
            est_cost  = gpu_kw * hours * price * pue
            est_carbon = gpu_kw * hours * carbon  # gCO2

            raw.append({
                "site":          site,
                "price":         price,
                "carbon":        carbon,
                "carbon_label":  cdata.get("carbon_label", self._carbon_label(carbon)),
                "carbon_color":  cdata.get("carbon_color", "#ffaa00"),
                "capacity_avail":capacity_avail,
                "pue":           pue,
                "est_cost":      est_cost,
                "est_carbon":    est_carbon,
            })

        if not raw:
            return {"status": "error", "message": "No online sites available"}

        # Normalize each metric to 0–1 (0 = worst, 1 = best)
        def norm_low(vals, v):   # lower raw = better score
            mn, mx = min(vals), max(vals)
            return 1.0 - ((v - mn) / (mx - mn)) if mx > mn else 1.0
        def norm_high(vals, v):  # higher raw = better score
            mn, mx = min(vals), max(vals)
            return (v - mn) / (mx - mn) if mx > mn else 1.0

        prices    = [r["price"]          for r in raw]
        carbons   = [r["carbon"]         for r in raw]
        caps      = [r["capacity_avail"] for r in raw]
        pues      = [r["pue"]            for r in raw]

        scores: list[SiteScore] = []
        for r in raw:
            ps = norm_low(prices,  r["price"])   * 100
            cs = norm_low(carbons, r["carbon"])  * 100
            as_ = norm_high(caps,  r["capacity_avail"]) * 100
            us  = norm_low(pues,   r["pue"])     * 100
            composite = (w["cost"] * ps + w["carbon"] * cs +
                         w["capacity"] * as_ + w["pue"] * us)

            site = r["site"]
            scores.append(SiteScore(
                site_id         = site["site_id"],
                name            = site["name"],
                iso             = site["iso"],
                location        = site["location"],
                price_usd_kwh   = round(r["price"], 5),
                carbon_gco2     = round(r["carbon"], 1),
                carbon_label    = r["carbon_label"],
                carbon_color    = r["carbon_color"],
                capacity_avail  = round(r["capacity_avail"], 1),
                pue             = r["pue"],
                price_score     = round(ps, 1),
                carbon_score    = round(cs, 1),
                capacity_score  = round(as_, 1),
                pue_score       = round(us, 1),
                composite_score = round(composite, 2),
                rank            = 0,
                recommended     = False,
                reasoning       = "",
                estimated_cost_usd = round(r["est_cost"], 4),
                estimated_carbon_g = round(r["est_carbon"], 1),
            ))

        # Rank and annotate
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        for i, s in enumerate(scores):
            s.rank = i + 1
        scores[0].recommended = True
        scores[0].reasoning   = self._build_reasoning(scores[0], scores, w, priority, job_type, gpu_count, duration_min)

        # Savings vs worst site
        best  = scores[0]
        worst = scores[-1]
        cost_saving_pct   = round((worst.estimated_cost_usd - best.estimated_cost_usd) / max(worst.estimated_cost_usd, 0.0001) * 100, 1)
        carbon_saving_pct = round((worst.estimated_carbon_g  - best.estimated_carbon_g)  / max(worst.estimated_carbon_g,  0.0001) * 100, 1)

        self._log_decision(job_type, gpu_count, duration_min, best, scores)

        return {
            "status":            "ok",
            "recommended_site":  asdict(best),
            "all_sites":         [asdict(s) for s in scores],
            "job_type":          job_type,
            "job_label":         JOB_PROFILES.get(job_type, {}).get("label", job_type),
            "gpu_count":         gpu_count,
            "duration_min":      duration_min,
            "priority":          priority,
            "weights":           w,
            "cost_saving_pct":   cost_saving_pct,
            "carbon_saving_pct": carbon_saving_pct,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }

    def fleet_summary(self) -> dict:
        """High-level fleet health and capacity summary."""
        sites        = self.get_sites()
        carbon_data  = self._fetch_carbon_snapshot()
        live_price   = self._fetch_price()

        summary = []
        total_racks = total_used = 0
        for site in sites:
            iso    = site["iso"]
            cdata  = carbon_data.get(iso, {})
            price  = live_price if iso == "CAISO" else self.REGIONAL_PRICES.get(iso, 0.055)
            carbon = cdata.get("carbon_intensity", self._fallback_carbon(iso))
            racks  = site.get("num_racks", 4)
            used   = round(racks * site.get("capacity_pct", 50) / 100, 1)
            total_racks += racks
            total_used  += used
            summary.append({
                "site_id":          site["site_id"],
                "name":             site["name"],
                "location":         site["location"],
                "iso":              site["iso"],
                "online":           site.get("online", True),
                "num_racks":        racks,
                "capacity_pct":     site.get("capacity_pct", 50),
                "racks_used":       used,
                "pue":              site.get("pue", 1.4),
                "tier":             site.get("tier", 3),
                "price_usd_kwh":    round(price, 5),
                "carbon_gco2":      round(carbon, 1),
                "carbon_label":     cdata.get("carbon_label", self._carbon_label(carbon)),
                "carbon_color":     cdata.get("carbon_color", "#ffaa00"),
                "fossil_free_pct":  cdata.get("fossil_free_pct"),
            })

        fleet_capacity_pct = round(total_used / total_racks * 100, 1) if total_racks else 0
        online_count = sum(1 for s in sites if s.get("online", True))

        return {
            "sites":               summary,
            "total_sites":         len(sites),
            "online_sites":        online_count,
            "total_racks":         total_racks,
            "fleet_capacity_pct":  fleet_capacity_pct,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _fallback_carbon(self, iso: str) -> float:
        FALLBACKS = {"CAISO":210,"ERCOT":380,"NYISO":170,"ISONE":220,"PJM":310}
        return FALLBACKS.get(iso, 300)

    def _carbon_label(self, gco2: float) -> str:
        if gco2 < 50:   return "Very Clean"
        if gco2 < 150:  return "Clean"
        if gco2 < 300:  return "Moderate"
        if gco2 < 450:  return "Dirty"
        return "Very Dirty"

    def _build_reasoning(self, best: SiteScore, all_scores: list, w: dict,
                         priority: str, job_type: str, gpu_count: int, duration_min: int) -> str:
        profile   = JOB_PROFILES.get(job_type, {})
        job_label = profile.get("label", job_type)
        second    = all_scores[1] if len(all_scores) > 1 else None

        parts = [
            f"Routing {gpu_count}-GPU {job_label} ({duration_min}min) using {priority} priority.",
            f"{best.name} ({best.iso}) scores highest at {best.composite_score:.1f}/100.",
        ]
        if best.price_score >= 80:
            parts.append(f"Cheapest grid at ${best.price_usd_kwh:.4f}/kWh.")
        if best.carbon_score >= 80:
            parts.append(f"Cleanest grid at {best.carbon_gco2:.0f} gCO₂/kWh ({best.carbon_label}).")
        if best.capacity_avail >= 60:
            parts.append(f"Ample capacity ({best.capacity_avail:.0f}% headroom).")
        if second:
            margin = best.composite_score - second.composite_score
            parts.append(f"Leads {second.name} by {margin:.1f} points.")
        parts.append(
            f"Estimated job cost: ${best.estimated_cost_usd:.3f}  |  "
            f"Carbon: {best.estimated_carbon_g:.0f}g CO₂."
        )
        return " ".join(parts)


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    orch = SiteOrchestrator()

    print("\n=== Fleet Summary ===")
    fs = orch.fleet_summary()
    for s in fs["sites"]:
        print(f"  {s['name']:20s} {s['iso']:6s}  "
              f"${s['price_usd_kwh']:.4f}/kWh  "
              f"{s['carbon_gco2']:5.0f} gCO₂  "
              f"{s['capacity_pct']:4.0f}% load  "
              f"PUE {s['pue']}")

    print("\n=== Route: LLM Training, 8 GPUs, 4 hours, balanced ===")
    d = orch.route_workload("llm_training", duration_min=240, gpu_count=8, priority="balanced")
    r = d["recommended_site"]
    print(f"  → {r['name']} ({r['iso']})  score={r['composite_score']}")
    print(f"  → Est cost: ${r['estimated_cost_usd']:.3f}  Carbon: {r['estimated_carbon_g']:.0f}g")
    print(f"  → {r['reasoning']}")
    print(f"\n  Cost saving vs worst site:   {d['cost_saving_pct']}%")
    print(f"  Carbon saving vs worst site: {d['carbon_saving_pct']}%")

    print("\n=== Route: Same job, carbon priority ===")
    d2 = orch.route_workload("llm_training", duration_min=240, gpu_count=8, priority="carbon")
    r2 = d2["recommended_site"]
    print(f"  → {r2['name']} ({r2['iso']})  score={r2['composite_score']}")
