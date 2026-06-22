"""
PPVFRA Variety Browser — FastAPI backend
Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import ConnectionFailure
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
# MONGO_URI: set this env var on Render to the Atlas connection string
# (same value as ATLAS_URI in your local .env). Falls back to local Mongo
# for `uvicorn main:app --reload` during development.
MONGO_URI  = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME    = "agri_db"
COLLECTION = "varieties"
PORT       = int(os.environ.get("PORT", 8000))
HERE       = Path(__file__).parent
GEODATA    = HERE.parent.parent / "geodata"

SORTABLE_FIELDS = {
    "denomination", "crop_name", "category", "variety_type",
    "classification", "maturity", "irrigation", "filing_date", "source_pdf",
}

LIST_PROJ = {
    "dus_grouping": 0, "dus_candidate": 0, "dus_reference": 0,
    "agronomic_attributes.ipm_schedule": 0,
    "agronomic_attributes.notes": 0,
}

# ── MongoDB lifespan ──────────────────────────────────────────────────────────
_col     = None
_loc_col = None

_db_error: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _col, _loc_col, _db_error
    safe_uri = re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", MONGO_URI)
    client = None
    try:
        kwargs: dict = {"serverSelectionTimeoutMS": 8000}
        if MONGO_URI.startswith("mongodb+srv"):
            from pymongo.server_api import ServerApi
            kwargs["server_api"] = ServerApi("1")
        client   = MongoClient(MONGO_URI, **kwargs)
        client.admin.command("ping")
        _col     = client[DB_NAME][COLLECTION]
        _loc_col = client[DB_NAME]["locations"]
        print(f"MongoDB connected  →  {safe_uri}  |  {DB_NAME}.{COLLECTION}")
    except Exception as exc:
        _db_error = f"Cannot connect to MongoDB ({safe_uri}): {exc}"
        print(f"ERROR: {_db_error}")
    yield
    if client:
        client.close()
        print("MongoDB disconnected.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PPVFRA Variety Browser",
    description="REST API for exploring plant variety passport data extracted from PPVFRA journals.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Pydantic response models ──────────────────────────────────────────────────
class StatEntry(BaseModel):
    name: str | None
    count: int

class StatsResponse(BaseModel):
    total: int
    by_category: list[StatEntry]
    by_type: list[StatEntry]

class VarietyListResponse(BaseModel):
    total: int
    page: int
    limit: int
    data: list[dict[str, Any]]


# ── Query builder ─────────────────────────────────────────────────────────────
def build_query(
    search: str,
    category: list[str],
    crop_name: list[str],
    variety_type: list[str],
    classification: list[str],
    maturity: list[str],
    irrigation: list[str],
    source_pdf: list[str],
    season: list[str],
    state: list[str],
    applicant_type: list[str],
) -> dict:
    q: dict = {}

    if search.strip():
        pat = re.compile(re.escape(search.strip()), re.IGNORECASE)
        q["$or"] = [
            {"denomination":    pat},
            {"applicant":       pat},
            {"crop_name":       pat},
            {"scientific_name": pat},
        ]

    for field, vals in [
        ("category",       category),
        ("crop_name",      crop_name),
        ("variety_type",   variety_type),
        ("classification", classification),
        ("maturity",       maturity),
        ("irrigation",     irrigation),
        ("source_pdf",     source_pdf),
    ]:
        clean = [v for v in vals if v]
        if len(clean) == 1:
            q[field] = clean[0]
        elif len(clean) > 1:
            q[field] = {"$in": clean}

    season_clean = [v for v in season if v]
    if season_clean:
        q["season"] = {"$in": season_clean}

    state_clean = [v for v in state if v]
    if state_clean:
        q["states_normalized"] = {"$in": state_clean}

    at = set(v for v in applicant_type if v)
    if at == {"individual"}:
        q["applicant_is_human"] = True
    elif at == {"organization"}:
        q["applicant_is_human"] = False

    return q


# ── DB guard ──────────────────────────────────────────────────────────────────
def require_db():
    if _col is None:
        raise HTTPException(
            status_code=503,
            detail=_db_error or "Database not connected. Set the MONGO_URI environment variable.",
        )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(HERE / "index.html")


@app.get("/api/meta", summary="Filter options + total count")
def get_meta() -> dict:
    require_db()

    def distinct_strings(field: str) -> list[str]:
        return sorted(v for v in _col.distinct(field) if isinstance(v, str))

    meta: dict = {}
    for field in ("category", "crop_name", "variety_type", "classification",
                  "maturity", "irrigation", "source_pdf"):
        meta[field] = distinct_strings(field)
    meta["season"] = distinct_strings("season")
    # states_normalized is the cleaned ~32-value domain (see tools/normalize_states_mongo.py) —
    # the raw "states" field still holds whatever variant spelling/zone-label text was extracted.
    meta["states"] = distinct_strings("states_normalized")
    meta["total"]  = _col.count_documents({})
    return meta


@app.get("/api/stats", response_model=StatsResponse, summary="Dashboard stats")
def get_stats() -> StatsResponse:
    require_db()
    by_category = list(_col.aggregate([
        {"$group": {"_id": "$category",     "count": {"$sum": 1}}},
        {"$sort":  {"count": -1}},
    ]))
    by_type = list(_col.aggregate([
        {"$group": {"_id": "$variety_type", "count": {"$sum": 1}}},
        {"$sort":  {"count": -1}},
    ]))
    return StatsResponse(
        total=_col.count_documents({}),
        by_category=[StatEntry(name=d["_id"], count=d["count"]) for d in by_category if d["_id"]],
        by_type    =[StatEntry(name=d["_id"], count=d["count"]) for d in by_type     if d["_id"]],
    )


@app.get("/api/varieties", response_model=VarietyListResponse, summary="Paginated variety list")
def list_varieties(
    page:           Annotated[int,        Query(ge=1)]         = 1,
    limit:          Annotated[int,        Query(ge=1, le=1000)] = 50,
    sort:           Annotated[str,        Query()]              = "denomination",
    order:          Annotated[str,        Query()]              = "asc",
    search:         Annotated[str,        Query()]              = "",
    category:       Annotated[list[str],  Query()]              = [],
    crop_name:      Annotated[list[str],  Query()]              = [],
    variety_type:   Annotated[list[str],  Query()]              = [],
    classification: Annotated[list[str],  Query()]              = [],
    maturity:       Annotated[list[str],  Query()]              = [],
    irrigation:     Annotated[list[str],  Query()]              = [],
    source_pdf:     Annotated[list[str],  Query()]              = [],
    season:         Annotated[list[str],  Query()]              = [],
    state:          Annotated[list[str],  Query()]              = [],
    applicant_type: Annotated[list[str],  Query()]              = [],
) -> VarietyListResponse:
    require_db()
    sort_field = sort if sort in SORTABLE_FIELDS else "denomination"
    direction  = ASCENDING if order == "asc" else DESCENDING

    q     = build_query(search, category, crop_name, variety_type, classification,
                        maturity, irrigation, source_pdf, season, state, applicant_type)
    total = _col.count_documents(q)
    docs  = list(
        _col.find(q, LIST_PROJ)
            .sort(sort_field, direction)
            .skip((page - 1) * limit)
            .limit(limit)
    )
    for d in docs:
        d["_id"] = str(d["_id"])

    return VarietyListResponse(total=total, page=page, limit=limit, data=docs)


@app.get("/api/variety/{variety_id}", summary="Full variety detail")
def get_variety(variety_id: str) -> dict:
    require_db()
    doc = _col.find_one({"_id": variety_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Variety '{variety_id}' not found")
    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/api/geodata/states", summary="GADM L1 state boundaries GeoJSON for India map overlay")
def get_states_geojson():
    p = GEODATA / "gadm41_IND_1.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="gadm41_IND_1.json not found in geodata/")
    return FileResponse(p, media_type="application/json")


@app.get("/api/locations", summary="Geo points for map view (respects active filters)")
def get_locations(
    search:         Annotated[str,        Query()] = "",
    category:       Annotated[list[str],  Query()] = [],
    crop_name:      Annotated[list[str],  Query()] = [],
    variety_type:   Annotated[list[str],  Query()] = [],
    classification: Annotated[list[str],  Query()] = [],
    maturity:       Annotated[list[str],  Query()] = [],
    irrigation:     Annotated[list[str],  Query()] = [],
    source_pdf:     Annotated[list[str],  Query()] = [],
    season:         Annotated[list[str],  Query()] = [],
    state:          Annotated[list[str],  Query()] = [],
    applicant_type: Annotated[list[str],  Query()] = [],
) -> list[dict]:
    """
    Returns geo data for the filtered variety set.
    Joins varieties → locations. Used by the map tab.
    Capped at 1000 records for map performance.
    """
    require_db()
    q = build_query(search, category, crop_name, variety_type, classification,
                    maturity, irrigation, source_pdf, season, state, applicant_type)

    variety_proj = {"_id": 1, "denomination": 1, "crop_name": 1,
                    "category": 1, "applicant": 1, "applicant_is_human": 1}
    varieties = list(_col.find(q, variety_proj).limit(1000))
    if not varieties:
        return []

    ids = [v["_id"] for v in varieties]
    locs = {l["_id"]: l for l in _loc_col.find({"_id": {"$in": ids}})}
    var_map = {v["_id"]: v for v in varieties}

    result = []
    for vid in ids:
        loc = locs.get(vid)
        if not loc:
            continue
        var = var_map[vid]
        result.append({
            "_id":              vid,
            "denomination":     var.get("denomination"),
            "crop_name":        var.get("crop_name"),
            "category":         var.get("category"),
            "applicant":        var.get("applicant"),
            "applicant_is_human": var.get("applicant_is_human"),
            "applicant_geo":       loc.get("applicant_geo"),
            "suitability_states":  loc.get("suitability_states", []),
            "suitability_districts": loc.get("suitability_districts", []),
        })
    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # On Render, the dashboard start command (uvicorn main:app --host 0.0.0.0
    # --port $PORT) is used instead, bypassing this block. This is for local
    # dev — set HOST=0.0.0.0 if testing from another device on your network.
    host = os.environ.get("HOST", "localhost")
    reload = host == "localhost"
    uvicorn.run("main:app", host=host, port=PORT, reload=reload)
