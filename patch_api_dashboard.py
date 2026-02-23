"""
Patches api.py to serve the dashboard HTML at GET /
Run from the energia project root:
    python patch_api_dashboard.py
"""
import os, sys

api_path = os.path.join(os.path.dirname(__file__), 'api', 'api.py')
if not os.path.exists(api_path):
    # Try running from project root directly
    api_path = 'api/api.py'
if not os.path.exists(api_path):
    print(f"ERROR: Cannot find api/api.py — run this from the energia project root")
    sys.exit(1)

content = open(api_path).read()

# ── Patch 1: Add StaticFiles + HTMLResponse to imports ────────────────────────
old_import = "from fastapi.responses import JSONResponse, FileResponse"
new_import = "from fastapi.responses import JSONResponse, FileResponse, HTMLResponse\nfrom fastapi.staticfiles import StaticFiles"

if "StaticFiles" in content:
    print("  [SKIP] StaticFiles already imported")
else:
    assert old_import in content, f"Import line not found:\n{old_import}"
    content = content.replace(old_import, new_import)
    print("  [OK]   Added StaticFiles + HTMLResponse imports")

# ── Patch 2: Mount dashboard static files after CORS middleware ────────────────
old_cors = """app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)"""

new_cors = """app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Dashboard: serve index.html at GET / ──────────────────────────────────────
_DASHBOARD = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'index.html')

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    \"\"\"Serve the Energia dashboard.\"\"\"
    if not os.path.exists(_DASHBOARD):
        return HTMLResponse(
            content="<h2>Dashboard not found</h2>"
                    "<p>Place <code>index.html</code> in <code>energia/dashboard/</code></p>"
                    "<p><a href='/docs'>API docs →</a></p>",
            status_code=404,
        )
    return HTMLResponse(content=open(_DASHBOARD).read())"""

if "serve_dashboard" in content:
    print("  [SKIP] Dashboard route already exists")
else:
    assert old_cors in content, "CORS middleware block not found"
    content = content.replace(old_cors, new_cors)
    print("  [OK]   Added GET / dashboard route")

open(api_path, 'w').write(content)
print(f"\n  Patched: {api_path}")
print("\n  Next steps:")
print("    mkdir -p dashboard")
print("    cp /path/to/index.html dashboard/index.html")
print("    uvicorn api.api:app --reload --host 0.0.0.0 --port 8000")
print("    → Open: http://localhost:8000/\n")
