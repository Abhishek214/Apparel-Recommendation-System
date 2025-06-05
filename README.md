import os
import sys
import platform
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

# Added pvt by Vipin
from pvt import PVT

# Added by Vipin for Health
uname = platform.uname()

def get_uptime_millis():
    with open('/proc/uptime', 'r') as f:
        uptime_seconds = float(f.readline().split()[0])
    return int(uptime_seconds * 1000)

uptime_millis = get_uptime_millis()
version = os.getenv("APP_VERSION")

# Added by Vipin for Health
app = FastAPI(root_path='/dcrest/v1/auth')

@app.get("/health", status_code=200, description="Status report of api and system")
def health():
    return {
        "healthy": "true",
        "eimId": "9929948",
        "server": uname.node,
        "componentName": "dcrest-auth",
        "version": version,
        "description": "Auth service for DCREST",
        "sourceCodeRepoUrl": "https://alm-github.systems.uk.hsbc/GBM-COT-ECO-ML/DCREST-AUTH.git",
        "documentationUrl": "NA",
        "apiSpecificationUrl": "https://dcrest-internal-sit.uk.hsbc/dcrest/v1/auth/openapi.json",
        "businessImpact": "This service handles authentication",
        "runtime": {
            "name": "PYTHON",
            "version": sys.version
        },
        "uptimeInMillis": uptime_millis
    }

@app.get("/ready", status_code=200, description="Status report of api and system")
def ready():
    return {
        "server": uname.node,
        "service_name": "Auth API",
        "status": "alive"
    }

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request):
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="auth",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    # Added PVT function by vipin before starting the application
    if PVT():
        uvicorn.run(
            "application:app",
            host="0.0.0.0",
            port=8000,
            log_level="debug",
            workers=1,
            reload=True
        )
    else:
        print("PVT FAILED SO NOT STARTING The Application")
