import platform
import os
import sys
import uvicorn
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# Import your auth and PVT modules
from source.auth import auth
from pvt import PVT

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
        "elmId": "9929948",
        "server": uname.node,
        "componentName": "dcrest-auth",
        "version": version,
        "description": "Description of this service",
        "sourceCodeRepoUrl": "https://alm-github.systems.uk.hsbc/GBM-COT-ECO-ML/DCREST-AUTH.git",
        "documentationUrl": "NA",
        "apiSpecificationUrl": "https://dcrest-internal-sit.uk.hsbc/dcrest/v1/auth/openapi.json",
        "businessImpact": "This service has business impact",
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

@app.post("/validate-token", status_code=200)
async def validate_token(
    DCREST_JWT_TOKEN: str = Header(),
    X_HSBC_Request_Correlation_Id: str = Header()
):
    """
    Validate JWT token and return user information.
    
    Args:
        DCREST_JWT_TOKEN: JWT token for authentication
        X_HSBC_Request_Correlation_Id: Request correlation ID for tracking
    
    Returns:
        User information if token is valid
    """
    try:
        # Validate the token using auth function
        verify = auth(DCREST_JWT_TOKEN)
        
        if verify is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        response = {
            "valid": True,
            "user_info": verify,
            "correlation_id": X_HSBC_Request_Correlation_Id
        }
        
        return jsonable_encoder(response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token validation failed"
        )

@app.post("/authenticate", status_code=200)
async def authenticate_user(
    DCREST_JWT_TOKEN: str = Header(),
    X_HSBC_Request_Correlation_Id: str = Header()
):
    """
    Authenticate user and return authentication status.
    """
    try:
        verify = auth(DCREST_JWT_TOKEN)
        
        if verify:
            return jsonable_encoder({
                "authenticated": True,
                "user": verify.get("user", "unknown"),
                "correlation_id": X_HSBC_Request_Correlation_Id
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

# Include router for auth functionality
app.include_router(AuthServer)

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
            host=host,
            port=port,
            log_level="debug",
            workers=1,
            reload=True
        )
        #ssl_keyfile=ssl_keyfile,
        #ssl_certfile = ssl_certfile
    else:
        print("PVT FAILED SO NOT STARTING THE APPLICATION")
