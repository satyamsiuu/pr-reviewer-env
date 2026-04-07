import uvicorn
from openenv.core.env_server import create_fastapi_app
from .environment import PrReviewerEnvEnvironment
from models import PRReviewAction, PRReviewObservation

# Wrap it in the OpenEnv FastAPI server (passing the CLASS, not an instance!)
app = create_fastapi_app(PrReviewerEnvEnvironment, PRReviewAction, PRReviewObservation)

def main():
    """Entry point for the validator and local execution."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()