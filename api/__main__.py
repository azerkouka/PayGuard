"""Start the scoring API: `python -m api`."""

import uvicorn

from config import settings

if __name__ == "__main__":
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port)
