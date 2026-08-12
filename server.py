import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from browser_use import Agent, ChatBrowserUse


app = FastAPI(
    title="Browser Use API",
    version="1.0.0",
)


class RunRequest(BaseModel):
    task: str
    model: str = "bu-2-0"


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "browser-use",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }


@app.post("/run")
async def run_browser_task(request: RunRequest):
    try:
        api_key = os.getenv("BROWSER_USE_API_KEY")

        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="BROWSER_USE_API_KEY is not configured",
            )

        llm = ChatBrowserUse(
            model=request.model,
        )

        agent = Agent(
            task=request.task,
            llm=llm,
        )

        history = await agent.run()

        return {
            "success": True,
            "result": history.final_result(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
