from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    white_engine: str
    black_engine: str
    depth: int = 3
    max_plies: int = 120


@app.post("/api/match/start")
def start_match(req: MatchRequest):
    result = subprocess.run(
        [
            "/Users/cahree/dev/systems/chess/engines/neuro-knight/build/run_web_match",
            req.white_engine,
            req.black_engine,
            str(req.depth),
            str(req.max_plies),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {
            "error": "engine failed",
            "stderr": result.stderr,
            "stdout": result.stdout,
        }

    return json.loads(result.stdout)
