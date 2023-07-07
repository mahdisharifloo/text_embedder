import os
import sys
import uvicorn
from typing import List 
from starlette.responses import FileResponse
from starlette.responses import StreamingResponse
from typing import Any, Union,List 
from pydantic import BaseModel
from fastapi import FastAPI , Query

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import etl.transform as transform
from utils.logger import Logger

log = Logger("embedder")



app = FastAPI()


class Item(BaseModel):
    corpus: List[str] = []
    embeddings: List[List[float]] = None


@app.post("/embedder")
async def embedder(prompt) -> Any:
    res = transform.run(embedder,[prompt])
    return res

@app.post("/embedder_list_input")
async def embedder_list_input(corpus:List[str] = Query(None)) -> Any:
    res = transform.run(corpus)
    return Item(
        corpus=corpus,
        embeddings=res.tolist()
    )


@app.get("/")
async def read_index():
    return FileResponse(f'{root_path}/api/index.html')


if __name__ == "__main__":
    transform.EmbedderModel()
    uvicorn.run(app, host="0.0.0.0", port=20001)
