from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from starlette.responses import FileResponse
from services.ai import generate_review
app = FastAPI()


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


@app.get("/api/gpt")
async def gpt(review: str):
    return generate_review(review)
