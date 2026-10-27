from fastapi import FastAPI

app = FastAPI(
    title="InferBench AutoTune API",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "InferBench AutoTune API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inferbench.api.main:app", host="127.0.0.1", port=8000, reload=True)
