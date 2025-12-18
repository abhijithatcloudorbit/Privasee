from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import upload, process, status, result

app = FastAPI(title="Privasee Backend MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(process.router)
app.include_router(status.router)
app.include_router(result.router)
