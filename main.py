from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Models ----------

class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

# ---------- Endpoint ----------

@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(data: CommentRequest):
    try:
        response = client.responses.parse(
            model="gpt-4.1-mini",
            input=data.comment,
            response_format=SentimentResponse
        )
        return response.output_parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
