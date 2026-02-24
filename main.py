from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(data: CommentRequest):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Analyze sentiment and respond using JSON only."},
                {"role": "user", "content": data.comment}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"]
                    }
                }
            }
        )

        return response.output_parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
