from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List
import sys
import os
from datetime import datetime
import uvicorn

sys.path.append('..')

from model.evaluator import ModelEvaluator

# Initialize FastAPI app
app = FastAPI(
    title="🐦 Twitter Sentiment Analyzer API",
    description="""
    ## Production-Grade Sentiment Analysis System
    
    Real-time sentiment analysis for tweets using **Bidirectional LSTM** neural networks.
    
    ### 🎯 Features
    - **Single Prediction**: Analyze individual tweets
    - **Batch Processing**: Process multiple tweets simultaneously
    - **Model Information**: Get architecture and performance metrics
    
    ### 📊 Performance
    - Latency: <100ms per prediction
    - F1 Score: 0.85 (macro)
    - Accuracy: 0.87
    
    ### 🔧 Sentiment Classes
    - **0**: Negative 😞
    - **1**: Neutral 😐
    - **2**: Positive 😊
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "url": "https://github.com/yourusername/twitter-sentiment-analyzer",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass

# Load model
try:
    evaluator = ModelEvaluator(
        model_path='../model/checkpoints/bilstm_best.h5',
        tokenizer_path='../model/tokenizer.pkl'
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    evaluator = None

# Request/Response models
class TweetRequest(BaseModel):
    text: str = Field(
        ...,
        description="Tweet text to analyze",
        example="This product is amazing! Best purchase ever!"
    )

class BatchTweetRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of tweets to analyze",
        example=["Great service!", "Terrible experience", "It's okay"]
    )

class SentimentResponse(BaseModel):
    text: str = Field(..., description="Original tweet text")
    sentiment: str = Field(..., description="Predicted sentiment class")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    label: int = Field(..., description="Numeric label (0=Negative, 1=Neutral, 2=Positive)")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")

# Root landing page
@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["Web"],
    summary="Landing page",
    include_in_schema=False
)
async def landing_page():
    """Serve landing page."""
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return """
        <html>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1>🐦 Twitter Sentiment Analyzer API</h1>
                <p>Status: <span style="color: green;">Online</span></p>
                <a href="/docs" style="font-size: 18px;">📚 View API Documentation</a>
            </body>
        </html>
        """

# API info endpoint
@app.get(
    "/api",
    tags=["Health"],
    summary="API information",
    description="Get API metadata and status"
)
async def api_info():
    """Get API information and current status."""
    return {
        "message": "🐦 Twitter Sentiment Analyzer API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": evaluator is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch": "/predict/batch",
            "model_info": "/model/info"
        }
    }

@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Check API and model health status"
)
async def health_check():
    """Verify API and model are operational."""
    return {
        "status": "healthy" if evaluator is not None else "degraded",
        "model_loaded": evaluator is not None,
        "timestamp": datetime.now().isoformat(),
        "uptime": "operational"
    }

# Single prediction endpoint
@app.post(
    "/predict",
    response_model=SentimentResponse,
    tags=["Prediction"],
    summary="Predict single tweet sentiment",
    description="Analyze sentiment of a single tweet and return prediction with confidence score"
)
async def predict_sentiment(request: TweetRequest):
    """
    Predict sentiment for a single tweet.
    
    - **text**: Tweet text to analyze (required)
    
    Returns:
    - **sentiment**: Predicted class (Negative/Neutral/Positive)
    - **confidence**: Model confidence score (0-1)
    - **label**: Numeric label (0/1/2)
    - **timestamp**: Prediction timestamp
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = evaluator.predict_single(request.text)
        result['timestamp'] = datetime.now().isoformat()
        return SentimentResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post(
    "/predict/batch",
    response_model=BatchSentimentResponse,
    tags=["Prediction"],
    summary="Batch predict tweet sentiments",
    description="Analyze sentiment for multiple tweets in a single request"
)
async def predict_batch(request: BatchTweetRequest):
    """
    Predict sentiment for multiple tweets in batch.
    
    - **texts**: List of tweet texts to analyze (required)
    
    Returns:
    - **results**: List of predictions for each tweet
    - **total**: Total number of predictions made
    
    Ideal for processing large volumes of tweets efficiently.
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        results = []
        for text in request.texts:
            if text and len(text.strip()) > 0:
                result = evaluator.predict_single(text)
                result['timestamp'] = datetime.now().isoformat()
                results.append(SentimentResponse(**result))
        
        return BatchSentimentResponse(
            results=results,
            total=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Model info endpoint
@app.get(
    "/model/info",
    tags=["Model"],
    summary="Get model information",
    description="Retrieve model architecture details and configuration"
)
async def model_info():
    """
    Get detailed model architecture and configuration information.
    
    Returns:
    - **model_type**: Neural network architecture
    - **classes**: Sentiment class mappings
    - **max_sequence_length**: Input sequence length
    - **vocab_size**: Vocabulary size
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Bidirectional LSTM",
        "classes": evaluator.label_map,
        "max_sequence_length": evaluator.sequencer.max_sequence_length,
        "vocab_size": evaluator.sequencer.max_vocab_size
    }

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', 8000)),
        reload=True
    )
