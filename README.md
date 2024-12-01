from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware
import json
import random

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
with open("advanced_fashion_dataset_corrected.json", "r") as f:
    fashion_data = json.load(f)

# Pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Generate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


# Precompute embeddings for all outfits
outfit_embeddings = {
    outfit["outfit_id"]: get_embeddings(outfit["description"])
    for outfit in fashion_data
}


# User query model
class Query(BaseModel):
    query: str
    color: str = None
    style: str = None
    occasion: str = None


@app.post("/recommend/")
async def recommend_outfits(query: Query):
    """
    Recommend outfits dynamically based on user input and preferences.
    """
    user_embedding = get_embeddings(query.query)
    recommendations = []

    # Compute similarity
    for outfit_id, embedding in outfit_embeddings.items():
        similarity = cosine_similarity(user_embedding, embedding)
        outfit = next(outfit for outfit in fashion_data if outfit["outfit_id"] == outfit_id)

        # Apply optional filters
        if query.color and query.color.lower() != outfit["color"].lower():
            continue
        if query.style and query.style.lower() != outfit["style"].lower():
            continue
        if query.occasion and query.occasion.lower() != outfit["occasion"].lower():
            continue

        recommendations.append((outfit, float(similarity[0][0])))

    # Sort and diversify recommendations
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    seen_categories = set()
    diverse_recommendations = []
    for rec in recommendations:
        if rec[0]['category'] not in seen_categories:
            diverse_recommendations.append(rec)
            seen_categories.add(rec[0]['category'])
        if len(diverse_recommendations) >= 5:
            break

    if not diverse_recommendations:
        return {
            "message": "Sorry, no outfits match your preferences. Try adjusting your filters or being more specific!",
            "recommendations": []
        }

    # Construct response with image links
    conversational_recommendations = [
        {
            "title": rec[0]['title'],
            "description": rec[0]['description'],
            "category": rec[0]['category'],
            "rating": rec[0]['rating'],
            "images": rec[0]['image_urls']
        }
        for rec in diverse_recommendations
    ]

    return {
        "message": "Here are some personalized outfit recommendations:",
        "recommendations": conversational_recommendations
    }


@app.get("/")
async def root():
    return {"message": "Fashion Outfit Recommendation System is running!"}
