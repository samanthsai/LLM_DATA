# from fastapi import FastAPI
# from pydantic import BaseModel
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoTokenizer, AutoModel
# import json
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
#
# app = FastAPI()
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # You can specify domains instead of "*"
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )
#
# # Load dataset
# with open("dataset.json", "r") as f:
#     course_data = json.load(f)
#
# # Pre-trained model and tokenizer
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
#
#
# # Precompute embeddings for courses
# def get_embeddings(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).detach().numpy()
#
#
# course_embeddings = {
#     course["course_id"]: get_embeddings(course["description"])
#     for course in course_data
# }
#
#
# # User query model
# class Query(BaseModel):
#     query: str
#     difficulty: str = None  # Optional filter for difficulty level
#     category: str = None  # Optional filter for category
#
#
# @app.post("/recommend/")
# async def recommend_courses(query: Query):
#     """
#     Recommend courses based on user query.
#     """
#     user_embedding = get_embeddings(query.query)
#     recommendations = []
#
#     # Compute similarity
#     for course_id, embedding in course_embeddings.items():
#         similarity = cosine_similarity(user_embedding, embedding)
#         course = next(course for course in course_data if course["course_id"] == course_id)
#
#         # Apply optional filters
#         if query.difficulty and query.difficulty != course["difficulty"]:
#             continue
#         if query.category and query.category != course["category"]:
#             continue
#
#         # Convert similarity to float for JSON compatibility
#         recommendations.append((course, float(similarity[0][0])))
#
#     # Sort recommendations by similarity
#     recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
#
#     return {
#         "recommendations": [
#             {
#                 "title": rec[0]["title"],
#                 "description": rec[0]["description"],
#                 "category": rec[0]["category"],
#                 "difficulty": rec[0]["difficulty"],
#                 "similarity_score": rec[1]
#             }
#             for rec in recommendations[:5]
#         ]
#     }
#
#
# @app.get("/")
# async def root():
#     return {"message": "E-Learning Content Recommendation System is running!"}
