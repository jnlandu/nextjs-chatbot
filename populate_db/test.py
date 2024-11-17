from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
# model = os.getenv("MODEL")
# model = "sentence-transformers/all-MiniLM-l6-v2"
model="intfloat/multilingual-e5-large-instruct"
hf_token = os.getenv("HUGGINGFACE_API_KEY")

if model:
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=model)
    embeddings =HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name=model,
    )
    print(f"Using model: {model}")
    text = "This is a test document"
    query_result = embeddings.embed_query(text)
    print(f"Test embedding: {type(query_result)}")
    print(f"Test embedding: {query_result[:3]}")

else:
    print("No model found. Please set the model in the environment variable VECTOR_MODEL")
    print("Using the default model: deepset/sentence_bert"),

if "__name__" == "__main__":
    print("Running test")
    print("Test passed")