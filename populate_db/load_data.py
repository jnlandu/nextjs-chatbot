import json
import os


import split_q_and_a

# from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from huggingface_hub import login


import time

from dotenv import load_dotenv
from astrapy.db import AstraDBCollection

#To do: add logger
load_dotenv() 


# Grab the Astra token and api endpoint from the environment
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
namespace = os.getenv("ASTRA_DB_NAMESPACE")
collection_name = os.getenv("ASTRADB_COLLECTION")
dimension = os.getenv("VECTOR_DIMENSION")
model = os.getenv("MODEL")
hf_token = os.getenv("HUGGINGFACE_API_KEY")


# print(f"Using Model: {model}")
SCRAPED_FILE="./scrape/scraped_results.json"
input_data = SCRAPED_FILE
# model = MODEL

# if not hf_token:
#     print("No Hugging Face token found. Please set the Hugging Face token in the environment variable HUGGINGFACE_TOKEN")
# else:
#     login(token=hf_token)
#     print("Login successfull to Hugging Face Hub")
#     print(f"Using Hugging Face token: {hf_token}")

# if not model:
#     # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     embeddings =HuggingFaceInferenceAPIEmbeddings()
# else:
if model:
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=model)
    embeddings =HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name=model,
    )
    print(f"Using model: {model}")
else:
    print("No model found. Please set the model in the environment variable VECTOR_MODEL")
    print("Using the default model: deepset/sentence_bert"),
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="intfloat/multilingual-e5-large",
    )
    print(f"Using model: {model}")

if not input_data:
    print("No input data found. Please set the input data in the environment variable SCRAPED_FILE")
    print("Using the default input data: scraped_results.json")
    input_data = "scraped_results.json"

else:
    print(f"Using input data: {input_data}")

def get_input_data():
    scraped_results_file = input_data
    with open(scraped_results_file) as f:
        scraped_data = json.load(f)

        faq_scraped_data = []
        for d in scraped_data:
            if "faq" in d["url"].lower():
                faq_scraped_data.append(d)
    return faq_scraped_data

def embed(text_to_embed):
    embedding = list(embeddings.embed_query(text_to_embed))
    return embedding

def main():
    if not namespace:
        collection = AstraDBCollection(collection_name=collection_name, token=token,
                                       api_endpoint=api_endpoint)
    else:
        collection = AstraDBCollection(collection_name=collection_name, token=token,
                                       api_endpoint=api_endpoint, namespace=namespace)

    input_data_faq = get_input_data()

    # process faq data
    for webpage in input_data_faq:
        print("Debugging webpage: ", webpage)
        q_and_a_data = split_q_and_a.split(webpage)
        print("Debugging q_and_b_data: ", len(q_and_a_data["questions"]))
        for i in range (0,len(q_and_a_data["questions"])):
            document_id = webpage["url"]
            question_id = i + 1
            question = q_and_a_data["questions"][i]
            answer = q_and_a_data["answers"][i]
            text_to_embed = f"{question}"
            embedding = embed(text_to_embed)
            time.sleep(1)
            to_insert = {"document_id": document_id, "question_id": question_id, "answer": answer,
                             "question": question, "$vector": embedding}
            if (question == " Cluster?") or (question == "?"):
                print("Malformed question. Not adding to vector db.")
            else:
                result = collection.insert_one(to_insert)
                print(f"{result} \tdocument_id: {document_id} question_id: {question_id}")

if __name__ == "__main__":
    main()
