import sys
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


import os
import json
from dotenv import load_dotenv
from astrapy.db import AstraDBCollection

load_dotenv(verbose=True) 

# Grab the Astra token and api endpoint from the environment
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
namespace = os.getenv("ASTRA_DB_NAMESPACE")
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HUGGINGFACE_API_KEY")
collection_name = os.getenv("ASTRA_DB_COLLECTION")
model = os.getenv("VECTOR_MODEL")


# langchain openai interface
# llm = OpenAI(openai_api_key=openai_api_key)
# langchain groq interface
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=groq_api_key,
)

# if not model:
#     embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
# else:
    # embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model=model)
if model:
    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name=model,
    )
else:
    print("No model found. Please set the model in the environment variable VECTOR_MODEL")

def get_similar_docs(query, number):
    if not namespace:
        collection = AstraDBCollection(collection_name=collection_name, token=token,
                                       api_endpoint=api_endpoint)
    else:
        collection = AstraDBCollection(collection_name=collection_name, token=token,
                                       api_endpoint=api_endpoint, namespace=namespace)
    embedding = list(embedding_model.embed_query(query))
    relevant_docs = collection.vector_find(embedding, limit=number)

    docs_contents = [row['answer'] for row in relevant_docs]
    docs_urls = [row['document_id'] for row in relevant_docs]
    return docs_contents, docs_urls

# prompt that is sent to openai using the response from the vector database and the users original query
prompt_boilerplate = "Answer the question posed in the user query section using the provided context"
user_query_boilerplate = "USER QUERY: "
document_context_boilerplate = "CONTEXT: "
final_answer_boilerplate = "Final Answer: "

def build_full_prompt(query):
    relevant_docs, urls = get_similar_docs(query, 3)
    docs_single_string = "\n".join(relevant_docs)

    if not urls:
        print("No URLs found for the given query")
    url = urls[0] # set(urls)

    nl = "\n"
    filled_prompt_template = prompt_boilerplate + nl + user_query_boilerplate+ query + nl + document_context_boilerplate + docs_single_string + nl + final_answer_boilerplate
    return filled_prompt_template, url

def send_to_groq(full_prompt):
    response = llm.invoke(full_prompt)
    return json.dumps(response.content)


if __name__ == "__main__":
    print("Running test")
    print("Test passed")

