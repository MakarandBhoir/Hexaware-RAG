"""
Contoso HR Policy RAG Application — LangChain version

Pipeline:
    Retrieve -> Augment -> Generate

LangChain is used for:
    - Embedding generation
    - Prompt templating
    - Chat model invocation and output parsing

Azure AI Search SDK is retained for vector retrieval so the app continues
using the existing index fields:
    chunk_id, chunk, title, text_vector

Install:
    pip install -r requirements.txt
"""

import os
import traceback

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery


load_dotenv()

# Azure AI Search index field names
KEY_FIELD = "chunk_id"
CONTENT_FIELD = "chunk"
TITLE_FIELD = "title"
VECTOR_FIELD = "text_vector"


def validate_environment():
    required_variables = [
        "AZURE_AI_FOUNDRY_ENDPOINT",
        "AZURE_AI_FOUNDRY_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX",
    ]

    missing = [name for name in required_variables if not os.getenv(name)]
    if missing:
        print("ERROR: Missing environment variables:")
        for name in missing:
            print(f"  - {name}")
        print("\\nCheck your .env file.")
        raise RuntimeError("Required environment variables are missing.")


def get_clients():
    # Azure AI Foundry / Azure OpenAI v1 endpoint.
    # LangChain's OpenAI-compatible integrations use the supplied base URL.
    endpoint = os.environ["AZURE_AI_FOUNDRY_ENDPOINT"].rstrip("/") + "/"
    api_key = os.environ["AZURE_AI_FOUNDRY_API_KEY"]

    embeddings = OpenAIEmbeddings(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        base_url=endpoint,
        api_key=api_key,
    )

    chat_model = ChatOpenAI(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        base_url=endpoint,
        api_key=api_key,
        temperature=1,
    )

    search_client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )

    return embeddings, chat_model, search_client


def print_index_fields():
    index_client = SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )
    index = index_client.get_index(os.environ["AZURE_SEARCH_INDEX"])

    print(f"\\nFields in index '{index.name}':")
    for field in index.fields:
        print(
            f"  - {field.name}"
            f" | type={field.type}"
            f" | searchable={field.searchable}"
        )


# STEP 1 — RETRIEVE
def retrieve(embeddings, search_client, user_question, k=3):
    print("\\nGenerating query embedding...")
    query_vector = embeddings.embed_query(user_question)
    print(f"Embedding generated ({len(query_vector)} dimensions)")

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=k,
        fields=VECTOR_FIELD,
    )

    print("Searching Azure AI Search...")
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=[KEY_FIELD, TITLE_FIELD, CONTENT_FIELD],
    )
    retrieved = list(results)

    print(f"\\nRetrieved {len(retrieved)} chunks:")
    for result in retrieved:
        key = result.get(KEY_FIELD, "Unknown")
        title = result.get(TITLE_FIELD, "Untitled")
        content = result.get(CONTENT_FIELD, "")
        print(f"\\n[{key}]\\nTitle   : {title}\\nContent : {content[:200]}...")

    return retrieved


# STEP 2 — AUGMENT
def build_rag_chain(chat_model):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the Contoso India HR Policy Assistant.
Answer the user's question using ONLY the information in the supplied context.

Rules:
1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the context, say that you don't have that information.
4. Mention the policy name/source used for the answer.
5. Give a concise and clear answer.
6. If multiple policies are relevant, mention each relevant policy.""",
            ),
            (
                "human",
                "Context:\\n-------------------------\\n{context}\\n"
                "-------------------------\\nQuestion:\\n{question}",
            ),
        ]
    )

    # LCEL: prompt -> chat model -> plain text
    return prompt | chat_model | StrOutputParser()


def build_context(retrieved):
    context_parts = []
    for result in retrieved:
        title = result.get(TITLE_FIELD, "Untitled")
        content = result.get(CONTENT_FIELD, "")
        context_parts.append(f"[Policy: {title}]\\n{content}")
    return "\\n\\n".join(context_parts)


# STEP 3 — GENERATE
def generate(rag_chain, retrieved, user_question):
    print("\\nGenerating answer...\\n")
    answer = rag_chain.invoke(
        {
            "context": build_context(retrieved),
            "question": user_question,
        }
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)
    if retrieved:
        for result in retrieved:
            key = result.get(KEY_FIELD, "Unknown")
            title = result.get(TITLE_FIELD, "Untitled")
            print(f"- {title} ({key})")
    else:
        print("- No documents were retrieved.")

    return answer


def main():
    validate_environment()

    embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    print("\\n" + "=" * 70)
    print("Contoso India HR Policy - LangChain RAG Assistant")
    print("=" * 70)
    print(f"\\nEmbedding deployment : {embedding_deployment}")
    print(f"Chat deployment      : {chat_deployment}")
    print(f"Search index         : {os.environ['AZURE_SEARCH_INDEX']}")

    embeddings, chat_model, search_client = get_clients()
    rag_chain = build_rag_chain(chat_model)

    user_question = input(
        "\\nAsk a question about Contoso India HR policy:\\n> "
    ).strip()

    if not user_question:
        print("Question cannot be empty.")
        return

    retrieved_chunks = retrieve(
        embeddings,
        search_client,
        user_question,
        k=3,
    )
    generate(rag_chain, retrieved_chunks, user_question)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("\\nERROR:")
        print(ex)
        print("\\nDetailed error:")
        traceback.print_exc()
