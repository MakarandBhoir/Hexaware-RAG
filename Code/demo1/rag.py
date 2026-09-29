"""
Contoso HR Policy RAG Application

Pipeline:
    Retrieve -> Augment -> Generate

Services:
    - Azure AI Foundry / Azure OpenAI v1 API
      - Embeddings
      - Chat model
    - Azure AI Search
      - Vector search

Requirements:
    pip install openai azure-search-documents python-dotenv

Environment variables (.env):

    AZURE_AI_FOUNDRY_ENDPOINT
    AZURE_AI_FOUNDRY_API_KEY

    AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    AZURE_OPENAI_CHAT_DEPLOYMENT

    AZURE_SEARCH_ENDPOINT
    AZURE_SEARCH_API_KEY
    AZURE_SEARCH_INDEX
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Azure AI Search index field names
# ============================================================

KEY_FIELD = "chunk_id"
CONTENT_FIELD = "chunk"
TITLE_FIELD = "title"
VECTOR_FIELD = "text_vector"

# ============================================================
# Validate configuration
# ============================================================

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

    missing = []

    for variable in required_variables:
        if not os.getenv(variable):
            missing.append(variable)

    if missing:
        print("ERROR: Missing environment variables:")
        for variable in missing:
            print(f"  - {variable}")

        print("\nCheck your .env file.")
        raise RuntimeError("Required environment variables are missing.")


# ============================================================
# Create Azure clients
# ============================================================

def get_clients():

    # Azure AI Foundry / Azure OpenAI v1 API
    aoai = OpenAI(
        base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
    )

    # Azure AI Search
    search_client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX"],
        credential=AzureKeyCredential(
            os.environ["AZURE_SEARCH_API_KEY"]
        ),
    )

    return aoai, search_client


# ============================================================
# Print Azure AI Search index fields
# ============================================================

def print_index_fields():

    index_client = SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(
            os.environ["AZURE_SEARCH_API_KEY"]
        ),
    )

    index = index_client.get_index(
        os.environ["AZURE_SEARCH_INDEX"]
    )

    print(f"\nFields in index '{index.name}':")

    for field in index.fields:
        print(
            f"  - {field.name}"
            f" | type={field.type}"
            f" | searchable={field.searchable}"
        )


# ============================================================
# Generate embedding
# ============================================================

def embed(aoai, embedding_deployment, text):

    response = aoai.embeddings.create(
        model=embedding_deployment,
        input=text,
    )

    return response.data[0].embedding


# ============================================================
# STEP 1 - RETRIEVE
#
# Convert question into embedding and perform
# vector search against Azure AI Search.
# ============================================================

def retrieve(
    aoai,
    embedding_deployment,
    search_client,
    user_question,
    k=3,
):

    print("\nGenerating query embedding...")

    query_vector = embed(
        aoai,
        embedding_deployment,
        user_question,
    )

    print(
        f"Embedding generated "
        f"({len(query_vector)} dimensions)"
    )

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=k,
        fields=VECTOR_FIELD,
    )

    print("Searching Azure AI Search...")

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=[
            KEY_FIELD,
            TITLE_FIELD,
            CONTENT_FIELD,
        ],
    )

    retrieved = list(results)

    print(
        f"\nRetrieved {len(retrieved)} chunks:"
    )

    for r in retrieved:

        key = r.get(KEY_FIELD, "Unknown")
        title = r.get(TITLE_FIELD, "Untitled")
        content = r.get(CONTENT_FIELD, "")

        print(
            f"\n[{key}]"
            f"\nTitle   : {title}"
            f"\nContent : {content[:200]}..."
        )

    return retrieved


# ============================================================
# STEP 2 - AUGMENT
#
# Build the prompt using retrieved documents.
# ============================================================

def augment(retrieved, user_question):

    context_parts = []

    for r in retrieved:

        title = r.get(
            TITLE_FIELD,
            "Untitled",
        )

        content = r.get(
            CONTENT_FIELD,
            "",
        )

        context_parts.append(
            f"[Policy: {title}]\n{content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are the Contoso India HR Policy Assistant.

Answer the user's question using ONLY the information
contained in the context below.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the context,
   say that you don't have that information.
4. Mention the policy name/source used for the answer.
5. Give a concise and clear answer.
6. If multiple policies are relevant, mention each
   relevant policy.

Context:
-------------------------
{context}
-------------------------

Question:
{user_question}
"""

    return prompt


# ============================================================
# STEP 3 - GENERATE
#
# Send augmented prompt to chat model.
# ============================================================

def generate(
    aoai,
    chat_deployment,
    prompt,
    retrieved,
):

    print("\nGenerating answer...\n")

    response = aoai.chat.completions.create(
        model=chat_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful HR policy assistant. "
                    "Answer only from the supplied context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=1,
    )

    answer = response.choices[0].message.content

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for r in retrieved:

        key = r.get(
            KEY_FIELD,
            "Unknown",
        )

        title = r.get(
            TITLE_FIELD,
            "Untitled",
        )

        print(f"- {title} ({key})")

    return answer


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    try:

        # ----------------------------------------------------
        # Validate environment
        # ----------------------------------------------------

        validate_environment()

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        embedding_deployment = os.environ[
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ]

        chat_deployment = os.environ[
            "AZURE_OPENAI_CHAT_DEPLOYMENT"
        ]

        print("\n" + "=" * 70)
        print("Contoso India HR Policy - RAG Assistant")
        print("=" * 70)

        print(
            f"\nEmbedding deployment : "
            f"{embedding_deployment}"
        )

        print(
            f"Chat deployment      : "
            f"{chat_deployment}"
        )

        print(
            f"Search index         : "
            f"{os.environ['AZURE_SEARCH_INDEX']}"
        )

        # ----------------------------------------------------
        # Create clients
        # ----------------------------------------------------

        aoai, search_client = get_clients()

        # ----------------------------------------------------
        # Ask question
        # ----------------------------------------------------

        user_question = input(
            "\nAsk a question about "
            "Contoso India HR policy:\n> "
        )

        if not user_question.strip():

            print("Question cannot be empty.")

            exit()

        # ----------------------------------------------------
        # STEP 1 - Retrieve
        # ----------------------------------------------------

        retrieved_chunks = retrieve(
            aoai,
            embedding_deployment,
            search_client,
            user_question,
            k=3,
        )

        # ----------------------------------------------------
        # STEP 2 - Augment
        # ----------------------------------------------------

        prompt = augment(
            retrieved_chunks,
            user_question,
        )

        # Uncomment if you want to see the complete prompt:
        #
        # print("\n===== AUGMENTED PROMPT =====")
        # print(prompt)

        # ----------------------------------------------------
        # STEP 3 - Generate
        # ----------------------------------------------------

        generate(
            aoai,
            chat_deployment,
            prompt,
            retrieved_chunks,
        )

    except Exception as ex:

        print("\nERROR:")
        print(ex)

        print("\nDetailed error:")
        import traceback
        traceback.print_exc()