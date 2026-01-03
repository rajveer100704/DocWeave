"""
Prompt templates for RAG answering with strict grounding and citations.
"""

from langchain_core.prompts import PromptTemplate

# Core system safety + grounding rules
SYSTEM_PROMPT = """
You are a helpful assistant that answers questions using the provided context.
Provide concise, accurate, and to-the-point answers based on the context and provide answer in a cleaned structured way.
Be brief and direct - focus on answering the question without unnecessary elaboration.
Do not include citations or source references in your answer - sources will be provided separately.
Only say "I don't know" if the context does not contain any relevant information to answer the question.
""".strip()


def build_stuff_prompt() -> PromptTemplate:
    """Prompt for the simple Stuff strategy (small context)."""
    template = """Context:
    {context}

    Question: {question}

    Based on the context provided above, answer the question concisely and directly.
    Provide a brief, to-the-point answer using only the relevant information from the context.
    Be precise and avoid unnecessary details or elaboration.
    Only say "I don't know" if the context does not contain any information that relates to the question.
    Do not include citations or source references in your answer.""".strip()
    return PromptTemplate(
        input_variables=["context", "question"], template=template
    )


def build_map_prompt() -> PromptTemplate:
    """Prompt for the map step of Map-Reduce."""
    template = """Context chunk:
    {context}

    Question: {question}

    Extract only the key information from this context chunk that directly answers the question.
    Return brief, concise bullet points with only the essential facts.
    Be selective - include only what's directly relevant to the question.
    If this chunk contains relevant information, extract it concisely. Only return "No relevant information" if this chunk truly has nothing related to the question.
    Do not include citations or source references.""".strip()
    return PromptTemplate(
        input_variables=["context", "question"], template=template
    )


def build_reduce_prompt() -> PromptTemplate:
    """Prompt for the reduce step of Map-Reduce."""
    template = """You received the following partial answers from different context chunks:
    {map_summaries}

    Question: {question}

    Combine all the partial answers into a single, concise, and direct answer.
    Synthesize the key information from all partial answers into a brief response.
    Be brief and to-the-point - avoid repetition and unnecessary details.
    Only say "I don't know" if none of the partial answers contain any information related to the question.
    Do not include citations or source references.""".strip()
    return PromptTemplate(
        input_variables=["map_summaries", "question"],
        template=template,
    )

