import os

def get_llm():
    """
    Returns a LangChain LLM:
    - If OPENAI_API_KEY is set -> use OpenAI Chat.
    - Else -> use a small local HuggingFace text2text model.
    """
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    else:
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline
        pipe = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )
        return HuggingFacePipeline(pipeline=pipe)
