import os

def get_llm():
    """
    Returns a LangChain LLM based on environment variables.
    If a GOOGLE_API_KEY is present, it will be used exclusively.
    Otherwise, it will fall back to OpenAI, and finally a local model.
    """
    # 1. If a Google API key is provided, exclusively try to use it.
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("✅ Attempting to use Google Gemini model...")
            
            # Add a specific check for placeholder keys
            if "your-google-api-key" in os.getenv("GOOGLE_API_KEY"):
                 print("❌ The GOOGLE_API_KEY in your .env file is still a placeholder. Please add your real key.")
                 return None

            # UPDATED: Changed model name to a current, valid model.
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
            print("✅ Google Gemini model loaded successfully.")
            return llm
        except ImportError:
            print("⚠️ To use Google Gemini, please run: pip install langchain-google-genai")
            return None # Stop execution
        except Exception as e:
            # This will catch authentication errors if the Google key is invalid
            print(f"❌ An error occurred while initializing Google Gemini: {e}")
            print("   Please ensure your Google API key in the .env file is correct and has the 'Generative Language API' enabled in your Google Cloud project.")
            return None # CRITICAL: Stop execution and do not fall back to OpenAI

    # 2. If no Google key, try OpenAI as a fallback.
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            print("✅ Using OpenAI model.")
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        except Exception as e:
            print(f"❌ Error initializing OpenAI: {e}")
            return None
            
    # 3. If no API keys are found, use the local model.
    print("✅ No API key found. Using local HuggingFace model as a fallback.")
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline
    try:
        pipe = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"❌ Failed to load local model: {e}")
        return None

