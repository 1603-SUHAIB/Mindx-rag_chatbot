import os

def get_llm():
    """
    Returns a LangChain LLM based on hardcoded or environment variables.
    Priority: Google Gemini -> OpenAI -> Local HuggingFace.
    """
    GOOGLE_API_KEY = "AIzaSyDg6ZKd9MZKXbLqlJGr2hwYrDIDJOgZ4cE"

    if GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("✅ Attempting to use Google Gemini model...")

            if GOOGLE_API_KEY.strip() == "" or "your-google-api-key" in GOOGLE_API_KEY:
                print("❌ The GOOGLE_API_KEY is invalid. Please add your real key.")
                return None

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY
            )
            print("✅ Google Gemini model loaded successfully.")
            return llm
        except ImportError:
            print("⚠️ To use Google Gemini, please run: pip install langchain-google-genai")
            return None
        except Exception as e:
            print(f"❌ An error occurred while initializing Google Gemini: {e}")
            return None

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            print("✅ Using OpenAI model.")
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        except Exception as e:
            print(f"❌ Error initializing OpenAI: {e}")
            return None

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
