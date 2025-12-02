import importlib.metadata

try:
    ver = importlib.metadata.version('langchain-qdrant')
    print(f"langchain-qdrant version: {ver}")
except:
    print("langchain-qdrant not installed")
