from qdrant_client import QdrantClient
import sys

try:
    print("Imported QdrantClient")
    print(f"QdrantClient methods: {[m for m in dir(QdrantClient) if 'search' in m]}")
    
    client = QdrantClient(location=":memory:")
    print(f"Client instance has search: {hasattr(client, 'search')}")
    
    # Check version
    import qdrant_client
    print(f"Version: {qdrant_client.__version__}")

except Exception as e:
    print(f"Error: {e}")

