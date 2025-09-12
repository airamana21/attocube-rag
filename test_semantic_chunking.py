#!/usr/bin/env python3
"""
Quick test script to verify semantic chunking functionality
"""

from rag import SemanticChunker
from langchain.schema import Document

def test_semantic_chunking():
    print("Testing Semantic Chunking...")
    
    # Create a test chunker
    chunker = SemanticChunker(max_chunk_size=200)
    
    # Create test documents
    test_pages = [
        Document(
            page_content="This is the first paragraph about temperature control. The system maintains precise temperatures. Next, we discuss pressure settings. The pressure must be carefully monitored. Finally, we cover safety procedures. Safety is paramount in all operations.",
            metadata={"source": "test.pdf", "page": 1, "doc_type": "manual"}
        )
    ]
    
    # Test chunking
    chunks = chunker.chunk_documents(test_pages)
    
    print(f"Original document length: {len(test_pages[0].page_content)} characters")
    print(f"Number of chunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk.page_content)} chars):")
        print(f"Content: {chunk.page_content[:100]}...")
        print(f"Metadata preserved: {chunk.metadata}")
    
    print("\n✅ Semantic chunking test completed successfully!")

if __name__ == "__main__":
    test_semantic_chunking()
