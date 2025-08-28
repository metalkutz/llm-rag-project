import asyncio
import sys
from pathlib import Path

from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.config.config import get_settings

async def test_llm_directly(rag_pipeline):
    """Test the LLM directly to see if it's working."""
    print("\n🧠 Testing LLM directly...")
    try:
        # Test with a simple prompt
        simple_prompt = "What is AI? Answer in one sentence."
        response = rag_pipeline.llm.complete(simple_prompt)
        
        print(f"Direct LLM test:")
        print(f"  Response type: {type(response)}")
        print(f"  Response text: '{response.text}'")
        print(f"  Response length: {len(response.text)}")
        
        if len(response.text.strip()) == 0:
            print("  ❌ LLM returning empty responses!")
            return False
        else:
            print("  ✅ LLM is working!")
            return True
            
    except Exception as e:
        print(f"  ❌ Direct LLM test failed: {e}")
        return False

async def test_retrieval_only(rag_pipeline, query):
    """Test just the retrieval part."""
    print(f"\n🔍 Testing retrieval only...")
    try:
        retriever = VectorIndexRetriever(
            index=rag_pipeline.index,
            similarity_top_k=3
        )
        
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes:")
        
        for i, node in enumerate(nodes):
            print(f"  Node {i+1}:")
            print(f"    Score: {getattr(node, 'score', 'N/A')}")
            print(f"    Text preview: {node.text[:100]}...")
            print(f"    Metadata: {node.metadata}")
        
        return nodes
        
    except Exception as e:
        print(f"  ❌ Retrieval test failed: {e}")
        return []

async def test_with_manual_context(rag_pipeline, query, nodes):
    """Test LLM with manually created context."""
    print(f"\n🛠️ Testing with manual context...")
    try:
        # Extract clean content from nodes (they contain Q&A format)
        context_parts = []
        for node in nodes[:2]:  # Use only top 2 nodes
            # Extract the answer part if it's in Q&A format
            text = node.text
            if "Answer:" in text:
                # Extract just the answer part
                answer_part = text.split("Answer:")[-1].strip()
                context_parts.append(answer_part[:150])  # Limit to 150 chars
            else:
                context_parts.append(text[:150])
        
        context = " ".join(context_parts)  # Join with space instead of newlines
        
        # Create simpler manual prompt
        manual_prompt = f"""Information: {context}

Question: {query}
Answer:"""
        
        print(f"Manual prompt length: {len(manual_prompt)} characters")
        print(f"Manual prompt: {manual_prompt[:300]}...")
        
        response = rag_pipeline.llm.complete(manual_prompt)
        
        print(f"Manual response:")
        print(f"  Type: {type(response)}")
        print(f"  Text: '{response.text}'")
        print(f"  Length: {len(response.text)}")
        
        return response.text
        
    except Exception as e:
        print(f"  ❌ Manual context test failed: {e}")
        return ""

async def main():
    print("🧪 Comprehensive RAG Testing\n")
    
    # Load settings and initialize pipeline
    settings = get_settings()
    rag_pipeline = RAGPipeline(settings)
    await rag_pipeline.initialize()

    # Check pipeline health
    health = await rag_pipeline.health_check()
    print(f"Pipeline health: {health}")
    
    stats = await rag_pipeline.get_stats()
    print(f"Pipeline stats: {stats}")
    
    if stats.get('document_count', 0) == 0:
        print("❌ No documents in vector store! Run ingestion first.")
        return
    
    test_query = "What is Retrieval-Augmented Generation?"
    
    # Test 1: Direct LLM
    llm_works = await test_llm_directly(rag_pipeline)
    
    # Test 2: Retrieval only
    nodes = await test_retrieval_only(rag_pipeline, test_query)
    
    # Test 3: Manual context
    if llm_works and nodes:
        manual_result = await test_with_manual_context(rag_pipeline, test_query, nodes)
    
    # Test 4: Standard pipeline query
    print(f"\n🔍 Testing standard pipeline query...")
    try:
        result = await rag_pipeline.query(test_query)
        print(f"Standard pipeline result:")
        print(f"  Answer: '{result['answer']}'")
        print(f"  Length: {len(result['answer'])}")
        print(f"  Sources: {len(result['sources'])}")
        print(f"  Confidence: {result.get('confidence_score', 'N/A')}")
    except Exception as e:
        print(f"  ❌ Standard query failed: {e}")
    
    # Test 5: Custom query engine
    print(f"\n🛠️ Testing custom query engine...")
    try:
        # Create simple, short prompt
        qa_prompt_template = PromptTemplate(
            """Context: {context_str}
Question: {query_str}
Answer:"""
        )
        
        response_synthesizer = get_response_synthesizer(
            llm=rag_pipeline.llm,
            text_qa_template=qa_prompt_template,
            response_mode=ResponseMode.COMPACT,
            use_async=True
        )
        
        retriever = VectorIndexRetriever(
            index=rag_pipeline.index,
            similarity_top_k=2  # Reduce to 2 sources
        )
        
        custom_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
        
        result = custom_query_engine.query(test_query)
        
        print(f"Custom query engine result:")
        print(f"  Response type: {type(result)}")
        print(f"  Response object: {result}")
        
        # Try different ways to access the response
        response_text = None
        
        # The correct way to access LlamaIndex Response
        response_text = str(result)  # Response objects can be cast to string
        print(f"  str(result): '{response_text}'")
        
        print(f"  Source nodes: {len(getattr(result, 'source_nodes', []))}")
        
        # Print source node details
        if hasattr(result, 'source_nodes'):
            for i, node in enumerate(result.source_nodes):
                print(f"    Source {i+1}: {node.text[:50]}...")
        
    except Exception as e:
        print(f"  ❌ Custom query failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())