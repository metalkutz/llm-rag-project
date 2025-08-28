"""
Download and prepare sample data for the RAG system.
This script downloads ML/AI educational content and Q&A pairs.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def create_sample_qa_data() -> List[Dict[str, Any]]:
    """Create sample Q&A data about AI/ML/RAG topics."""
    qa_data = [
        {
            "question": "What is Retrieval-Augmented Generation (RAG)?",
            "answer": "Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge retrieval. Instead of relying solely on the model's training data, RAG systems first retrieve relevant documents from a knowledge base and then use this information to generate more accurate and contextually relevant responses. This approach helps reduce hallucinations and provides more up-to-date information.",
            "source": "AI Research Papers",
            "category": "RAG",
            "difficulty": "beginner"
        },
        {
            "question": "How does ChromaDB work as a vector database?",
            "answer": "ChromaDB is an open-source vector database designed for storing and querying high-dimensional embeddings. It works by converting text and other data into vector representations using embedding models, then storing these vectors in an optimized format for similarity search. When you query ChromaDB, it finds the most similar vectors using distance metrics like cosine similarity or Euclidean distance. ChromaDB supports persistence, metadata filtering, and can handle large-scale vector collections efficiently.",
            "source": "ChromaDB Documentation",
            "category": "Vector Databases",
            "difficulty": "intermediate"
        },
        {
            "question": "What are the benefits of using LangChain for RAG pipelines?",
            "answer": "LangChain provides several benefits for building RAG pipelines: 1) Modular components that can be easily combined and customized, 2) Built-in integrations with popular vector databases and LLMs, 3) Document loaders for various file formats (PDF, web pages, etc.), 4) Text splitters for optimal chunking strategies, 5) Memory management for conversational AI, 6) Chains and agents for complex reasoning workflows, and 7) Evaluation tools for measuring RAG performance. This makes it easier to build, test, and deploy production-ready RAG systems.",
            "source": "LangChain Documentation",
            "category": "RAG Frameworks",
            "difficulty": "intermediate"
        },
        {
            "question": "Explain the role of embeddings in RAG systems",
            "answer": "Embeddings are vector representations of text that capture semantic meaning in high-dimensional space. In RAG systems, embeddings serve multiple crucial roles: 1) Document encoding - converting knowledge base documents into searchable vectors, 2) Query encoding - transforming user questions into the same vector space, 3) Similarity matching - enabling fast retrieval of relevant documents through vector similarity, 4) Semantic understanding - allowing the system to find conceptually related content even when exact keywords don't match. The quality of embeddings directly impacts RAG performance, making model selection critical.",
            "source": "NLP Research",
            "category": "Embeddings",
            "difficulty": "intermediate"
        },
        {
            "question": "How do you evaluate RAG system performance?",
            "answer": "RAG system evaluation involves multiple metrics across different dimensions: 1) Retrieval metrics - measuring how well the system finds relevant documents (precision@k, recall@k, NDCG), 2) Generation metrics - assessing answer quality (BLEU, ROUGE, human evaluation), 3) End-to-end metrics - evaluating overall system performance (accuracy, helpfulness, truthfulness), 4) Faithfulness - ensuring answers are grounded in retrieved documents, 5) Relevance - measuring how well retrieved documents match the query, and 6) Latency and throughput for production deployment. A/B testing with human evaluators often provides the most reliable assessment.",
            "source": "RAG Evaluation Research",
            "category": "Evaluation",
            "difficulty": "advanced"
        },
        {
            "question": "What is the difference between fine-tuning and RAG?",
            "answer": "Fine-tuning and RAG are two different approaches to customizing language models: Fine-tuning involves training a pre-trained model on domain-specific data, updating the model's parameters to specialize it for particular tasks or knowledge areas. This creates a static knowledge base within the model. RAG, on the other hand, keeps the base model unchanged and instead retrieves relevant information from an external knowledge base at inference time. RAG offers advantages like real-time knowledge updates, transparency in sources, and lower computational costs for adding new information, while fine-tuning can provide more seamless integration of domain knowledge.",
            "source": "Machine Learning Best Practices",
            "category": "Model Customization",
            "difficulty": "intermediate"
        },
        {
            "question": "What are common challenges in implementing RAG systems?",
            "answer": "Common RAG implementation challenges include: 1) Chunking strategy - determining optimal document splitting for retrieval and context, 2) Embedding model selection - balancing quality, speed, and domain relevance, 3) Retrieval quality - ensuring relevant documents are found consistently, 4) Context window management - fitting retrieved content within model limits, 5) Hallucination control - preventing the model from generating information not in retrieved documents, 6) Latency optimization - maintaining fast response times, 7) Evaluation complexity - measuring system performance across multiple dimensions, and 8) Knowledge base maintenance - keeping information current and handling conflicting sources.",
            "source": "RAG Implementation Guide",
            "category": "Implementation",
            "difficulty": "advanced"
        },
        {
            "question": "How does LlamaIndex differ from other RAG frameworks?",
            "answer": "LlamaIndex (formerly GPT Index) is specifically designed for building RAG applications with several distinctive features: 1) Data connectors for 100+ data sources (APIs, PDFs, databases), 2) Advanced indexing structures (tree, keyword, knowledge graph indices), 3) Query engines with sophisticated routing and sub-question capabilities, 4) Built-in evaluation and observability tools, 5) Strong focus on production deployment with async support, 6) Tight integration with LLM providers and vector databases, and 7) Comprehensive data ingestion pipeline management. While LangChain is more general-purpose for LLM applications, LlamaIndex specializes in the data ingestion and retrieval aspects of RAG systems.",
            "source": "LlamaIndex Documentation",
            "category": "RAG Frameworks",
            "difficulty": "intermediate"
        },
        {
            "question": "What are vector similarity search algorithms?",
            "answer": "Vector similarity search algorithms are methods for finding the most similar vectors in high-dimensional space. Key algorithms include: 1) Exact search - brute force comparison using metrics like cosine similarity, Euclidean distance, or dot product, 2) Approximate Nearest Neighbor (ANN) - faster algorithms like LSH (Locality Sensitive Hashing), 3) HNSW (Hierarchical Navigable Small World) - graph-based approach offering good speed-accuracy tradeoffs, 4) IVF (Inverted File) - clustering-based method for large-scale search, and 5) Product Quantization - compression technique for memory efficiency. The choice depends on requirements for speed, accuracy, memory usage, and dataset size.",
            "source": "Information Retrieval Research",
            "category": "Vector Search",
            "difficulty": "advanced"
        },
        {
            "question": "How do you handle multi-modal data in RAG systems?",
            "answer": "Multi-modal RAG systems can process text, images, audio, and other data types: 1) Multi-modal embeddings - using models that can encode different data types into a shared vector space, 2) Modality-specific processing - separate pipelines for each data type before fusion, 3) Cross-modal retrieval - finding relevant content across different modalities, 4) Multi-modal generation - producing responses that may include text, images, or other formats, 5) Metadata utilization - leveraging structured information about multimedia content, and 6) Fusion strategies - combining information from multiple modalities effectively. Tools like CLIP for text-image understanding and specialized multi-modal embedding models enable these capabilities.",
            "source": "Multi-modal AI Research",
            "category": "Multi-modal RAG",
            "difficulty": "advanced"
        }
    ]
    
    return qa_data


def create_educational_articles() -> List[Dict[str, Any]]:
    """Create educational articles about AI/ML topics."""
    articles = [
        {
            "title": "Introduction to Vector Databases",
            "content": """Vector databases are specialized databases designed to store, index, and query high-dimensional vectors efficiently. Unlike traditional databases that work with structured data like numbers and strings, vector databases handle embeddings - dense numerical representations of unstructured data like text, images, or audio.

The core concept behind vector databases lies in the ability to perform similarity searches. When you have millions of vectors representing documents, images, or other data, you need to quickly find the most similar ones to a given query vector. Traditional databases struggle with this task because they're not optimized for high-dimensional similarity computations.

Key features of vector databases include:

1. Efficient Storage: Optimized data structures for high-dimensional vectors
2. Fast Similarity Search: Algorithms like HNSW, IVF, or LSH for approximate nearest neighbor search
3. Scalability: Ability to handle millions or billions of vectors
4. Metadata Support: Store additional information alongside vectors
5. Real-time Updates: Support for adding, updating, and deleting vectors dynamically

Popular vector databases include Pinecone, Weaviate, Qdrant, Milvus, and ChromaDB. Each has its own strengths in terms of performance, features, and ease of use.

Vector databases are essential for modern AI applications like recommendation systems, semantic search, RAG systems, and similarity-based content discovery.""",
            "source": "AI Education Portal",
            "category": "Vector Databases",
            "author": "AI Research Team"
        },
        {
            "title": "Understanding Transformer Architecture",
            "content": """The Transformer architecture, introduced in the "Attention Is All You Need" paper, revolutionized natural language processing and became the foundation for modern large language models like GPT, BERT, and T5.

Key components of the Transformer include:

1. Self-Attention Mechanism: Allows the model to weigh the importance of different words in a sequence when processing each word. This enables the model to capture long-range dependencies and contextual relationships.

2. Multi-Head Attention: Uses multiple attention mechanisms in parallel, allowing the model to focus on different types of relationships simultaneously.

3. Positional Encoding: Since Transformers don't have inherent sequence order understanding like RNNs, positional encodings are added to input embeddings to provide position information.

4. Feed-Forward Networks: Dense layers that process the attention outputs, adding non-linearity and transformation capacity.

5. Layer Normalization: Stabilizes training and improves convergence by normalizing inputs to each sub-layer.

6. Residual Connections: Skip connections that help with gradient flow during training and enable deeper networks.

The Transformer's parallel processing capability makes it much more efficient to train than sequential models like RNNs or LSTMs. This efficiency, combined with its superior performance on various NLP tasks, led to its widespread adoption and the development of increasingly large and capable language models.""",
            "source": "ML Fundamentals Guide",
            "category": "Deep Learning",
            "author": "Neural Network Specialists"
        },
        {
            "title": "Building Production RAG Systems",
            "content": """Building production-ready RAG (Retrieval-Augmented Generation) systems requires careful consideration of multiple factors beyond basic functionality. Here's a comprehensive guide to production RAG deployment:

Architecture Considerations:
- Microservices design for scalability and maintainability
- Separate services for document ingestion, retrieval, and generation
- Load balancing and horizontal scaling capabilities
- Caching strategies for frequently accessed content

Data Pipeline Management:
- Automated document ingestion and processing workflows
- Data validation and quality checks
- Version control for knowledge base updates
- Monitoring data freshness and relevance

Performance Optimization:
- Embedding model selection based on domain and latency requirements
- Vector database tuning for optimal retrieval speed
- Batch processing for efficient document updates
- Query optimization and caching strategies

Quality Assurance:
- Comprehensive evaluation metrics (relevance, faithfulness, completeness)
- A/B testing frameworks for continuous improvement
- Human-in-the-loop validation processes
- Automated testing for regression prevention

Monitoring and Observability:
- Real-time performance metrics and alerting
- Query analysis and user behavior tracking
- System health monitoring and error tracking
- Cost monitoring and optimization

Security and Compliance:
- Access control and authentication mechanisms
- Data privacy and GDPR compliance
- Audit trails for all system interactions
- Secure handling of sensitive information

The key to successful production RAG deployment is treating it as a complete system rather than just a model, with proper engineering practices, monitoring, and continuous improvement processes.""",
            "source": "Production AI Systems",
            "category": "MLOps",
            "author": "AI Engineering Team"
        }
    ]
    
    return articles


async def download_sample_data():
    """Download and prepare all sample data."""
    try:
        print("Creating sample data for RAG system...")
        
        # Ensure data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Create Q&A data
        qa_data = create_sample_qa_data()
        qa_file = data_dir / "sample_qa.json"
        
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created {len(qa_data)} Q&A pairs in {qa_file}")
        
        # Create educational articles
        articles = create_educational_articles()
        articles_file = data_dir / "educational_articles.json"
        
        with open(articles_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        print(f"Created {len(articles)} educational articles in {articles_file}")
        
        # Create combined dataset
        combined_data = []
        
        # Add Q&A data as documents
        for qa in qa_data:
            combined_data.append({
                "content": f"Question: {qa['question']}\n\nAnswer: {qa['answer']}",
                "source": qa["source"],
                "type": "qa",
                "category": qa["category"],
                "metadata": {
                    "difficulty": qa["difficulty"],
                    "question": qa["question"],
                    "answer": qa["answer"]
                }
            })
        
        # Add articles as documents
        for article in articles:
            combined_data.append({
                "content": f"{article['title']}\n\n{article['content']}",
                "source": article["source"],
                "type": "article",
                "category": article["category"],
                "metadata": {
                    "title": article["title"],
                    "author": article["author"]
                }
            })
        
        combined_file = data_dir / "combined_documents.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created combined dataset with {len(combined_data)} documents in {combined_file}")
        
        # Create a simple text file for testing
        sample_text = """
This is a sample text document for testing the RAG system.
It contains information about artificial intelligence and machine learning.

Artificial Intelligence (AI) is a broad field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. Machine Learning (ML) is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed.

Key concepts in AI/ML include:
- Neural Networks: Computing systems inspired by biological neural networks
- Deep Learning: ML techniques using neural networks with multiple layers
- Natural Language Processing: AI's ability to understand and generate human language
- Computer Vision: AI's capability to interpret and understand visual information
- Reinforcement Learning: Learning through interaction with an environment

The combination of these technologies enables powerful applications like chatbots, recommendation systems, autonomous vehicles, and medical diagnosis tools.
        """.strip()
        
        sample_text_file = data_dir / "sample_text.txt"
        with open(sample_text_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        print(f"Created sample text file: {sample_text_file}")
        
        print("✅ Sample data creation completed successfully!")
        
        return {
            "qa_data": qa_file,
            "articles": articles_file,
            "combined": combined_file,
            "sample_text": sample_text_file,
            "total_documents": len(combined_data)
        }
        
    except Exception as e:
        print(f"Failed to create sample data: {e}")
        raise


def main():
    """Main function to run the data download script."""
    import asyncio
    asyncio.run(download_sample_data())


if __name__ == "__main__":
    main()
