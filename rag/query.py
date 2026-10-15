"""
RAG Query Handler.
Handles queries against the knowledge base with context-aware responses.
"""

from typing import List, Dict, Optional

from rag.index import vector_index
from services.openai_client import openai_client
from utils.logging import logger
from config import RAG_TOP_K


async def query_knowledge_base(
    query: str,
    conversation_history: Optional[List[Dict]] = None,
    allow_image_generation: bool = False,
):
    """
    Query the knowledge base and generate response.
    
    Args:
        query: User's query
        conversation_history: Previous conversation messages
    
    Returns:
        Generated response based on retrieved context
    """
    try:
        # Search for relevant documents
        logger.debug(f"Searching knowledge base for: {query}")
        results = vector_index.similarity_search_with_score(query, k=RAG_TOP_K)
        
        if not results:
            logger.warning("No relevant documents found, using fallback")
            return await _fallback_response(query, conversation_history, allow_image_generation)
        
        # Prepare context from retrieved documents
        context = _prepare_context(results)
        
        # Generate response with context
        response = await _generate_rag_response(
            query=query,
            context=context,
            conversation_history=conversation_history,
            allow_image_generation=allow_image_generation,
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        # Fallback to regular GPT response
        return await _fallback_response(query, conversation_history, allow_image_generation)


def _prepare_context(results: List[tuple]) -> str:
    """
    Prepare context from search results.
    
    Args:
        results: List of (document, score) tuples
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.strip()
        
        context_parts.append(
            f"[Источник {i}: {source}]\n{content}\n"
        )
    
    return "\n".join(context_parts)


async def _generate_rag_response(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None,
    allow_image_generation: bool = False,
):
    """
    Generate response using RAG context.
    
    Args:
        query: User's query
        context: Retrieved context from knowledge base
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    # Build prompt with context
    system_prompt = """Ты - личный ассистент с доступом к базе знаний.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай на основе предоставленного контекста
2. Если контекст содержит ответ - используй его
3. Если контекст не содержит ответа - честно скажи об этом
4. Всегда указывай источники информации
5. Отвечай на русском языке четко и структурированно

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

Используй этот контекст для ответа на вопрос пользователя."""
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": system_prompt.format(context=context)
        }
    ]
    
    # Add conversation history if available
    if conversation_history:
        # Limit history to avoid token limits
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        messages.extend(recent_history)
    
    # Add current query
    messages.append({
        "role": "user",
        "content": query
    })
    
    # Generate response
    if allow_image_generation:
        return await openai_client.generate_reply_or_image(messages)
    response = await openai_client.generate_text_response(messages)
    
    return response


async def _fallback_response(
    query: str,
    conversation_history: Optional[List[Dict]] = None,
    allow_image_generation: bool = False,
):
    """
    Fallback to regular GPT response when RAG fails.
    
    Args:
        query: User's query
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    logger.info("Using fallback response (no RAG context)")
    
    system_message = {
        "role": "system",
        "content": """Ты - личный ассистент. 
        
База знаний пока пуста или не содержит информации по этому вопросу.
Ответь на основе своих общих знаний, но предупреди пользователя, 
что это не основано на специфической базе знаний."""
    }
    
    messages = [system_message]
    
    if conversation_history:
        messages.extend(conversation_history[-6:])
    
    messages.append({
        "role": "user",
        "content": query
    })
    
    if allow_image_generation:
        decision = await openai_client.generate_reply_or_image(messages)
        if "text" in decision:
            decision["text"] = f"⚠️ База знаний не содержит информации по этому вопросу.\n\n{decision['text']}"
        return decision
    response = await openai_client.generate_text_response(messages)
    
    return f"⚠️ База знаний не содержит информации по этому вопросу.\n\n{response}"


async def add_document_to_knowledge_base(file_path: str) -> dict:
    """
    Add a document to the knowledge base.
    
    Args:
        file_path: Path to document file
    
    Returns:
        Dictionary with status and details
    """
    try:
        from pathlib import Path
        
        file_path = Path(file_path)
        chunk_count = vector_index.index_document(file_path)
        skipped = vector_index.last_indexing_summary["skipped"] > 0
        logger.info("Processed %s: %s new or updated chunks", file_path.name, chunk_count)
        
        return {
            "success": True,
            "file": file_path.name,
            "chunks": chunk_count,
            "status": "unchanged" if skipped else "indexed",
            "message": (
                f"Документ {file_path.name} уже проиндексирован, изменений нет."
                if skipped else f"Документ {file_path.name} обработан ({chunk_count} новых или обновлённых фрагментов)."
            ),
        }
        
    except Exception as e:
        logger.error(f"Error adding document to knowledge base: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Ошибка при добавлении документа: {e}"
        }


def get_knowledge_base_stats() -> dict:
    """
    Get statistics about the knowledge base.
    
    Returns:
        Dictionary with statistics
    """
    return vector_index.get_stats()

