"""
Clean node for LangGraph - Cleans unstructured guidelines using LLM.
Converts messy, formatted content into plain natural language while preserving all context.
"""
import logging
from typing import Dict, Any, List, Optional

from schemas import CleanedGuidelineResult
from prompts import GUIDELINE_CLEANING_PROMPT_TEMPLATE
from utils import create_llm, retry_with_max_attempts, RetryError

logger = logging.getLogger(__name__)

# LlamaIndex imports with fallback
try:
    from llama_index.core.program import LLMTextCompletionProgram
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.error("LlamaIndex not available. Install: pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False
    LLMTextCompletionProgram = None


def _clean_guideline_with_llm(
    content: str,
    source_title: str = "",
    retry_delay: float = 2.0
) -> str:
    """
    Clean unstructured guideline content using LLM.
    Converts formatted, messy content into plain natural language while preserving all context.
    
    Args:
        content: Raw unstructured guideline content to clean
        source_title: Title of the source (for logging)
        retry_delay: Initial delay between retries in seconds
        
    Returns:
        Cleaned plain text guideline
    """
    if not content or not content.strip():
        return ""
    
    if not LLAMAINDEX_AVAILABLE:
        logger.warning("LlamaIndex not available, returning original content")
        return content
    
    llm = create_llm()
    if not llm:
        logger.warning("LLM not available, returning original content")
        return content
    
    from utils import _get_token_count
    total_tokens = _get_token_count(content)
    logger.info(f"Cleaning content from '{source_title[:50]}...' (total: {total_tokens} tokens, {len(content)} chars)")
    
    def _process_content():
        # Format prompt - process entire content at once to preserve all context
        prompt = GUIDELINE_CLEANING_PROMPT_TEMPLATE.format(
            content=content
        )
        
        # Use LlamaIndex for structured extraction
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=CleanedGuidelineResult,
            llm=llm,
            prompt_template_str="{input_str}",
            verbose=False
        )
        
        result = program(input_str=prompt)
        if not result:
            return None
        return result.cleaned_text
    
    try:
        logger.debug(f"Cleaning content from '{source_title[:50]}...'")
        cleaned_text = retry_with_max_attempts(
            operation=_process_content,
            operation_name=f"LLM cleaning for '{source_title[:50]}...'",
            max_attempts=5,
            retry_delay=retry_delay,
            should_retry_on_empty=True
        )
        
        if cleaned_text and cleaned_text.strip():
            logger.info(f"Cleaned content: {len(cleaned_text)} chars (original: {len(content)} chars)")
            return cleaned_text.strip()
        else:
            logger.warning(f"Empty cleaned text, using original content")
            return content.strip()
            
    except RetryError as e:
        logger.error(f"Cleaning failed after max attempts: {e}")
        # Fallback to original content if cleaning fails
        return content.strip()


def clean_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean node that processes unstructured guidelines from search results.
    Converts formatted, messy content into plain natural language while preserving all context.
    
    Args:
        state: Pipeline state dictionary (should have 'search_results' from search_node)
        
    Returns:
        Updated state with cleaned search results
    """
    try:
        if not LLAMAINDEX_AVAILABLE:
            logger.warning("LlamaIndex not available. Skipping cleaning, using original content.")
            return {}
        
        search_results = state.get('search_results', [])
        
        if not search_results:
            logger.warning("No search results to clean")
            return {}
        
        # Get progress callback from metadata if available
        metadata = state.get('metadata', {})
        progress_callback = metadata.get('progress_callback')
        
        # Clean each search result's content
        cleaned_results = []
        total_results = len(search_results)
        
        for idx, result in enumerate(search_results):
            source_title = result.get('title', 'Unknown')
            source_url = result.get('url', '')
            snippet = result.get('snippet', '')
            
            if not snippet or not snippet.strip():
                logger.debug(f"Skipping empty snippet for '{source_title}'")
                cleaned_results.append(result)
                continue
            
            try:
                logger.info(f"Cleaning guideline from '{source_title[:50]}...'")
                
                # Clean the snippet content
                cleaned_snippet = _clean_guideline_with_llm(
                    content=snippet,
                    source_title=source_title
                )
                
                # Update result with cleaned content
                cleaned_result = result.copy()
                cleaned_result['snippet'] = cleaned_snippet
                cleaned_result['original_snippet'] = snippet  # Preserve original for reference
                cleaned_results.append(cleaned_result)
                
                logger.info(f"✓ Cleaned '{source_title[:50]}...' ({len(cleaned_snippet)} chars)")
                
            except Exception as e:
                logger.error(f"Error cleaning '{source_title}': {e}", exc_info=True)
                # On error, keep original content
                cleaned_results.append(result)
                continue
            
            # Emit progress for this result
            if progress_callback and total_results > 0:
                sub_progress = ((idx + 1) / total_results) * 100.0
                progress_callback('clean', sub_progress, f'Cleaned {idx + 1}/{total_results} guidelines')
        
        logger.info(f"Cleaned {len(cleaned_results)} search results")
        
        return {
            'search_results': cleaned_results
        }
        
    except RetryError as e:
        error_msg = f"Clean node failed: {e.operation_name} - {str(e)}"
        logger.error(error_msg)
        # Record error in state and stop pipeline
        errors = state.get('errors', [])
        errors.append(error_msg)
        raise Exception(error_msg) from e
    except Exception as e:
        logger.error(f"Error in clean_node: {e}", exc_info=True)
        # Record error in state
        errors = state.get('errors', [])
        errors.append(f"Clean node error: {str(e)}")
        raise
