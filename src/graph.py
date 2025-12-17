"""
LangGraph workflow definition for the Molecular Pipeline.
"""
import logging
from typing import Dict, Any, TypedDict, List, Optional

from langgraph.graph import StateGraph, END

from nodes.search_node import search_node
from nodes.extract_node import extract_node
from nodes.parse_node import parse_node
from nodes.rank_node import rank_node
from nodes.synthesize_node import synthesize_node
from nodes.icd_transform_node import icd_transform_node
from nodes.enrichment_node import enrichment_node

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """TypedDict for LangGraph state."""
    input_parameters: Dict[str, str]
    icd_transformation: Optional[Dict[str, Any]]
    search_query: Optional[str]
    search_results: List[Dict[str, Any]]
    source_results: List[Dict[str, Any]]
    extraction_date: Optional[str]
    result: Optional[Dict[str, Any]]
    errors: List[str]
    metadata: Dict[str, Any]


def create_pipeline_graph() -> StateGraph:
    """
    Create and configure the LangGraph workflow.
    
    Returns:
        Configured StateGraph instance
    """
    # Create state graph with TypedDict
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("rank", rank_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("icd_transform", icd_transform_node)
    workflow.add_node("enrichment", enrichment_node)
    
    # Define edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", "parse")
    workflow.add_edge("parse", "rank")  # Rank not_known entries
    workflow.add_edge("rank", "synthesize")  # Then synthesize with clean categories
    workflow.add_edge("synthesize", "icd_transform")  # Transform ICD codes before enrichment
    workflow.add_edge("icd_transform", "enrichment")  # Enrichment uses ICD code names
    workflow.add_edge("enrichment", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app


def run_pipeline(
    graph: StateGraph,
    input_parameters: Dict[str, str],
    perplexity_client: Any,
    max_search_results: int = 5,
    cached_search_results: Optional[Dict] = None,
    cache_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the pipeline with given inputs.
    
    Args:
        graph: Compiled LangGraph instance
        input_parameters: Input parameters dict
        perplexity_client: PerplexitySearch client instance
        max_search_results: Maximum number of search results
        cached_search_results: Cached search results if available
        cache_path: Path to cache file for saving search results
        
    Returns:
        Final state dictionary
    """
    # Prepare initial state
    initial_state = {
        'input_parameters': input_parameters,
        'metadata': {
            'perplexity_client': perplexity_client,
            'max_search_results': max_search_results,
            'cached_search_results': cached_search_results,
            'cache_path': cache_path
        }
    }
    
    # Run graph
    try:
        final_state = graph.invoke(initial_state)
        return final_state
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        raise

