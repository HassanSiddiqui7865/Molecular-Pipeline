"""
FastAPI backend API for Molecular Pipeline with Server-Sent Events (SSE) for real-time progress.
"""
import json
import sys
import logging
import threading
import queue
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, Body
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from nodes.search_node import PerplexitySearch
from graph import create_pipeline_graph, run_pipeline
from config import get_perplexity_config, get_output_config
from db_session import init_db_pool, save_session, get_session, list_sessions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Molecular Pipeline API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database connection pool on startup."""
    init_db_pool()

# Progress queue for SSE
progress_queues = {}

# Store input parameters for each session (for database persistence)
session_inputs = {}

# Execution lock to ensure only one pipeline runs at a time
execution_lock = threading.Lock()
current_execution = None  # Track current session_id being executed


def emit_progress(session_id: str, stage: str, message: str, progress: int = None):
    """Emit progress update to SSE stream and save to database."""
    if session_id in progress_queues:
        data = {
            'stage': stage,
            'message': message,
            'progress': progress
        }
        progress_queues[session_id].put(data)
    
    # Save progress to database
    status = 'running'
    if stage == 'complete':
        status = 'completed'
    elif stage == 'error':
        status = 'error'
    
    # Get input parameters if available
    input_params = session_inputs.get(session_id, {})
    
    save_session(
        session_id=session_id,
        input_parameters=input_params,
        status=status,
        progress=progress or 0,
        current_stage=stage
    )


def run_pipeline_with_progress(session_id: str, input_params: dict):
    """Run pipeline and emit progress updates."""
    try:
        # Save initial session to database
        save_session(
            session_id=session_id,
            input_parameters=input_params,
            status='running',
            progress=0,
            current_stage='initializing'
        )
        
        emit_progress(session_id, 'initializing', 'Initializing pipeline...', 0)
        
        # Load configuration
        perplexity_config = get_perplexity_config()
        if not perplexity_config.get('api_key') or perplexity_config.get('api_key') == 'YOUR_PERPLEXITY_API_KEY':
            emit_progress(session_id, 'error', 'Please set your PERPLEXITY_API_KEY in .env file', 0)
            return
        
        perplexity = PerplexitySearch(
            api_key=perplexity_config['api_key'],
            max_tokens=perplexity_config.get('max_tokens', 50000),
            max_tokens_per_page=perplexity_config.get('max_tokens_per_page', 4096)
        )
        
        # Check for cached search results
        from main import get_cache_filename, load_cached_search_results
        cache_path = get_cache_filename(input_params)
        cached_data = load_cached_search_results(cache_path)
        
        # Create pipeline graph
        graph = create_pipeline_graph()
        
        # Run pipeline with progress tracking
        max_search_results = perplexity_config.get('max_search_results', 10)
        
        # Prepare initial state with progress callback
        def progress_callback(stage: str, sub_progress: float, message: str = None):
            """Callback for granular progress updates from nodes."""
            # Calculate overall progress based on stage and sub-progress
            stage_ranges = {
                'icd_transform': (0, 10),
                'search': (10, 20),
                'extract': (20, 40),
                'parse': (40, 45),
                'rank': (45, 55),
                'synthesize': (55, 70),
                'enrichment': (70, 100)
            }
            
            if stage in stage_ranges:
                start, end = stage_ranges[stage]
                overall_progress = int(start + (end - start) * sub_progress / 100.0)
                emit_progress(session_id, stage, message or f'{stage} ({sub_progress:.0f}%)', overall_progress)
        
        initial_state = {
            'input_parameters': input_params,
            'errors': [],
            'rank_memory': {},
            'metadata': {
                'perplexity_client': perplexity,
                'max_search_results': max_search_results,
                'cached_search_results': cached_data,
                'cache_path': str(cache_path),
                'progress_callback': progress_callback
            }
        }
        
        # Run graph - progress will be tracked by nodes through progress_callback
        final_state = graph.invoke(initial_state)
        
        # Nodes report their own progress, so we don't need duplicate updates here
        # Just ensure final completion is reported
        if 'result' in final_state:
            # Final completion will be handled by emit_progress below
            pass
        
        # Prepare output
        output_data = {
            'input_parameters': final_state.get('input_parameters', input_params),
            'extraction_date': final_state.get('extraction_date'),
            'result': final_state.get('result', {}),
            'icd_transformation': final_state.get('icd_transformation', {}),
            'negative_organisms': (
                final_state.get('metadata', {}).get('negative_organisms') or
                final_state.get('result', {}).get('negative_organisms') or
                []
            ),
            'negative_resistance_genes': (
                final_state.get('metadata', {}).get('negative_resistance_genes') or
                final_state.get('result', {}).get('negative_resistance_genes') or
                []
            )
        }
        
        emit_progress(session_id, 'complete', 'Pipeline completed successfully!', 100)
        
        # Save final result to database
        save_session(
            session_id=session_id,
            input_parameters=input_params,
            status='completed',
            progress=100,
            current_stage='complete',
            result=output_data
        )
        
        # Store result
        progress_queues[session_id].put({'result': output_data, 'stage': 'complete'})
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        error_msg = str(e)
        emit_progress(session_id, 'error', f'Error: {error_msg}', 0)
        
        # Save error to database
        save_session(
            session_id=session_id,
            input_parameters=input_params,
            status='error',
            progress=0,
            current_stage='error',
            error_message=error_msg
        )
        
        progress_queues[session_id].put({'error': error_msg, 'stage': 'error'})


# Pydantic models for request/response
class PipelineInput(BaseModel):
    pathogens: list
    resistant_genes: list
    severity_codes: list
    age: Optional[int] = None
    panel: Optional[str] = None
    systemic: Optional[bool] = None
    allergy: Optional[List[str]] = None


@app.get('/')
async def index():
    """Serve the frontend."""
    return FileResponse('static/index.html')


@app.post('/api/run')
async def run_pipeline_api(request: Request):
    """Start pipeline execution. Only one execution allowed at a time."""
    global current_execution
    try:
        # Check if there's already a running execution
        with execution_lock:
            if current_execution is not None:
                # Check if the current execution is still running
                if current_execution in progress_queues:
                    # Get session status from database
                    session = get_session(current_execution)
                    if session and session.get('status') == 'running':
                        return JSONResponse(
                            status_code=409,
                            content={'error': 'Pipeline is already running. Please wait for the current execution to complete.', 'current_session_id': current_execution}
                        )
                    else:
                        # Previous execution finished, clean up
                        if current_execution in progress_queues:
                            del progress_queues[current_execution]
                        if current_execution in session_inputs:
                            del session_inputs[current_execution]
                        current_execution = None
        
        input_data = await request.json()
        if not input_data:
            return JSONResponse(
                status_code=400,
                content={'error': 'No JSON data provided'}
            )
        
        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())
        
        # Acquire lock and set current execution
        with execution_lock:
            if current_execution is not None:
                return JSONResponse(
                    status_code=409,
                    content={'error': 'Pipeline is already running. Please wait for the current execution to complete.', 'current_session_id': current_execution}
                )
            current_execution = session_id
        
        # Store input parameters for this session
        session_inputs[session_id] = input_data
        
        # Create progress queue
        progress_queues[session_id] = queue.Queue()
        
        # Start pipeline in background thread
        def run_and_release():
            global current_execution
            try:
                run_pipeline_with_progress(session_id, input_data)
            finally:
                # Release lock when execution completes
                with execution_lock:
                    if current_execution == session_id:
                        current_execution = None
        
        thread = threading.Thread(target=run_and_release)
        thread.daemon = True
        thread.start()
        
        return JSONResponse(content={'session_id': session_id})
        
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}", exc_info=True)
        # Release lock on error (if session_id was created)
        with execution_lock:
            if 'session_id' in locals() and current_execution == session_id:
                current_execution = None
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.get('/api/sessions')
async def get_sessions_list(limit: int = 50, status: Optional[str] = None):
    """Get list of recent sessions."""
    from datetime import datetime
    sessions = list_sessions(limit=limit, status=status)
    
    # Convert datetime objects to ISO format strings for JSON serialization
    serializable_sessions = []
    for session in sessions:
        serializable_session = dict(session)
        # Convert datetime fields to ISO format strings
        for key, value in serializable_session.items():
            if isinstance(value, datetime):
                serializable_session[key] = value.isoformat()
        serializable_sessions.append(serializable_session)
    
    return JSONResponse(content={'sessions': serializable_sessions})


@app.get('/api/sessions/active')
async def get_active_session():
    """Get the currently active/running session if any."""
    from datetime import datetime
    
    def serialize_session(session):
        """Convert datetime objects in session to ISO format strings."""
        if not session:
            return None
        serializable = dict(session)
        for key, value in serializable.items():
            if isinstance(value, datetime):
                serializable[key] = value.isoformat()
        return serializable
    
    with execution_lock:
        if current_execution is not None:
            session = get_session(current_execution)
            if session and session.get('status') == 'running':
                return JSONResponse(content={
                    'active': True,
                    'session_id': current_execution,
                    'session': serialize_session(session)
                })
    
    # Check database for any running sessions (in case of server restart)
    running_sessions = list_sessions(limit=1, status='running')
    if running_sessions:
        session = running_sessions[0]
        return JSONResponse(content={
            'active': True,
            'session_id': session['session_id'],
            'session': serialize_session(session)
        })
    
    return JSONResponse(content={'active': False})


@app.get('/api/sessions/{session_id}')
async def get_session_data(session_id: str):
    """Get session data by ID."""
    from datetime import datetime
    session = get_session(session_id)
    if not session:
        return JSONResponse(
            status_code=404,
            content={'error': 'Session not found'}
        )
    
    # Convert datetime objects to ISO format strings for JSON serialization
    serializable_session = dict(session)
    for key, value in serializable_session.items():
        if isinstance(value, datetime):
            serializable_session[key] = value.isoformat()
    
    return JSONResponse(content=serializable_session)


@app.get('/api/progress/{session_id}')
async def progress_stream(session_id: str):
    """SSE endpoint for progress updates."""
    
    async def generate():
        global current_execution
        # If session not in memory queues, check database and create queue if needed
        if session_id not in progress_queues:
            session = get_session(session_id)
            if session and session.get('status') == 'running':
                # Recreate queue for active session
                progress_queues[session_id] = queue.Queue()
                # Send current state from database
                state_data = {
                    'stage': session.get('current_stage', 'running'),
                    'message': f"Resumed: {session.get('current_stage', 'running')}",
                    'progress': session.get('progress', 0)
                }
                yield f"data: {json.dumps(state_data)}\n\n"
            else:
                yield f"data: {json.dumps({'error': 'Invalid or completed session ID'})}\n\n"
                return
        
        q = progress_queues[session_id]
        
        while True:
            try:
                # Wait for progress update using asyncio
                # No timeout - pipeline can take as long as needed
                loop = asyncio.get_event_loop()
                try:
                    # Use run_in_executor to handle blocking queue.get()
                    # Poll every second to check for updates
                    item = await loop.run_in_executor(None, lambda: q.get(timeout=1))
                except queue.Empty:
                    # Continue polling - send keepalive to prevent connection timeout
                    yield f": keepalive\n\n"
                    continue
                
                if 'result' in item or 'error' in item:
                    # Final result or error
                    yield f"data: {json.dumps(item)}\n\n"
                    # Clean up
                    if session_id in progress_queues:
                        del progress_queues[session_id]
                    # Release execution lock
                    try:
                        with execution_lock:
                            if current_execution == session_id:
                                current_execution = None
                    except Exception as lock_error:
                        logger.warning(f"Error releasing execution lock: {lock_error}")
                    break
                else:
                    # Progress update
                    yield f"data: {json.dumps(item)}\n\n"
                    
            except Exception as e:
                logger.error(f"Error in progress stream: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.post('/api/download-pdf')
async def download_pdf_api(request: Request):
    """Generate and download PDF from JSON data or session_id."""
    try:
        data = await request.json()
        session_id = data.get('session_id')
        json_data = data.get('data')
        
        # If session_id provided, fetch from database
        if session_id:
            session = get_session(session_id)
            if not session:
                return JSONResponse(
                    status_code=404,
                    content={'error': 'Session not found'}
                )
            
            # Get result from session
            result_data = session.get('result')
            if not result_data:
                return JSONResponse(
                    status_code=400,
                    content={'error': 'No result data found for this session'}
                )
            
            # Use the stored result data
            pdf_data = result_data
        elif json_data:
            # Use provided JSON data
            pdf_data = json_data
        else:
            return JSONResponse(
                status_code=400,
                content={'error': 'Either session_id or data must be provided'}
            )
        
        # Import export function
        from export_pdf import export_to_pdf
        
        # Generate PDF (optionally saves to disk based on config)
        pdf_name, pdf_buffer = export_to_pdf(pdf_data, save_to_disk=None)
        
        # Create a temporary file-like object for response
        pdf_buffer.seek(0)
        
        # Return PDF file for download in browser
        return StreamingResponse(
            BytesIO(pdf_buffer.read()),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{Path(pdf_name).name}"'
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={'error': f'Error generating PDF: {str(e)}'}
        )


@app.get('/api/download-pdf/{session_id}')
async def download_pdf_by_session(session_id: str):
    """Generate and download PDF from session_id (GET endpoint for direct links)."""
    try:
        session = get_session(session_id)
        if not session:
            return JSONResponse(
                status_code=404,
                content={'error': 'Session not found'}
            )
        
        # Get result from session
        result_data = session.get('result')
        if not result_data:
            return JSONResponse(
                status_code=400,
                content={'error': 'No result data found for this session'}
            )
        
        # Import export function
        from export_pdf import export_to_pdf
        from config import get_output_config
        from io import BytesIO
        
        # Generate PDF (optionally saves to disk based on config)
        pdf_name, pdf_buffer = export_to_pdf(result_data, save_to_disk=None)
        
        # Create a temporary file-like object for response
        pdf_buffer.seek(0)
        
        # Return PDF file for download in browser
        return StreamingResponse(
            BytesIO(pdf_buffer.read()),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{Path(pdf_name).name}"'
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={'error': f'Error generating PDF: {str(e)}'}
        )


if __name__ == '__main__':
    import uvicorn
    import sys
    
    # Check for environment flag: env.dev for development, env for production
    env_mode = None
    if len(sys.argv) > 1:
        if 'env.dev' in sys.argv:
            env_mode = 'dev'
            sys.argv.remove('env.dev')
        elif 'env' in sys.argv:
            env_mode = 'prod'
            sys.argv.remove('env')
    
    # Set ENV environment variable before loading config
    if env_mode:
        import os
        os.environ['ENV'] = env_mode
        logger.info(f"Running in {env_mode} mode")
    
    uvicorn.run(app, host='0.0.0.0', port=7653, log_level='info')

