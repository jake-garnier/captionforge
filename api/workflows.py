"""
Workflows API - Browser automation workflow management.

Provides endpoints for:
- Managing VNC browser sessions
- Recording user actions
- Saving/loading workflows
- Executing workflows with live progress
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import Workflow
from utils.workflow_session import workflow_session_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Workflows"])


# =============================================================================
# Pydantic Models
# =============================================================================

class SessionStartRequest(BaseModel):
    start_url: str = "about:blank"


class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    target_site: Optional[str] = None
    actions: List[Dict[str, Any]]
    variables: Optional[List[Dict[str, Any]]] = None


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_site: Optional[str] = None
    actions: Optional[List[Dict[str, Any]]] = None
    variables: Optional[List[Dict[str, Any]]] = None


class ExecuteRequest(BaseModel):
    variables: Optional[Dict[str, str]] = None
    speed_multiplier: float = 1.0


class ExecuteActionsRequest(BaseModel):
    actions: List[Dict[str, Any]]
    variables: Optional[Dict[str, str]] = None
    speed_multiplier: float = 1.0


# =============================================================================
# Session Management
# =============================================================================

@router.post("/session/start")
async def start_session(request: SessionStartRequest):
    """Start a browser session with VNC display."""
    result = await workflow_session_manager.start_session(request.start_url)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


@router.post("/session/stop")
async def stop_session():
    """Stop the browser session."""
    result = await workflow_session_manager.stop_session()
    return result


@router.get("/session/status")
async def get_session_status():
    """Get current session status."""
    return workflow_session_manager.get_status()


@router.get("/session/screenshot")
async def get_screenshot():
    """Get the latest screenshot as base64."""
    return workflow_session_manager.get_latest_screenshot()


@router.post("/session/screenshot")
async def take_screenshot():
    """Take a new screenshot."""
    if not workflow_session_manager.page:
        raise HTTPException(status_code=400, detail="No active session")
    path = await workflow_session_manager._take_screenshot("manual")
    if path:
        return {"status": "success", "path": path}
    raise HTTPException(status_code=500, detail="Failed to take screenshot")


@router.post("/session/navigate")
async def navigate(url: str):
    """Navigate to a URL."""
    result = await workflow_session_manager.navigate(url)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


# =============================================================================
# Recording
# =============================================================================

@router.post("/recording/start")
async def start_recording():
    """Start recording user actions."""
    result = await workflow_session_manager.start_recording()
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


@router.post("/recording/stop")
async def stop_recording():
    """Stop recording and return captured actions."""
    result = await workflow_session_manager.stop_recording()
    return result


@router.get("/recording/actions")
async def get_recorded_actions():
    """Get currently recorded actions (poll during recording)."""
    actions = await workflow_session_manager.get_recorded_actions()
    return {"actions": actions, "count": len(actions)}


@router.get("/recording/status")
async def get_recording_status():
    """Get recording status."""
    status = workflow_session_manager.get_status()
    return {
        "recording_active": status.get("recording_active", False),
        "action_count": status.get("recorded_action_count", 0),
    }


# =============================================================================
# Workflow CRUD
# =============================================================================

@router.get("/")
async def list_workflows(
    target_site: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all saved workflows."""
    query = db.query(Workflow)
    if target_site:
        query = query.filter(Workflow.target_site == target_site)
    workflows = query.order_by(Workflow.updated_at.desc()).all()

    return [
        {
            "id": w.id,
            "name": w.name,
            "description": w.description,
            "target_site": w.target_site,
            "action_count": len(w.actions or []),
            "variable_count": len(w.variables or []),
            "run_count": w.run_count,
            "last_run_at": w.last_run_at.isoformat() if w.last_run_at else None,
            "last_run_status": w.last_run_status,
            "created_at": w.created_at.isoformat() if w.created_at else None,
            "updated_at": w.updated_at.isoformat() if w.updated_at else None,
        }
        for w in workflows
    ]


@router.post("/")
async def create_workflow(
    request: WorkflowCreate,
    db: Session = Depends(get_db)
):
    """Create a new workflow."""
    # Check for duplicate name
    existing = db.query(Workflow).filter(Workflow.name == request.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Workflow '{request.name}' already exists")

    workflow = Workflow(
        name=request.name,
        description=request.description,
        target_site=request.target_site,
        actions=request.actions,
        variables=request.variables or [],
    )
    db.add(workflow)
    db.commit()
    db.refresh(workflow)

    logger.info(f"Created workflow: {workflow.name} ({len(workflow.actions)} actions)")
    return {
        "id": workflow.id,
        "name": workflow.name,
        "action_count": len(workflow.actions),
    }


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Get a workflow by ID."""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "target_site": workflow.target_site,
        "actions": workflow.actions,
        "variables": workflow.variables,
        "run_count": workflow.run_count,
        "last_run_at": workflow.last_run_at.isoformat() if workflow.last_run_at else None,
        "last_run_status": workflow.last_run_status,
        "last_run_error": workflow.last_run_error,
        "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
        "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None,
    }


@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: int,
    request: WorkflowUpdate,
    db: Session = Depends(get_db)
):
    """Update a workflow."""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if request.name is not None:
        # Check for duplicate name
        existing = db.query(Workflow).filter(
            Workflow.name == request.name,
            Workflow.id != workflow_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Workflow '{request.name}' already exists")
        workflow.name = request.name

    if request.description is not None:
        workflow.description = request.description
    if request.target_site is not None:
        workflow.target_site = request.target_site
    if request.actions is not None:
        workflow.actions = request.actions
    if request.variables is not None:
        workflow.variables = request.variables

    db.commit()
    logger.info(f"Updated workflow: {workflow.name}")
    return {"status": "updated", "id": workflow.id}


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Delete a workflow."""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    name = workflow.name
    db.delete(workflow)
    db.commit()
    logger.info(f"Deleted workflow: {name}")
    return {"status": "deleted", "name": name}


# =============================================================================
# Execution
# =============================================================================

@router.post("/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: int,
    request: ExecuteRequest,
    db: Session = Depends(get_db)
):
    """Execute a saved workflow."""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if not workflow_session_manager.page:
        raise HTTPException(status_code=400, detail="No active session - start a session first")

    # Execute workflow
    result = await workflow_session_manager.execute_workflow(
        actions=workflow.actions,
        variables=request.variables,
        speed_multiplier=request.speed_multiplier,
    )

    # Update workflow stats
    workflow.run_count = (workflow.run_count or 0) + 1
    workflow.last_run_at = datetime.utcnow()
    workflow.last_run_status = result.get("status")
    if result.get("status") == "error":
        workflow.last_run_error = result.get("error")
    else:
        workflow.last_run_error = None
    db.commit()

    return result


@router.post("/execute-actions")
async def execute_actions(request: ExecuteActionsRequest):
    """Execute actions directly (without saving as workflow)."""
    if not workflow_session_manager.page:
        raise HTTPException(status_code=400, detail="No active session - start a session first")

    result = await workflow_session_manager.execute_workflow(
        actions=request.actions,
        variables=request.variables,
        speed_multiplier=request.speed_multiplier,
    )
    return result


@router.get("/execution/status")
async def get_execution_status():
    """Get current execution status for frontend highlighting."""
    return workflow_session_manager.get_execution_status()


@router.post("/execution/cancel")
async def cancel_execution():
    """Cancel ongoing execution."""
    result = await workflow_session_manager.cancel_execution()
    return result


# =============================================================================
# WebSocket for Live Execution Updates
# =============================================================================

@router.websocket("/ws/execution")
async def execution_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for live execution updates.

    Sends updates whenever execution step changes:
    {
        "executing": true,
        "current_step": 3,
        "total_steps": 10,
        "status": "running"
    }
    """
    await websocket.accept()
    logger.info("[WORKFLOW WS] Client connected")

    queue = workflow_session_manager.subscribe_to_execution()

    try:
        # Send initial status
        await websocket.send_json(workflow_session_manager.get_execution_status())

        # Listen for updates
        while True:
            try:
                # Wait for update with timeout
                update = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(update)
            except asyncio.TimeoutError:
                # Send heartbeat/ping
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        logger.info("[WORKFLOW WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WORKFLOW WS] Error: {e}")
    finally:
        workflow_session_manager.unsubscribe_from_execution(queue)


# =============================================================================
# Variable Detection
# =============================================================================

@router.post("/detect-variables")
async def detect_variables(actions: List[Dict[str, Any]]):
    """
    Analyze actions and detect potential variables.

    Returns typed text that could be parameterized.
    """
    potential_variables = []

    for i, action in enumerate(actions):
        if action.get("type") == "type":
            text = action.get("text", "")
            if len(text) >= 3:  # Only suggest variables for non-trivial text
                potential_variables.append({
                    "action_index": i,
                    "text": text,
                    "suggested_name": f"{{{{text_{i}}}}}",
                    "selector": action.get("selector"),
                })

    return {"potential_variables": potential_variables}
