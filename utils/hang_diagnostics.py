"""
Comprehensive diagnostics collector for worker hang analysis.

Collects detailed system state when a hang is detected to help identify
root causes and patterns.
"""
import subprocess
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def collect_hang_diagnostics(worker_status: Dict) -> Dict[str, Any]:
    """
    Collect comprehensive diagnostics when a hang is detected.

    Args:
        worker_status: Current worker status from get_worker_status()

    Returns:
        Dictionary with all collected diagnostics
    """
    diagnostics = {
        'collected_at': datetime.utcnow().isoformat(),
        'worker_status': worker_status,
    }

    # Collect GPU diagnostics
    diagnostics['gpu'] = collect_gpu_diagnostics()

    # Collect system memory/CPU
    diagnostics['system'] = collect_system_diagnostics()

    # Collect worker process info
    heartbeat = worker_status.get('heartbeat', {})
    worker_pid = heartbeat.get('pid') if heartbeat else None
    diagnostics['worker_process'] = collect_process_diagnostics(worker_pid)

    # Collect video info if we know which video was being processed
    current_task = worker_status.get('current_task', {})
    video_id = current_task.get('video_id') if current_task else None
    if video_id:
        diagnostics['video'] = collect_video_diagnostics(video_id)

    return diagnostics


def collect_gpu_diagnostics() -> Dict[str, Any]:
    """Collect GPU state including memory, utilization, and processes."""
    result = {
        'available': False,
        'error': None
    }

    try:
        # Basic GPU info
        gpu_info = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )

        if gpu_info.returncode == 0:
            result['available'] = True
            gpus = []
            for line in gpu_info.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_used_mb': int(parts[2]),
                        'memory_total_mb': int(parts[3]),
                        'utilization_percent': int(parts[4]) if parts[4] != '[N/A]' else None,
                        'temperature_c': int(parts[5]) if parts[5] != '[N/A]' else None
                    })
            result['gpus'] = gpus

            # Get first GPU stats for summary
            if gpus:
                result['memory_used_mb'] = gpus[0]['memory_used_mb']
                result['memory_total_mb'] = gpus[0]['memory_total_mb']
                result['utilization_percent'] = gpus[0]['utilization_percent']

        # Get GPU processes
        processes_info = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,name,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )

        if processes_info.returncode == 0 and processes_info.stdout.strip():
            processes = []
            for line in processes_info.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    processes.append({
                        'pid': int(parts[0]),
                        'name': parts[1],
                        'memory_mb': int(parts[2]) if parts[2] != '[N/A]' else None
                    })
            result['processes'] = processes
        else:
            result['processes'] = []

    except FileNotFoundError:
        result['error'] = 'nvidia-smi not found'
    except subprocess.TimeoutExpired:
        result['error'] = 'nvidia-smi timed out'
    except Exception as e:
        result['error'] = str(e)

    return result


def collect_system_diagnostics() -> Dict[str, Any]:
    """Collect system memory and CPU info."""
    result = {
        'available': False,
        'error': None
    }

    try:
        import psutil

        result['available'] = True

        # Memory
        mem = psutil.virtual_memory()
        result['memory_total_mb'] = mem.total // (1024 * 1024)
        result['memory_used_mb'] = mem.used // (1024 * 1024)
        result['memory_percent'] = mem.percent

        # CPU
        result['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        result['cpu_count'] = psutil.cpu_count()

        # Disk
        disk = psutil.disk_usage('/')
        result['disk_total_gb'] = disk.total // (1024 * 1024 * 1024)
        result['disk_used_gb'] = disk.used // (1024 * 1024 * 1024)
        result['disk_percent'] = disk.percent

        # Load average (Unix only)
        try:
            load = os.getloadavg()
            result['load_avg_1min'] = load[0]
            result['load_avg_5min'] = load[1]
            result['load_avg_15min'] = load[2]
        except (OSError, AttributeError):
            pass

    except ImportError:
        result['error'] = 'psutil not available'
    except Exception as e:
        result['error'] = str(e)

    return result


def collect_process_diagnostics(pid: Optional[int]) -> Dict[str, Any]:
    """Collect information about the worker process."""
    result = {
        'available': False,
        'error': None,
        'pid': pid
    }

    if not pid:
        result['error'] = 'No PID provided'
        return result

    try:
        import psutil

        try:
            proc = psutil.Process(pid)
            result['available'] = True

            # Basic info
            result['name'] = proc.name()
            result['status'] = proc.status()
            result['create_time'] = datetime.fromtimestamp(proc.create_time()).isoformat()

            # Memory
            mem_info = proc.memory_info()
            result['memory_rss_mb'] = mem_info.rss // (1024 * 1024)
            result['memory_vms_mb'] = mem_info.vms // (1024 * 1024)
            result['memory_percent'] = proc.memory_percent()

            # CPU
            result['cpu_percent'] = proc.cpu_percent(interval=0.1)

            # Threads
            result['num_threads'] = proc.num_threads()

            # Open files
            try:
                open_files = proc.open_files()
                result['num_open_files'] = len(open_files)
                # List first 10 open files for debugging
                result['open_files_sample'] = [f.path for f in open_files[:10]]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                result['num_open_files'] = None

            # Connections
            try:
                conns = proc.connections()
                result['num_connections'] = len(conns)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                result['num_connections'] = None

            # Children
            try:
                children = proc.children(recursive=True)
                result['num_children'] = len(children)
                result['children'] = [{'pid': c.pid, 'name': c.name()} for c in children[:5]]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                result['num_children'] = None

        except psutil.NoSuchProcess:
            result['error'] = f'Process {pid} not found'
        except psutil.AccessDenied:
            result['error'] = f'Access denied for process {pid}'

    except ImportError:
        result['error'] = 'psutil not available'
    except Exception as e:
        result['error'] = str(e)

    return result


def collect_video_diagnostics(video_id: int) -> Dict[str, Any]:
    """Collect information about the video being processed."""
    result = {
        'available': False,
        'error': None,
        'video_id': video_id
    }

    try:
        from database.db import get_db_context
        from database.models import Video

        with get_db_context() as db:
            video = db.query(Video).filter_by(id=video_id).first()

            if video:
                result['available'] = True
                result['source_post_id'] = video.source_post_id
                result['source_subreddit'] = video.source_subreddit
                result['storage_path'] = video.storage_path
                result['file_size_bytes'] = video.file_size_bytes
                result['file_size_mb'] = round(video.file_size_bytes / (1024 * 1024), 2) if video.file_size_bytes else None
                result['duration_seconds'] = video.duration_seconds
                result['resolution'] = video.resolution
                result['processing_status'] = video.processing_status
                result['upvotes'] = video.upvotes

                # Check if file exists and get actual file info
                if video.storage_path and os.path.exists(video.storage_path):
                    result['file_exists'] = True
                    stat = os.stat(video.storage_path)
                    result['actual_file_size_bytes'] = stat.st_size
                    result['file_modified_at'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                else:
                    result['file_exists'] = False
            else:
                result['error'] = f'Video {video_id} not found in database'

    except Exception as e:
        result['error'] = str(e)

    return result


def store_hang_incident(
    worker_status: Dict,
    diagnostics: Dict,
    recovery_action: Optional[str] = None,
    recovery_successful: Optional[bool] = None
) -> Optional[int]:
    """
    Store a hang incident in the database for later analysis.

    Args:
        worker_status: Worker status at time of hang
        diagnostics: Full diagnostics collected
        recovery_action: Action taken ('manual', 'auto_restart', 'none')
        recovery_successful: Whether recovery succeeded

    Returns:
        ID of the created HangIncident record, or None if failed
    """
    try:
        from database.db import get_db_context
        from database.models import HangIncident

        # Extract key fields from diagnostics
        current_task = worker_status.get('current_task', {}) or {}
        heartbeat = worker_status.get('heartbeat', {}) or {}
        gpu = diagnostics.get('gpu', {})
        system = diagnostics.get('system', {})
        worker = diagnostics.get('worker_process', {})
        video = diagnostics.get('video', {})

        # Calculate heartbeat age
        heartbeat_age = None
        heartbeat_at = None
        if heartbeat and heartbeat.get('timestamp'):
            try:
                heartbeat_at = datetime.fromisoformat(heartbeat['timestamp'])
                heartbeat_age = (datetime.utcnow() - heartbeat_at).total_seconds()
            except (ValueError, TypeError):
                pass

        with get_db_context() as db:
            incident = HangIncident(
                # Task info
                task_name=current_task.get('task_name'),
                task_id=current_task.get('task_id'),
                video_id=current_task.get('video_id'),

                # Heartbeat info
                last_heartbeat_stage=heartbeat.get('stage'),
                last_heartbeat_at=heartbeat_at,
                heartbeat_age_seconds=heartbeat_age,

                # Task duration
                task_duration_seconds=worker_status.get('hang_duration_seconds'),

                # Video metadata
                video_file_size_mb=video.get('file_size_mb'),
                video_duration_seconds=video.get('duration_seconds'),
                video_resolution=video.get('resolution'),
                video_subreddit=video.get('source_subreddit'),

                # GPU state
                gpu_memory_used_mb=gpu.get('memory_used_mb'),
                gpu_memory_total_mb=gpu.get('memory_total_mb'),
                gpu_utilization_percent=gpu.get('utilization_percent'),
                gpu_processes=gpu.get('processes'),

                # System state
                system_memory_used_mb=system.get('memory_used_mb'),
                system_memory_percent=system.get('memory_percent'),
                cpu_percent=system.get('cpu_percent'),

                # Worker process
                worker_pid=worker.get('pid'),
                worker_memory_mb=worker.get('memory_rss_mb'),
                worker_threads=worker.get('num_threads'),
                worker_open_files=worker.get('num_open_files'),

                # Full dump for reference
                full_diagnostics=diagnostics,

                # Recovery info
                recovery_action=recovery_action,
                recovery_successful=recovery_successful,
            )

            db.add(incident)
            db.commit()
            db.refresh(incident)

            logger.info(f"Stored hang incident #{incident.id}")
            return incident.id

    except Exception as e:
        logger.error(f"Failed to store hang incident: {e}")
        return None


def format_diagnostics_for_logging(diagnostics: Dict) -> str:
    """Format diagnostics as a readable log message."""
    lines = []
    lines.append("=" * 60)
    lines.append("HANG DIAGNOSTICS REPORT")
    lines.append("=" * 60)

    # GPU
    gpu = diagnostics.get('gpu', {})
    if gpu.get('available'):
        lines.append(f"\nGPU:")
        lines.append(f"  Memory: {gpu.get('memory_used_mb', '?')} / {gpu.get('memory_total_mb', '?')} MB")
        lines.append(f"  Utilization: {gpu.get('utilization_percent', '?')}%")
        if gpu.get('gpus'):
            for g in gpu['gpus']:
                lines.append(f"  GPU {g['index']}: {g['name']} - {g.get('temperature_c', '?')}C")
        if gpu.get('processes'):
            lines.append(f"  Processes using GPU:")
            for p in gpu['processes']:
                lines.append(f"    PID {p['pid']}: {p['name']} ({p.get('memory_mb', '?')} MB)")
    else:
        lines.append(f"\nGPU: Not available ({gpu.get('error', 'unknown')})")

    # System
    system = diagnostics.get('system', {})
    if system.get('available'):
        lines.append(f"\nSystem:")
        lines.append(f"  Memory: {system.get('memory_used_mb', '?')} / {system.get('memory_total_mb', '?')} MB ({system.get('memory_percent', '?')}%)")
        lines.append(f"  CPU: {system.get('cpu_percent', '?')}%")
        lines.append(f"  Disk: {system.get('disk_used_gb', '?')} / {system.get('disk_total_gb', '?')} GB ({system.get('disk_percent', '?')}%)")
        if system.get('load_avg_1min'):
            lines.append(f"  Load: {system['load_avg_1min']:.2f} / {system['load_avg_5min']:.2f} / {system['load_avg_15min']:.2f}")

    # Worker process
    worker = diagnostics.get('worker_process', {})
    if worker.get('available'):
        lines.append(f"\nWorker Process (PID {worker.get('pid', '?')}):")
        lines.append(f"  Status: {worker.get('status', '?')}")
        lines.append(f"  Memory: {worker.get('memory_rss_mb', '?')} MB RSS")
        lines.append(f"  Threads: {worker.get('num_threads', '?')}")
        lines.append(f"  Open files: {worker.get('num_open_files', '?')}")
        if worker.get('open_files_sample'):
            lines.append(f"  Recent files: {', '.join(worker['open_files_sample'][:3])}")

    # Video
    video = diagnostics.get('video', {})
    if video.get('available'):
        lines.append(f"\nVideo being processed:")
        lines.append(f"  ID: {video.get('video_id')}")
        lines.append(f"  Subreddit: r/{video.get('source_subreddit', '?')}")
        lines.append(f"  Post ID: {video.get('source_post_id', '?')}")
        lines.append(f"  Size: {video.get('file_size_mb', '?')} MB")
        lines.append(f"  Duration: {video.get('duration_seconds', '?')} seconds")
        lines.append(f"  Resolution: {video.get('resolution', '?')}")
        lines.append(f"  File exists: {video.get('file_exists', '?')}")

    lines.append("=" * 60)
    return '\n'.join(lines)
