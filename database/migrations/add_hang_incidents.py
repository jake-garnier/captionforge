"""
Migration: Add hang incidents table for tracking worker hangs
Run this to create the hang_incidents table for debugging and pattern analysis
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, JSON, Float, Boolean, ForeignKey
from sqlalchemy.sql import func
from database.models import Base
from config.settings import settings


def upgrade():
    """Add hang_incidents table"""
    from sqlalchemy import Table, MetaData

    engine = create_engine(settings.database_url)
    metadata = MetaData()

    # Create hang_incidents table
    hang_incidents = Table(
        'hang_incidents',
        metadata,
        Column('id', Integer, primary_key=True, index=True),

        # When the hang was detected
        Column('detected_at', TIMESTAMP(timezone=True), server_default=func.now(), index=True),

        # Task information
        Column('task_name', String(200), nullable=True),
        Column('task_id', String(100), nullable=True),
        Column('video_id', Integer, ForeignKey("videos.id"), nullable=True),

        # Heartbeat info at time of hang
        Column('last_heartbeat_stage', String(100), nullable=True),
        Column('last_heartbeat_at', TIMESTAMP(timezone=True), nullable=True),
        Column('heartbeat_age_seconds', Float, nullable=True),

        # Task duration
        Column('task_duration_seconds', Float, nullable=True),

        # Video metadata (for pattern analysis)
        Column('video_file_size_mb', Float, nullable=True),
        Column('video_duration_seconds', Integer, nullable=True),
        Column('video_resolution', String(50), nullable=True),
        Column('video_subreddit', String(100), nullable=True, index=True),

        # System state at hang
        Column('gpu_memory_used_mb', Integer, nullable=True),
        Column('gpu_memory_total_mb', Integer, nullable=True),
        Column('gpu_utilization_percent', Integer, nullable=True),
        Column('gpu_processes', JSON, nullable=True),
        Column('system_memory_used_mb', Integer, nullable=True),
        Column('system_memory_percent', Float, nullable=True),
        Column('cpu_percent', Float, nullable=True),

        # Worker process info
        Column('worker_pid', Integer, nullable=True),
        Column('worker_memory_mb', Float, nullable=True),
        Column('worker_threads', Integer, nullable=True),
        Column('worker_open_files', Integer, nullable=True),

        # Full diagnostics dump (JSON for flexibility)
        Column('full_diagnostics', JSON, nullable=True),

        # Recovery action taken
        Column('recovery_action', String(50), nullable=True),
        Column('recovery_successful', Boolean, nullable=True),

        # Optional notes
        Column('notes', Text, nullable=True),
    )

    metadata.create_all(engine)
    print("Created hang_incidents table")


def downgrade():
    """Remove hang_incidents table"""
    from sqlalchemy import Table, MetaData

    engine = create_engine(settings.database_url)
    metadata = MetaData()

    hang_incidents = Table('hang_incidents', metadata)
    hang_incidents.drop(engine)
    print("Dropped hang_incidents table")


if __name__ == "__main__":
    print("Running migration: add_hang_incidents")
    upgrade()
