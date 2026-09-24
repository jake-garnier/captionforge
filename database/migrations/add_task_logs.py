"""
Migration: Add task execution logs table
Run this to create the task_logs table for storing task execution history
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, JSON
from sqlalchemy.sql import func
from database.models import Base
from config.settings import settings


def upgrade():
    """Add task_logs table"""
    from sqlalchemy import Table, MetaData

    engine = create_engine(settings.database_url)
    metadata = MetaData()

    # Create task_logs table
    task_logs = Table(
        'task_logs',
        metadata,
        Column('id', Integer, primary_key=True, index=True),
        Column('task_name', String(200), nullable=False, index=True),
        Column('celery_task_name', String(200), nullable=False),
        Column('task_id', String(100), nullable=False, index=True),
        Column('execution_number', Integer, nullable=True),
        Column('timestamp', TIMESTAMP(timezone=True), server_default=func.now(), index=True),
        Column('level', String(20), nullable=False),
        Column('message', Text, nullable=False),
        Column('extra_data', JSON, nullable=True),
        Column('created_at', TIMESTAMP(timezone=True), server_default=func.now()),
    )

    metadata.create_all(engine)
    print("✅ Created task_logs table")


def downgrade():
    """Remove task_logs table"""
    from sqlalchemy import Table, MetaData

    engine = create_engine(settings.database_url)
    metadata = MetaData()

    task_logs = Table('task_logs', metadata)
    task_logs.drop(engine)
    print("✅ Dropped task_logs table")


if __name__ == "__main__":
    print("Running migration: add_task_logs")
    upgrade()
