"""
Migration: Update trained model adapter_path from /app/training/models to /data/training_models

This fixes the volume mount conflict where bind mount ./:/app was overriding the
named volume mount for training models. Moving to /data/training_models avoids
the conflict entirely.
"""

import logging
from database.db import get_db_context
from database.models import TrainedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Update all trained model adapter_path entries to new location."""
    with get_db_context() as db:
        # Find all models with old path
        models = db.query(TrainedModel).filter(
            TrainedModel.adapter_path.like('/app/training/models/%')
        ).all()

        if not models:
            logger.info("No models found with old path, migration not needed")
            return

        updated = 0
        for model in models:
            old_path = model.adapter_path
            new_path = old_path.replace('/app/training/models/', '/data/training_models/')

            logger.info(f"Updating model {model.id} ({model.name}): {old_path} -> {new_path}")
            model.adapter_path = new_path
            updated += 1

        db.commit()
        logger.info(f"Successfully updated {updated} model paths")


if __name__ == "__main__":
    migrate()
