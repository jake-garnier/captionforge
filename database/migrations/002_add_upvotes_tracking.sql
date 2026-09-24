-- Migration: Add upvotes and last_upvote_check columns to videos table
-- Date: 2025-11-17
-- Description: Adds upvote tracking functionality for Reddit posts

-- Add upvotes column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'videos' AND column_name = 'upvotes'
    ) THEN
        ALTER TABLE videos ADD COLUMN upvotes INTEGER DEFAULT 0;
        CREATE INDEX IF NOT EXISTS idx_videos_upvotes ON videos(upvotes DESC);
    END IF;
END $$;

-- Add last_upvote_check column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'videos' AND column_name = 'last_upvote_check'
    ) THEN
        ALTER TABLE videos ADD COLUMN last_upvote_check TIMESTAMP WITH TIME ZONE;
    END IF;
END $$;

-- Add upvotes column to scraped_captions if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scraped_captions' AND column_name = 'upvotes'
    ) THEN
        ALTER TABLE scraped_captions ADD COLUMN upvotes INTEGER;
    END IF;
END $$;

-- Populate upvotes from scraped_captions if any exist
UPDATE videos v
SET upvotes = sc.upvotes
FROM scraped_captions sc
WHERE v.id = sc.video_id
  AND sc.upvotes IS NOT NULL
  AND v.upvotes IS NULL;

-- Set default upvotes to 0 for any remaining NULL values
UPDATE videos SET upvotes = 0 WHERE upvotes IS NULL;
