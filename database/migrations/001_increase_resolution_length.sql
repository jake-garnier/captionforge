-- Migration: Increase resolution column length from 20 to 50
-- Date: 2025-11-15
-- Reason: Some resolutions with floating point precision (e.g., 854.2222222222223x480) exceed 20 characters

-- Alter the resolution column to allow up to 50 characters
ALTER TABLE videos ALTER COLUMN resolution TYPE VARCHAR(50);
