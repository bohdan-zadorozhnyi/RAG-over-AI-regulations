-- Step 7: what the answer step returned, beside what retrieval did.
--
-- status is the model's verdict on the retrieved sources: answer (they answer
-- the question), scoped (they answer part of it; the rest is named), refuse.
-- citations are provisions ("32016R0679:32") traced to retrieved chunks; a
-- source number the model invented is dropped, never stored. Nullable: rows
-- from before answer generation have none of these.
ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS status        TEXT CHECK (status IN ('answer', 'scoped', 'refuse')),
    ADD COLUMN IF NOT EXISTS citations     TEXT[],
    ADD COLUMN IF NOT EXISTS model         TEXT,
    ADD COLUMN IF NOT EXISTS input_tokens  INT,
    ADD COLUMN IF NOT EXISTS output_tokens INT;
