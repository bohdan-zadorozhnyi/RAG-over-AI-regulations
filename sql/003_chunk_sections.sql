-- Every section a chunk's text overlaps.
--
-- chunks.section_id is the section a chunk starts in. A recursive-baseline
-- chunk can straddle several sections (and articles), and the eval credits it
-- with every article it overlaps: maximally generous to the baseline, so any
-- remaining gap is due to chunking, not to how articles were assigned.
-- Section-aware chunks always have exactly one row here.
CREATE TABLE IF NOT EXISTS chunk_sections (
    chunk_id   BIGINT NOT NULL REFERENCES chunks(id)   ON DELETE CASCADE,
    section_id BIGINT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, section_id)
);

CREATE INDEX IF NOT EXISTS chunk_sections_section_idx ON chunk_sections (section_id);
