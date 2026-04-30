A short note about the SDET brain corpus that has no YAML frontmatter
and is comfortably below the chunker's 800-character soft target. The
purpose of this fixture is to assert that a single tiny document still
produces exactly one chunk and that the heading path stays empty when
the file has no headings at all.

There is one paragraph of regular prose followed by another short
paragraph so the chunker has to flush at the blank line boundary
without splitting anything. No code, no tables, just text.
