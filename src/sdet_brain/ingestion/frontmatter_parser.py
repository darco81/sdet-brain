"""YAML frontmatter extraction.

`python-frontmatter` is a thin wrapper around PyYAML. We use it for
parsing only - the body it returns is exactly the original file minus
the YAML header.
"""

from __future__ import annotations

import logging

import frontmatter
import yaml

logger = logging.getLogger(__name__)


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Return ``(metadata_dict, body_without_frontmatter)``.

    A document without a YAML header yields ``({}, text)``. A document
    with a malformed header is treated as having no header (the raw
    text becomes the body) and the YAML error is logged once.
    """
    try:
        post = frontmatter.loads(text)
    except yaml.YAMLError as exc:
        logger.warning("Malformed YAML frontmatter, treating file as bodyless: %s", exc)
        return {}, text
    metadata = dict(post.metadata)
    return metadata, post.content
