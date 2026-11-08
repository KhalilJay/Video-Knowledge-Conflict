"""Resolve local model directories without downloads or weight inspection."""
from __future__ import annotations

import os
from pathlib import Path

from vconflict_pipeline.settings import (
    COSMOS_NANO_QA_MODEL, COSMOS_SUPER_QA_MODEL, GEMMA_QA_MODEL, QWEN38_QA_MODEL,
)

DEFAULT_MODELS = ('qwen', 'cosmos3-nano', 'gemma')
VCD_MODELS = (*DEFAULT_MODELS, 'cosmos3-super')
BASELINE_MODELS = (*DEFAULT_MODELS, 'cosmos3-super')
MODELS = {'cosmos3-nano': COSMOS_NANO_QA_MODEL, 'cosmos3-super': COSMOS_SUPER_QA_MODEL,
          'gemma': GEMMA_QA_MODEL, 'qwen': QWEN38_QA_MODEL}
REVISIONS = {'cosmos3-nano': 'main', 'cosmos3-super': 'main', 'qwen': 'main',
             'gemma': '842da3794eaa0b77d5f08bae87a17459d91ff475'}


def canonical_model(model):
    return 'cosmos3-nano' if model == 'cosmos' else model


def processing_family(model):
    """Keep model identity separate from the shared Cosmos implementation."""
    model = canonical_model(model)
    return 'cosmos' if model in ('cosmos3-nano', 'cosmos3-super') else model


def resolve_model(family):
    family = canonical_model(family)
    cache = (Path(os.environ['HF_HUB_CACHE']) if os.environ.get('HF_HUB_CACHE') else
             Path(os.environ['HF_HOME']) / 'hub' if os.environ.get('HF_HOME') else
             Path('/cache/huggingface/hub')).expanduser().resolve()
    model, revision = MODELS[family], REVISIONS[family]
    detail = (f'Cannot resolve cached model {model} at revision {revision} in {cache}. '
              'Set HF_HUB_CACHE or HF_HOME to the offline cache; no download is attempted.')
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:
        raise ValueError(detail + ' Offline resolution requires huggingface_hub in the active environment.') from exc
    cached = try_to_load_from_cache(model, 'config.json', revision=revision, cache_dir=cache)
    if not isinstance(cached, str):
        raise ValueError(detail)
    # Resolve the snapshot directory, not the config symlink into the blob store.
    return Path(cached).parent.resolve(strict=True)
