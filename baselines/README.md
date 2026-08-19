# Baselines

The systems Matryoshka is compared against in Section 7 of the paper. They are
kept in the repository so the evaluation can be reproduced, but they are
deliberately **outside the installed `matryoshka` distribution**: between them
they require Neo4j, PyTorch, AutoGluon, Elasticsearch, an OpenAI-compatible LLM
client and several pinned scientific packages, none of which Matryoshka itself
needs. Importing the library never touches this directory.

## Using a baseline

Baselines are resolved by name, lazily, from the repository root:

```python
import matryoshka as mk

mk.available_baselines()
cls = mk.resolve_baseline('ArdaAugmenter')   # imported on first use
```

`resolve_baseline` raises `ImportError` naming the missing dependency if the
adapter cannot be imported. Registering your own is one call:

```python
mk.register_baseline('MyAugmenter', 'mypackage.augmenters:MyAugmenter')
```

Alternatively, drive them through `DiscoveryConfig(baseline=True,
strategy='ArdaAugmenter', params={...})` and `JoinSelection.find_best_joins`,
which is the path the paper's harness uses.

Every adapter implements `run(**params)` and returns
`(augmented_table, discovery_seconds, selection_seconds, augmentation_plan)`.

## Contents

### `augmenters/` — feature discovery and augmentation systems

| Module | System | Upstream | Extra requirements |
|---|---|---|---|
| `arda.py` | ARDA | Chepurko et al., *ARDA: Automatic Relational Data Augmentation for Machine Learning*, PVLDB 13(9), 2020 | Aurum index |
| `arda_og.py` | ARDA, original implementation | as above | vendored upstream checkout, see below |
| `autofeat.py` | AutoFeat | Ionescu et al., *Feature Discovery for Data Augmentation*, ICDE 2024 | Neo4j, AutoGluon |
| `autofeat_og.py` | AutoFeat, original implementation | as above | vendored upstream checkout, see below |
| `caafe.py` | CAAFE | Hollmann et al., *Large Language Models for Automated Data Science*, NeurIPS 2023 | `caafe`, an LLM API key |
| `kitana.py` | Kitana | Liu et al., *Kitana: Efficient Data Augmentation Search*, ICDE 2024 | PyTorch, Neo4j |
| `qcr.py` | QCR correlation sketches | Santos et al., *Correlation Sketches for Approximate Join-Correlation Queries*, SIGMOD 2021 | its own PostgreSQL index |
| `cocoa.py` | COCOA | Esmailoghli et al., *COCOA: Correlation Coefficient-Aware Data Augmentation*, EDBT 2021 | `cocoa-system`, its own index |
| `metam.py` | Metam | Galhotra et al., *Metam: Goal-Oriented Data Discovery*, ICDE 2023 | `feature_engine`, an AutoML backend |

Supporting code: `*_utils/` hold the vendored parts of each upstream
implementation; `db_handlers.py` holds the PostgreSQL schemas for the COCOA and
QCR indexes; `neo4j_join_discovery.py` and `_og_paths.py` support the Neo4j-based
join-path discovery that AutoFeat, CAAFE and Kitana share.

### `discovery/` — join-discovery indexes

| Directory | System | Purpose |
|---|---|---|
| `Aurum/` | Aurum (Fernandez et al., ICDE 2018) | MinHash and HNSW join-discovery graphs; `aurum_join_discovery.py` wraps it behind the interface the augmenters expect |
| `Josie/` | JOSIE (Zhu et al., SIGMOD 2019) | exact set-containment search |
| `LSH/` | LSH Ensemble (Zhu et al., PVLDB 9(12), 2016) | with a vendored `datasketch` |
| `DeepJoin/` | DeepJoin (Dong et al., PVLDB 16(10), 2023) | embedding-based joinable-column search |

Each carries its own `*.md` with build and query instructions.

## What is not in the repository

Three classes of artefact were removed during the migration because they are
large binary or generated data, not source:

- **Pre-built discovery indexes.** `Aurum/graphs/` (13 GB of pickled MinHash and
  HNSW graphs), `DeepJoin/{nyc,cuk,gittables}_index/` (74 GB of embeddings and
  models). Rebuild them with the scripts in each directory.
- **`autofeat_og_utils/`**, a 462 MB checkout of the upstream AutoFeat
  repository including its own `.git`, datasets and AutoGluon model artefacts.
  Clone it from upstream next to `baselines/augmenters/` if the original
  implementation is needed; `_og_paths.py` resolves the path.
- **Baseline index dumps** for QCR and COCOA. Rebuild with `qcr_index.py` and
  `cocoa_index.py`.

## Provenance and licensing

The `*_utils/` trees and `discovery/` subdirectories are derived from the
authors' released implementations, adapted to a common interface and to
Matryoshka's lake readers. They remain under their original licences. Consult
each upstream repository before redistributing. The adapter modules at the top
of `augmenters/` are original to this project.
