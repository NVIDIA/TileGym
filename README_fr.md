<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

[English](README.md) | [简体中文](README_chs.md) | [繁體中文](README_cht.md) | [日本語](README_ja.md) | Français

# TileGym

TileGym est une bibliothèque de noyaux CUDA Tile qui fournit une riche collection de tutoriels et d'exemples de noyaux pour la programmation GPU basée sur les tuiles.

[**Aperçu**](#aperçu) |
[**Fonctionnalités**](#fonctionnalités) |
[**Installation**](#installation) |
[**Démarrage rapide**](#démarrage-rapide) |
[**Contribution**](#contribution) |
[**Licence**](#licence-et-avis-relatifs-aux-tiers)

## Aperçu

Ce dépôt vise à fournir des tutoriels et des exemples de noyaux utiles pour la programmation GPU basée sur les tuiles. TileGym est un terrain d'expérimentation pour CUDA Tile, où vous pouvez apprendre à construire des noyaux GPU efficaces et explorer leur intégration dans des modèles de langage à grande échelle tels que Llama 3.1 et DeepSeek V2. Que vous appreniez la programmation GPU basée sur les tuiles ou que vous cherchiez à optimiser vos implémentations de LLM, TileGym offre des exemples pratiques et des conseils complets.
<img width="95%" alt="tilegym_1_newyear" src="https://github.com/user-attachments/assets/f37010f5-14bc-44cd-bddf-f517dc9922b8" />

## Fonctionnalités

- Riche collection d'exemples de noyaux CUDA Tile
- Implémentations pratiques de noyaux pour les opérateurs courants d'apprentissage profond
- Benchmarks de performance pour évaluer l'efficacité des noyaux
- Exemples d'intégration de bout en bout avec des LLM populaires (Llama 3.1, DeepSeek V2)

## Installation

### Prérequis

> **Support GPU** : TileGym nécessite **CUDA 13.1+** et un **GPU Blackwell** (ex. B200, RTX 5080, RTX 5090). Les **GPU NVIDIA Ampere** (ex. A100) sont également supportés avec **CUDA 13.2+**. Tous les noyaux publiés sont validés sur les deux architectures. Téléchargez CUDA depuis [Téléchargements NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads).

- PyTorch (version 2.9.1 ou compatible)
- **[CUDA 13.1+](https://developer.nvidia.com/cuda-downloads)** (Requis - TileGym est construit et testé exclusivement sur CUDA 13.1+)
- Triton (inclus avec l'installation de PyTorch)

### Étapes d'installation

#### 1. Préparer l'environnement `torch` et `triton`

Si vous avez déjà `torch` et `triton`, passez cette étape.

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/cu130
```

Nous avons vérifié que `torch==2.9.1` fonctionne. Vous pouvez également obtenir les paquets `triton` lors de l'installation de `torch`.

#### 2. Installer TileGym

TileGym utilise [`cuda-tile`](https://github.com/nvidia/cutile-python) (≥ 1.3.0) pour la programmation de noyaux GPU, qui dépend du compilateur `tileiras` à l'exécution.

##### Installer depuis PyPI (recommandé)

```bash
pip install tilegym[tileiras]
```

Ceci installe TileGym et toutes les dépendances d'exécution, y compris `cuda-tile[tileiras]` qui intègre le compilateur `tileiras` directement dans votre environnement Python.

Si `tileiras` est déjà disponible sur votre système (par ex. depuis [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads)), vous pouvez omettre l'extra :

```bash
pip install tilegym
```

##### Installer depuis les sources

```bash
git clone https://github.com/NVIDIA/TileGym.git
cd TileGym
pip install .[tileiras]   # ou : pip install .  (si vous avez tileiras sur votre système)
```

Pour le mode éditable (développement), utilisez `pip install -e .` ou `pip install -e .[tileiras]`.

Toutes les dépendances d'exécution sont déclarées dans [`requirements.txt`](requirements.txt) et sont installées automatiquement par `pip install tilegym` et `pip install .`.

Nous fournissons également un Dockerfile, vous pouvez consulter [modeling/transformers/README.md](modeling/transformers/README.md).

### Backends

TileGym fournit des noyaux pour les backends suivants, chacun dans son propre répertoire sous `src/tilegym/ops/` :

- **cuTile** (par défaut) — [`src/tilegym/ops/cutile`](src/tilegym/ops/cutile), voir plus de détails dans [cutile-python](https://github.com/nvidia/cutile-python).
- **CUDA Tile C++** — [`src/tilegym/ops/tilecpp`](src/tilegym/ops/tilecpp), voir plus de détails dans [README.tilecpp.md](README.tilecpp.md).
- **Triton CUDA Tile IR** — [`src/tilegym/ops/triton`](src/tilegym/ops/triton), voir plus de détails dans [Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR).

Pour utiliser le backend Triton CUDA Tile IR, installez son wheel dans un répertoire séparé et sélectionnez-le à l'exécution avec `ENABLE_TILE=1`. Des wheels pour CPython 3.12 et 3.13 sont disponibles sur la [page des versions](https://github.com/triton-lang/Triton-to-tile-IR/releases) :

```bash
# Installez dans un répertoire séparé, à l'écart de l'environnement par défaut
pip install --target /opt/nvtriton <nvtriton-wheel-for-your-python>.whl

# Sélectionnez le backend Triton CUDA Tile IR à l'exécution
PYTHONPATH=/opt/nvtriton ENABLE_TILE=1 python your_script.py
```

## Démarrage rapide

Il existe trois façons principales d'utiliser TileGym :

### 1. Explorer les exemples de noyaux

Toutes les implémentations de noyaux se trouvent dans le répertoire `src/tilegym/ops/`. Vous pouvez tester des opérations individuelles avec des scripts minimaux. L'utilisation au niveau des fonctions et les scripts minimaux pour les opérations individuelles sont documentés dans [tests/ops/README.md](tests/ops/README.md)

### 2. Exécuter les benchmarks

Évaluez les performances des noyaux avec des micro-benchmarks :

```bash
cd tests/benchmark
bash run_all.sh
```

Le guide complet des benchmarks est disponible dans [tests/benchmark/README.md](tests/benchmark/README.md)

### 3. Exécuter les exemples LLM Transformer

Utilisez les noyaux TileGym dans des scénarios d'inférence de bout en bout. Nous fournissons des scripts exécutables et des instructions pour les modèles de langage Transformer (par ex. Llama 3.1-8B) accélérés à l'aide des noyaux TileGym.

Tout d'abord, installez la dépendance supplémentaire :

```bash
pip install accelerate==1.13.0 --no-deps
```

**Configuration conteneurisée (Docker)** :

```bash
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

Plus de détails dans [modeling/transformers/README.md](modeling/transformers/README.md)

### 4. Noyaux Julia (cuTile.jl) (Optionnel)

TileGym inclut également des implémentations expérimentales de noyaux [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) en Julia. Ceux-ci sont autonomes dans le répertoire `julia/` et ne nécessitent pas le paquet Python TileGym.

**Prérequis** : [Julia 1.12+](https://julialang.org/downloads/), CUDA 13.1, GPU Blackwell

```bash
# Installer Julia (si non installé)
curl -fsSL https://install.julialang.org | sh

# Installer les dépendances
julia --project=julia/ -e 'using Pkg; Pkg.instantiate()'

# Exécuter les tests
julia --project=julia/ julia/test/runtests.jl
```

Consultez `julia/Project.toml` pour la liste complète des dépendances.

### 5. Activer le backend cuTile-rs (Rust) (Optionnel)

Un sous-ensemble d'opérateurs fournit un backend **cuTile-rs** supplémentaire sous
[`src/tilegym/ops/cutile_rs`](src/tilegym/ops/cutile_rs) — des noyaux écrits en Rust
avec [`cutile-rs`](https://github.com/NVlabs/cutile-rs) et chargés via une
`libcutile_kernels.so` à ABI C. Il est optionnel et utilisable uniquement depuis une
installation depuis les sources.

**Prérequis** (en plus de l'installation de base ci-dessus), conformes à
[cuTile-rs](https://github.com/NVlabs/cutile-rs) :

- **Rust 1.89+** — `cargo` et `rustc` dans le `PATH` :

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup default stable
  ```

- **CUDA toolkit avec les en-têtes** — la compilation Rust exécute `bindgen` sur
  `cuda.h`. Définissez `CUDA_TOOLKIT_PATH` vers votre installation ; s'il n'est pas
  défini, cuTile-rs utilise `/usr/local/cuda` par défaut :

  ```bash
  export CUDA_TOOLKIT_PATH=/usr/local/cuda   # doit contenir include/cuda.h
  ```

**Utilisation.** Le chargeur du backend compile la bibliothèque partagée de façon paresseuse
lors de la première utilisation (`cargo build --release`), aucune étape de compilation manuelle n'est donc requise :

```python
import tilegym
tilegym.set_backend("cutile-rs")

from tilegym.backend.selector import get_available_backends
print(get_available_backends())        # doit inclure "cutile-rs"

from tilegym.ops import bmm             # import indépendant du backend
# ... bmm(...) est désormais dispatché vers le noyau cuTile-rs
```

**Variables d'environnement optionnelles :**

```bash
export CUTILE_RS_AUTOBUILD=0                          # ignorer la recompilation paresseuse ; utiliser un .so pré-compilé
export CUTILE_RS_KERNELS_DIR=/abs/path/to/cutile_kernels   # remplacer l'emplacement du crate
```

> Si `cargo` n'est pas dans le `PATH` et qu'aucune `libcutile_kernels.so` pré-compilée n'est présente,
> le backend se déclare indisponible et les tests cuTile-rs sont ignorés plutôt qu'échoués.

**Benchmark de cuTile-rs.** Pour mesurer les performances de cuTile-rs, exécutez les tests
de perf avec **`CUPTI=1`** (utilise le temps GPU de CUPTI / `torch.profiler` au lieu des
CUDA events). Les noyaux cuTile-rs ont souvent une surcharge host/launch différente de la
référence, que le chronométrage en temps réel des CUDA events surestime sur les petits
noyaux (sous la microseconde) ; CUPTI mesure le temps GPU pur du noyau et donne un ratio
stable et comparable :

```bash
CUPTI=1 pytest tests/ops/test_bmm.py -k "test_perf and cutile_rs" --print-record
```

## Contribution

Nous accueillons les contributions de toutes sortes. Veuillez lire notre [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives, y compris le processus d'accord de licence de contributeur (CLA).

## Licence et avis relatifs aux tiers

- Licence du projet : MIT
  - [LICENSE](LICENSE)
- Attributions et textes de licence des tiers :
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
