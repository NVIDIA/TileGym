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

> ⚠️ **Important** : TileGym nécessite **CUDA 13.1** et des **GPU d'architecture NVIDIA Blackwell** (par ex. B200, RTX 5080, RTX 5090). Nous prendrons en charge d'autres architectures GPU à l'avenir. Téléchargez CUDA depuis [Téléchargements NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads).

- PyTorch (version 2.9.1 ou compatible)
- **[CUDA 13.1](https://developer.nvidia.com/cuda-downloads)** (Requis - TileGym est construit et testé exclusivement sur CUDA 13.1)
- Triton (inclus avec l'installation de PyTorch)

### Étapes d'installation

#### 1. Préparer l'environnement `torch` et `triton`

Si vous avez déjà `torch` et `triton`, passez cette étape.

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/cu130
```

Nous avons vérifié que `torch==2.9.1` fonctionne. Vous pouvez également obtenir les paquets `triton` lors de l'installation de `torch`.

#### 2. Installer TileGym

```bash
git clone <tilegym-repository-url>
cd tilegym
pip install .
```
Cela installera automatiquement `cuda-tile`, voir https://github.com/nvidia/cutile-python.

Si vous souhaitez utiliser le mode édition pour `TileGym`, exécutez `pip install -e .`

Nous fournissons également un Dockerfile, vous pouvez consulter [modeling/transformers/README.md](modeling/transformers/README.md).

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
pip install accelerate --no-deps
```

**Configuration conteneurisée (Docker)** :

```bash
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

Plus de détails dans [modeling/transformers/README.md](modeling/transformers/README.md)

## Contribution

Nous accueillons les contributions de toutes sortes. Veuillez lire notre [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives, y compris le processus d'accord de licence de contributeur (CLA).

## Licence et avis relatifs aux tiers

- Licence du projet : MIT
  - [LICENSE](LICENSE)
- Attributions et textes de licence des tiers :
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
