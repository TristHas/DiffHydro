# DiffHydro
Differentiable Hydrological Pipelines Made Easy


## Overview

**DiffHydro** is a research-oriented Python library for building, assembling, and training **fully differentiable hydrological modeling pipelines** using PyTorch.

DiffHydro formulates hydrological models as pipelines of operators (catchment interpolation, runoff generation, routing, dams, indundation modules, etc.). 
If each operator is differentiable, the *entire hydrological system* can be trained end-to-end using gradient-based optimization.

DiffHydro provides the building blocks and reference pipelines to make this practical, modular, and extensible.

> ⚠️ **Status**: DiffHydro is in **early development**. APIs may change, and the library is primarily intended for research and experimentation.

---

## Key Features

* **End-to-end differentiable hydrological pipelines**
* **Modular design**: compose models from reusable operators
* **Learnable physical parameters** (routing, response functions, etc.)
* **PyTorch-native** (GPU-friendly, autograd-compatible)
* **Scalability** Important efforts have been put to make very large problem sizes fit into GPU memories.
* **Reference pipelines** for common hydrological learning setups
* **Examples on real datasets**, from toy problems to larger-scale cases

---

## Library Structure

DiffHydro is organized around four main components:

```text
diffhydro/
├── structs/     # Core data structures
├── modules/     # Individual differentiable operators
├── pipelines/   # High-level model & training pipelines
examples/        # End-to-end usage examples
```

### `structs/`

Defines the **core data structures** manipulated throughout the library. 

---

### `modules/`

Contains **individual differentiable operators**.
The current releases includes three main components:

* Catchment Interpolation
* Runoff Generation
* Routing Operator

Each module is designed to be:

* Differentiable
* Composable
* Reusable across pipelines

Expect more components to be added soon.

---

### `pipelines/`

Provides **high-level abstractions** for assembling and training models.

This includes:

* **Models**: containers that assemble multiple modules and expose learnable parameters
* **Modules**: utilities that orchestrate data loading, scheduling, and optimization

Pipelines illustrate *typical and recommended usage patterns*, but they are not restrictive: 
advanced users are encouraged to assemble their own pipelines directly from modules.

---

### `examples/`

Example notebooks demonstrating how to use DiffHydro on **actual hydrological problems**, covering:

* Different spatial scales
* Different levels of physical complexity
* Both small and large datasets

If you are new to DiffHydro, **start here**.

---

## Installation

Install from pip:

```bash
pip install diffhydro
```

Or install the latest version directly from source:

```bash
git clone https://github.com/TristHas/DiffHydro.git
cd DiffHydro
pip install -e .
```