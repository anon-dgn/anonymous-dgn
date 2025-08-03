# Anonymous Multi-Instance Learning Framework (Under Submission)

This repository contains the **official anonymous implementation** of a novel graph-based Multi-Instance Learning (MIL) framework, currently **under review** at a top-tier AI conference.

---

## 🧠 Overview

This project implements a novel Multi-Instance Learning (MIL) framework that enhances bag-level classification by modeling fine-grained instance combinations across bags.

Unlike conventional MIL methods that treat each bag as an isolated unit and rely only on intra-bag features, our approach constructs a **global instance correlation graph** that breaks bag boundaries and enables learning of inter-bag instance relationships.

To achieve this, we propose a graph-based MIL architecture that models the entire instance space as a **“drawer graph”**, where:
- Bags occupy distinct regions in the graph,
- Instances are distributed across these regions,
- And classification is achieved via global instance-level message passing.

A contrastive loss is incorporated to enhance representation learning across both intra- and inter-bag instance relations. Experimental results demonstrate that this framework outperforms existing SOTA methods on several MIL benchmarks.

---

## ⚙️ Requirements

This codebase has been tested with:

- Python 3.8+
- CUDA 11.7/11.8
- PyTorch 1.13.0
- torch-geometric 2.3.1
- DGL 1.1.2+cu117

### 🔧 Installation

Create a Python environment (conda or venv recommended), then install dependencies:

```bash
pip install -r requirements.txt
