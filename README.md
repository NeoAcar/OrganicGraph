## Setup Instructions

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management and dependency installation. Please refer to the official uv installation page for instructions on installing uv.

### 1. Clone the Repository

```bash
git clone https://github.com/NeoAcar/OrganicGraph.git
cd OrganicGraph
```

### 2. Create and Activate the `uv` Environment

```bash
uv venv --python 3.14
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
uv pip install -e .
```