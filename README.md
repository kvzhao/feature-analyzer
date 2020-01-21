# Feature Analyzer

## Development Branch
TODO
- intra-class and inter-class analysis
- cross model or cross temperal analysis

## Quick Start

### Installation

```
python setup.py install
```
The command-line tool will be installed as `featureAnalyzer`

### Dependencies and `hnswlib`
Run the script to install
```bash
  sh install_hnsw.sh
```
Source: [hnswlib](https://github.com/nmslib/hnswlib)

### Usage
```bash
 featureAnalyzer -c configs/eval_container_example.yaml -dd feature-examples/container_example/
```
