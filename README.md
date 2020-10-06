# Feature Analyzer

## Development Branch: Dynamic
TODO
- intra-class and inter-class analysis
- cross model or cross temperal analysis
- Analyze feature behaviour across difference checkpoints
- Support online (training stage) evaluation

### functions
- ~~embedding_container load from template folder~~
- trace assigned identities

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
 featAnalyzer -c configs/default.yaml -dd sample-data/embedding_container_example/
```

Variance analysis example
```bash
python scripts/analyze_variance.py -ec sample-data/embedding_container_example/ -rc sample-data/variance_example/
```