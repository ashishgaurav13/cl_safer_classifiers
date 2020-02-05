This repo contains the code accompanying the SafeAI-20 paper "Simple Continual Learning Strategies for Safer Classifiers". To cite this work, please use the following bibtex entry:

```
@article{gaurav2020simple,
  title={Simple Continual Learning Strategies for Safer Classifiers},
  author={Gaurav, Ashish and Vernekar, Sachin and Lee, Jaeyoung and Abdelzad, Vahdat and Czarnecki, Krzysztof and Sedwards, Sean},
  year={2020}
}
```

**Paper**: Link will be updated after publication in CEUR-WS.

**Methods**:

* <u>Baseline</u>: no regularization
```
python3 baseline.py -dataset {datasetid}
```
* <u>EWC</u>: Elastic weight consolidation with accumulated Fishers and quadratic losses
```
python3 ewc.py -dataset {datasetid} -const {lambda}
```
* <u>SI</u>: Synaptic Intelligence
```
python3 si.py -dataset {datasetid} -c {c} -xi {xi}
```
* <u>DM</u>: Direct Minimization
```
python3 dm.py -dataset {datasetid} -const {lambda} -case {1/2/3/4} -norm {l1/l2/e0.5}
```
* Direct Minimization with Fine Control
```
python3 dm.py -dataset {datasetid} -const {lambda} -case {1/2/3/4} -norm {l1/l2/e0.5} -c1 {c1} -c2 {c2}
```
* <u>EWC-p</u>: Fisher Freezing
```
python3 ewcgrad.py -dataset {datasetid} -const {lambda} -fix {p value}
```

**How to use (Python3)**:

* Install dependencies as listed in `requirements.txt`
* Download all datasets first by running `python download.py`
* Use scripts `{baseline,ewc,si,dm,ewcgrad}.py`
* Logs will be created in `logs` folder.
