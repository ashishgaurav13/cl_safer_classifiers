This repo contains the code accompanying the SafeAI-20 paper "Simple Continual Learning Strategies for Safer Classifiers". To cite this work, please use the following bibtex entry:

```
@article{gauravsimple,
  title={Simple Continual Learning Strategies for Safer Classifiers},
  author={Gaurav, Ashish and Vernekar, Sachin and Lee, Jaeyoung and Abdelzad, Vahdat and Czarnecki, Krzysztof and Sedwards, Sean},
  year={2020}
}
```

**Paper**: [[Workshop Proceedings](http://ceur-ws.org/Vol-2560/)] [[Paper PDF](http://ceur-ws.org/Vol-2560/paper6.pdf)]

**Parameters for Methods**:
* `{datasetid}`: could be P-MNIST (0), S-MNIST (1), Sim-EMNIST (2), CIFAR100 (3), Sim-CIFAR100 (4)
* `{lambda}`: regularization constant parameter for EWC or DM
* `{c}, {xi}`: parameters from Synaptic Intelligence paper
* `{p value}`: decimal number in `[0, 1]` for EWC-p

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
* Install dependencies: `pip3 install -r requirements.txt`
* Download all datasets first by running `python3 download.py`
* Use scripts `python3 {baseline,ewc,si,dm,ewcgrad}.py`
* Logs will be created in `logs` folder.

**References and Sources**:
* Elastic Weight Consolidation (Kirkpatrick et al., 2017): [Paper link](https://www.pnas.org/content/114/13/3521.short)
* Synaptic Intelligence (Zenke et al., 2017): [Paper link](https://dl.acm.org/doi/10.5555/3305890.3306093), [Github Link](https://github.com/ganguli-lab/pathint) - Official Code
* EWC code based on unofficial implementation by stokesj: [Github Link](https://github.com/stokesj/EWC)

**License**:

MIT License. For EWC/SI code, also check the licensing information in repositories provided in references.
