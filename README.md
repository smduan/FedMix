# FedMix

This repo is the implementation of paper [FedMix: Approximation of Mixup under Mean Augmented Federated Learning](https://openreview.net/pdf?id=Ogga20D2HO-)



### Code structure

- `conf.py`: configuration file
- `data`: directory containing data
- `fedavg`
  - `client.py`
  - `datasets.py`
  - `models.py`
  - `server.py`
- `main.py`: the main procedure
- `utils.py`: utility codes



### Run this repo

1. `clinical` dataset is provided in this repo as an example. If you want to train on your own data, please organize it after the following form:

   ```bash
   data
   ├── clinical
   │   ├── clinical_test.csv
   │   ├── beta0.05
   │   │   ├── clinical_node_0.csv
   │   │   ├── clinical_node_1.csv
   │   │   ├── clinical_node_2.csv
   │   │   ├── clinical_node_3.csv
   │   │   └── clinical_node_4.csv
   │   ├── beta0.5
   │   └── ...
   └── ...
   ```
   
   where `clinical_test.csv` is the test set and `clinical_node_{i}.csv` is the training set in the i'th client. `beta0.05` indicates that the training data are partitioned following a Dirichlet distribution with parameter equals to 0.05.
   
2. Edit the configuration file `conf.py`. Some important arguments are:

   - `global_epochs`: number of global epochs
   - `local_epochs`: number of local epochs
   - `beta`: parameter of Dirichlet distribution
   - `mean_batch`: number of instances used in computing the average in FedMix
   - `lambda`: coefficient in loss of FedMix
   - `lr`, `momentum`: optimizer settings
   - `num_parties`: number of parties (clients)
   - ...

3. Start training:

   ```bash
   python main.py
   ```
   
   The best models will be saved in `./save_model/` after training.

