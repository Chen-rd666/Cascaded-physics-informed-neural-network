The dataset is in the file all.xlsx. The ring numbers in the first column differ slightly from those in the paper: the first ring is numbered 100 in the file, whereas it is 0 in the paper.

The file all_predictions.xlsx contains the training history of the models from the paper, recorded every 100 iterations. The values recorded in the file are normalized and have not been inverse-transformed.

The file model_params.xlsx contains the model hyperparameters and other statistical data.

The file hypa.xlsx is an auxiliary file for CPINN.py. It contains the five physics‑informed hyperparameters of the CPINN model. By modifying the values in this file, different hyperparameters can be assigned to the CPINN model when used.

The file CPINN.py can be used to train a single model given the hyperparameters (due to later debugging and other reasons, the hyperparameters in this file are not exactly the same as those in the paper).

The file BO-CPINN.py uses Bayesian optimization for automatic hyperparameter tuning and builds a large number of CPINN models.

The  file model_weights.pth contains the model weights used in the paper.

The  file Invoke.py is used to read the weight file (model_weights.pth), the dataset file (all.xls), and the hyperparameter file (model_params.xlsx) to reproduce the paper's results.

In the modeling file “BO‑CPINN.py”, the architecture can be adjusted as needed. In the reproduction file “Invoke.py”, the author strictly adheres to the architecture described in the paper.
