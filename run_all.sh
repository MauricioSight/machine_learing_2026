# bash run_all.sh

# .venv/bin/python3 tunning_models.py --config bayes_class
.venv/bin/python3 tunning_models.py --config bayes_knn
.venv/bin/python3 tunning_models.py --config bayes_parzen
.venv/bin/python3 tunning_models.py --config bayes_logistic_regression
.venv/bin/python3 tunning_models.py --config majority_voting_classifier