conda create -n lachlan_models_airpoll python=3.10 -y

conda activate lachlan_models_airpoll

pip install torch
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tqdm
pip install ucimlrepo

pip install tabulate