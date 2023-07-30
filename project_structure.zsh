#!/usr/bin/env zsh

mkdir -p stochastiqML
touch stochastiqML/__init__.py
touch stochastiqML/neural_sde.py
touch stochastiqML/sde_vae.py
touch stochastiqML/sde_gan.py
touch stochastiqML/sde_rnn.py

mkdir -p tests
touch tests/__init__.py
touch tests/test_neural_sde.py
touch tests/test_sde_vae.py
touch tests/test_sde_gan.py
touch tests/test_sde_rnn.py

mkdir -p examples
touch examples/neural_sde_example.ipynb
touch examples/sde_vae_example.ipynb
touch examples/sde_gan_example.ipynb
touch examples/sde_rnn_example.ipynb

mkdir data
mkdir runs

touch setup.py
touch pyproject.toml
touch README.md
echo '*.pyc' >> .gitignore
