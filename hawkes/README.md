# Hawkes Process Analysis
$hawkes = @"
# Analyse par Processus de Hawkes

Ce dossier contient le notebook pour l'analyse par processus de Hawkes
du Flash Crash BTC/ETH du 10 octobre 2025.

## Contenu
- ``hawkes.ipynb`` : estimation des processus de Hawkes univariés et
  multivariés, adaptation du modèle 2T-POT (Tomlinson et al., 2024),
  analyse du taux de branchement et quantification de la contagion
  directionnelle ETH -> BTC.

## Dépendances
``pip install tick numpy pandas matplotlib scipy``
