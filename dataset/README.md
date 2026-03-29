# Dataset — Data Download and Analysis
# Données — Téléchargement et Analyse Exploratoire

Ce dossier contient les notebooks pour la collecte et l'analyse
exploratoire des données horaires BTC/USD et ETH/USD sur 2025.

## Contenu
- ``data_download.ipynb`` : téléchargement des prix horaires BTC/USD
  et ETH/USD via yfinance (année complète 2025) et des données à la
  minute depuis l'API publique Binance (fenêtre du Flash Crash,
  6--12 octobre 2025).
- ``Data_analysis.ipynb`` : analyse exploratoire — niveaux de prix,
  log-rendements, volatilité roulante, volume et visualisation
  du Flash Crash.

## Sources des données
- Yahoo Finance (yfinance) : BTC-USD, ETH-USD, SPX, VIX horaires
- API publique Binance : BTCUSDT, ETHUSDT à la minute

## Dépendances
``pip install yfinance pandas numpy matplotlib requests``
