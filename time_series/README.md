# Time Series Analysis
# Analyse des Séries Temporelles Linéaires

Ce dossier contient le notebook pour l'analyse des séries temporelles
linéaires des log-rendements horaires de BTC et ETH sur 2025.

## Contenu
- ``time_series_analysis.ipynb`` : tests de stationnarité (ADF, KPSS),
  analyse ACF/PACF, sélection de l'ordre ARMA par grille BIC,
  analyse spectrale (périodogramme de Welch), estimation de la mémoire
  longue (exposant de Hurst via R/S, estimateur GPH) et différenciation
  fractionnaire (d minimum pour la stationnarité).

## Résultats principaux
- Log-rendements stationnaires I(0) pour les deux actifs
- Modèle retenu : ARMA(0,1) avec theta_1 ~ 0.02
- Absence de mémoire longue significative (d_GPH ~ 0.02, H ~ 0.54)
- Différenciation fractionnaire minimale d_min = 0.25 pour les log-prix

## Dépendances
``pip install yfinance statsmodels scipy matplotlib numpy``
