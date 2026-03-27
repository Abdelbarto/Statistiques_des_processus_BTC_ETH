
# Statistiques des Processus Stochastiques — BTC/ETH 2025

> Projet académique — École Polytechnique
> Modélisation statistique des crypto-actifs Bitcoin et Ethereum
> sur données horaires haute fréquence (janvier–décembre 2025)

---

## Objectif

Ce projet applique un pipeline complet de modélisation statistique des
risques extrêmes sur les log-rendements horaires de BTC/USDT et ETH/USDT.
Il couvre la détection des effets ARCH, la modélisation de la volatilité
conditionnelle, la théorie des valeurs extrêmes (EVT) et la modélisation
de la dépendance via les copules.

---
```
## Structure du projet

Statistiques_des_processus_BTC_ETH/
│
├── Garch/
│   ├── figures/
│   ├── garch_step1_arch_tests.py
│   ├── garch_step2_estimation.py
│   └── garch_step3_flash_crash.py
│
├── evt/
│   ├── figures/
│   ├── evt_step1_data.py
│   ├── evt_step2_diagnostics.py
│   ├── evt_step3_threshold.py
│   ├── evt_step4_gev.py
│   ├── evt_step5_gpd_fit.py
│   ├── evt_step6_risk_measures.py
│   └── evt_step7_gbevt.py
│
├── copulas/
│   └── figures/
│
├── data/
│   ├── returns_btc_eth.csv
│   └── garch_residuals.csv
│
├── .gitignore
└── README.md
```



## Installation

```bash
git clone https://github.com/Abdelbarto/Statistiques_des_processus_BTC_ETH.git
cd Statistiques_des_processus_BTC_ETH

pip install arch scipy statsmodels matplotlib pandas numpy scikit-learn
```

**Python** : 3.10+

---



## Références

- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity*. Journal of Econometrics.
- Nelson, D. B. (1991). *Conditional heteroskedasticity in asset returns*. Econometrica.
- Pickands, J. (1975). *Statistical inference using extreme order statistics*. Annals of Statistics.
- Balkema, A. & de Haan, L. (1974). *Residual life time at great age*. Annals of Probability.
- Velthoen, J. et al. (2023). *Gradient boosting for extreme quantile regression*. Extremes.

---


## Auteurs

| Nom | Partie |
|-----|--------|
| Taha Meziane | *à compléter* |
| Abdelbar Ghassoub |  EVT |
| Kaoutar Bouaachra | *à compléter* | 


***


