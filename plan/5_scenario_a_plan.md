# Fáza 5 — Scenár A (zadanie č. 2): Parametrické vs. Neparametrické modely
### LoL Match Outcome Prediction

---

## Cieľ

Porovnanie minimálne 3 parametrických a 3 neparametrických modelov na agregovaných dátach. Vyhodnotenie výkonu, explainability, predpokladov a reprodukovateľnosti každej skupiny. Všetky modely sa trénujú a porovnávajú na **rovnakom feature sete** pre čisté porovnanie.

---

## 5.1 — Parametrické modely

Predpokladajú konkrétnu formu vzťahu medzi features a targetom alebo distribúciu dát.

### Logistická regresia
Predpokladá lineárny vzťah medzi features a log-odds targetu. Koeficienty priamo interpretovateľné — každý koeficient hovorí o vplyve danej feature na pravdepodobnosť výhry.

Overiť: linearita vzťahu (napr. cez partial regression plots). Ak je porušená, skúsiť regularizáciu (L1/L2) alebo polynomiálne features.

### LDA (Linear Discriminant Analysis)
Predpokladá normalitu features v rámci každej triedy a homogénnu kováriančnú maticu oboch tried.

Overiť predpoklady:
- Normalita: Q-Q ploty, Shapiro-Wilk test
- Homogénna kovariancia: Boxov M test

Ak predpoklady nie sú splnené:
- Skúsiť log transformáciu zošikmených features
- Prepnúť na **QDA** (Quadratic Discriminant Analysis) ktorá nevyžaduje homogénnu kováriančnú maticu — prirodzené rozšírenie LDA

### Naive Bayes
Predpokladá podmienečnú nezávislosť features — v praxi takmer vždy porušený predpoklad pri korelovaných dátach. Napriek tomu často funguje prekvapivo dobre.

Zdôvodniť v notebooku prečo je predpoklad porušený (korelácie z EDA) a čo to znamená pre interpretáciu výsledkov.

---

## 5.2 — Neparametrické modely

Nerobia predpoklady o distribúcii dát, učia sa priamo z dát.

### KNN (K-Nearest Neighbors)
Klasifikuje podľa majority triedy k najbližších susedov. Citlivý na scale (riešené v Fáze 4 — bez scalingu by features s veľkým rozsahom dominovali vzdialenostiam). Citlivý na vysoký počet features — curse of dimensionality: čím viac features, tým viac sú všetky body rovnako ďaleko od seba a KNN prestáva fungovať dobre. Z tohto dôvodu KNN profituje z feature selection (Scenár B) viac ako tree-based modely — toto zdôvodniť v diskusii.

Hyperparameter: počet susedov `k` — naladiť cez cross-validation.

### Random Forest
Ensemble rozhodovacích stromov. Robustný voči multikolinearite a outlierom. Poskytuje feature importance.

Hyperparametre: `n_estimators`, `max_depth`, `min_samples_split` — naladiť cez CV. Nastaviť `random_state` pre reprodukovateľnosť.

### Gradient Boosting
Sekvenčné budovanie stromov kde každý opravuje chyby predchádzajúceho. Zvyčajne najlepší výkon na tabulárnych dátach. Poskytuje feature importance.

Použiť `sklearn GradientBoostingClassifier` alebo `XGBoost`. Hyperparametre: `n_estimators`, `learning_rate`, `max_depth`.

---

## 5.3 — Overenie predpokladov

Pre parametrické modely explicitne overiť predpoklady pred fitovaním a zdokumentovať:
- Ak predpoklad splnený → pokračovať
- Ak nie → skúsiť transformáciu alebo alternatívny model (napr. QDA miesto LDA)
- Nedroppovať features len kvôli nesplneniu predpokladov

---

## 5.4 — Hyperparameter tuning

Nie každý model vyžaduje tuning — prístup zdôvodniť pre každý model zvlášť.

### Modely kde tuning má zmysel
- **KNN** — `k` (počet susedov) zásadne ovplyvňuje výkon. Malé `k` = overfitting, veľké `k` = underfitting. Tuning nutný.
- **Random Forest** — `n_estimators`, `max_depth`, `min_samples_split`. Výkon citlivý na tieto parametre.
- **Gradient Boosting** — `learning_rate`, `n_estimators`, `max_depth`. Veľmi citlivý na hyperparametre.
- **Logistická regresia** — parameter `C` (sila regularizácie). Dôležité hlavne pre LASSO v Scenári B kde `C` priamo ovplyvňuje koľko features vypadne.

### Modely kde tuning nie je kritický
- **LDA / QDA** — minimum hyperparametrov, použiť default hodnoty a zdôvodniť v notebooku.
- **Naive Bayes** — `var_smoothing` existuje ale výkon na ňom veľmi nezávisí. Tuning vynechať a zdôvodniť.

### Postup tuningu
Základný grid search cez cross-validation na **train sete** — nie je potrebný exhaustive search, stačí pár kľúčových hodnôt. Finálne vyhodnotenie vždy na test sete ktorý sa počas tuningu nepoužíval.

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_
```

---

## 5.5 — Detekcia a riešenie overfittingu

### Ako identifikovať overfitting
Porovnať train AUC vs. test AUC pre každý model. Ak je rozdiel výrazný, model sa naučil trénovacie dáta naspamäť ale negeneralizuje.

```python
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])
print(f"Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}, Rozdiel: {train_auc - test_auc:.3f}")
```

Orientačné pravidlo:
- Rozdiel < 0.02 — model generalizuje dobre
- Rozdiel 0.02–0.05 — mierne overfitting, sledovať
- Rozdiel > 0.05 — výrazný overfitting, treba riešiť

### Learning curves
Vizualizovať ako sa train a test výkon menia s veľkosťou trénovacej sady. Ak sa krivky nepribližujú, model má problém s generalizáciou.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5, scoring='roc_auc'
)
```

### Ktoré modely majú tendenciu overfitovať
- **Random Forest a Gradient Boosting** — bez regularizácie môžu perfektne fitovať train set
- **KNN s malým k** — príliš citlivý na jednotlivé body
- **Parametrické modely** — menej náchylné, ale možné pri veľkom počte features voči počtu riadkov

### Ako riešiť overfitting
Riešenie závisí od modelu — zdôvodniť zvolený prístup v notebooku:

| Model | Riešenie |
|-------|----------|
| Random Forest | Znížiť `max_depth`, zvýšiť `min_samples_split` |
| Gradient Boosting | Znížiť `learning_rate`, pridať `subsample` |
| KNN | Zvýšiť `k` |
| LogReg | Znížiť `C` (silnejšia regularizácia) |
| Všetky | Redukcia features zo Scenára B |

### Underfitting
Opačný problém — model je príliš jednoduchý a nedokáže zachytiť vzory v dátach. Prejaví sa nízkym výkonom na train aj test sete. Riešenie: komplexnejší model, viac features, slabšia regularizácia.

---

## 5.6 — Vyhodnotenie a porovnanie

### Accuracy
Percento správne klasifikovaných zápasov. Má zmysel použiť práve preto že náš dataset má ~50/50 class balance — pri imbalanced datasete by accuracy klamala (model ktorý vždy predikuje majoritnú triedu by mal vysokú accuracy). Tu je to spoľahlivá metrika.

Limitácia: nehovorí nič o tom kde a ako model chybuje — preto nestačí samotná.

### ROC-AUC
Plocha pod ROC krivkou — meria ako dobre model rozlišuje výhry od prehier naprieč všetkými možnými klasifikačnými thresholdmi. Hodnota 0.5 = náhodný model, 1.0 = perfektný model.

Toto je **hlavná metrika pre porovnanie modelov** pretože je nezávislá od konkrétneho klasifikačného threshold a robustná voči class balance. Umožňuje spravodlivé porovnanie modelov ktoré predikujú pravdepodobnosti (LogReg, RF) aj modelov ktoré predikujú priamo triedy (KNN).

### Confusion matrix
Tabuľka skutočných vs. predikovaných tried — zobrazuje True Positives, True Negatives, False Positives, False Negatives. Hovorí konkrétne kde model chybuje — či systematicky predikuje výhry ako prehry alebo naopak. Pre náš problém (výhra vs. prehra) sú oba typy chýb rovnako závažné, takže asymetria v confusion matrix by bola zaujímavý nález.

### Precision / Recall / F1
- **Precision** — z tých zápasov ktoré model predikoval ako výhru, koľko percent skutočne bolo výhra
- **Recall** — z tých zápasov ktoré skutočne boli výhra, koľko percent model správne identifikoval
- **F1** — harmonický priemer Precision a Recall

Pre náš problém (symetrický, 50/50 balance) sú tieto metriky doplnkové — primárne slúžia na detekciu systematickej chyby jedným smerom. Zdôvodniť v notebooku.

### Train AUC vs. Test AUC
Porovnanie výkonu na trénovacej a testovacej sade. Veľký rozdiel (napr. train AUC 0.95, test AUC 0.70) signalizuje overfitting — model sa naučil trénovacie dáta naspamäť ale negeneralizuje. Neparametrické modely (hlavne Random Forest a GBM) majú väčšiu tendenciu overfitovať ak nie sú správne regularizované — toto je dôležitý argument v diskusii parametrické vs. neparametrické.

### Zhrnutie výsledkov

| Model | Typ | Accuracy | ROC-AUC | Train AUC | Predpoklady splnené |
|-------|-----|----------|---------|-----------|----------------------|
| Logistická regresia | parametrický | | | | |
| LDA / QDA | parametrický | | | | |
| Naive Bayes | parametrický | | | | |
| KNN | neparametrický | | | | |
| Random Forest | neparametrický | | | | |
| Gradient Boosting | neparametrický | | | | |

---

## 5.7 — Explainability

Kľúčová časť diskusie v notebooku:

**Parametrické modely** — koeficienty priamo interpretovateľné. Ktoré features majú najväčší vplyv? Zhodujú sa s hypotézami z EDA?

**Neparametrické modely** — feature importance z Random Forest a Gradient Boosting. Porovnať s koeficientmi parametrických modelov — zhodujú sa ktoré features sú dôležité?

Diskutovať: ktorá skupina modelov je explainabilnejšia a prečo. Aký je trade-off medzi výkonom a interpretovateľnosťou.

---

## 5.8 — Prepojenie so Scenárom B

Po dokončení Scenára B (feature selection) zopakovať kľúčové modely na **redukovanom feature sete** a porovnať:
- Zmenil sa výkon (ROC-AUC)?
- Zlepšila sa interpretovateľnosť?
- Ktoré features Scenár B vyhodil a zhoduje sa to s feature importance zo Scenára A?

> Detailný plán Scenára B v `scenario_b_plan.md`
