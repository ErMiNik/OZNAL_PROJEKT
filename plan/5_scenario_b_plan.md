# Fáza 6 — Scenár B (zadanie č. 3): Feature Selection
### LoL Match Outcome Prediction

---

## Cieľ

Porovnanie embedded a wrapper/algoritmických metód feature selection na identifikáciu najdôležitejších herných metrík. Výstupom je redukovaný feature set ktorý sa použije v Scenári A pre porovnanie výkonu modelov na plnom vs. redukovanom feature sete.

---

## 6.1 — Dôležitý princíp: leakage pri feature selection

Feature selection sa musí vykonávať **výlučne na train sete** — rovnaký princíp ako scaling. Ak by sme vybrali features na celom datasete (vrátane test setu) a až potom robili train/test split, informácia z test setu by ovplyvnila výber features čo je data leakage.

Správne poradie:
1. Train/test split (Fáza 4)
2. Feature selection fit na train sete
3. Transformácia train aj test setu (ponechať len vybrané features)
4. Trénovanie modelov na train sete
5. Vyhodnotenie na test sete

Odporúča sa zabaliť do `sklearn.pipeline.Pipeline` ktorý toto poradie vynúti automaticky.

---

## 6.2 — Embedded metódy

Feature selection je priamo súčasťou trénovania modelu — penalizácia alebo štruktúra modelu sama vyradí nerelevantné features.

### LASSO (L1 regularizácia)
Logistická regresia s L1 penalizáciou ktorá tlačí koeficienty slabých features presne na nulu — výsledkom je automaticky redukovaný feature set. Features s koeficientom = 0 sú vyradené.

Sila regularizácie sa kontroluje parametrom `C` (inverzná regularizácia — menší C = silnejšia penalizácia = viac features vypadne). Optimálne `C` naladiť cez cross-validation.

```python
from sklearn.linear_model import LogisticRegression

lasso = LogisticRegression(penalty='l1', solver='liblinear', C=C_optimal)
lasso.fit(X_train_scaled, y_train)
selected = X.columns[lasso.coef_[0] != 0]
```

Dôležité: LASSO je viazané na lineárny model — vyberá features ktoré sú dôležité z pohľadu lineárneho vzťahu s targetom.

### Ridge (L2 regularizácia)
Logistická regresia s L2 penalizáciou — koeficienty sa zmenšujú ale nikdy presne na nulu. Nevykonáva skutočný feature selection ale poskytuje ranking relatívnej dôležitosti features. Slúži ako porovnanie s LASSO.

### Feature importance z Random Forest
Každý strom v ensembli meria o koľko každá feature zlepšuje čistotu splitov (Gini importance). Agregát cez všetky stromy dáva ranking features — tie s najnižšou importance možno vyradiť.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
```

Dôležité: Random Forest importance je viazané na tree-based model — môže dávať iné výsledky ako LASSO. Ak sa zhodujú na top features, je to silný dôkaz že tieto features sú skutočne dôležité nezávisle od modelu. Ak sa líšia, diskutovať prečo.

---

## 6.3 — Wrapper / algoritmické metódy

Vyberajú features iteratívne na základe výkonu modelu.

### RFE (Recursive Feature Elimination)
Trénuje model na všetkých features, odstráni najmenej dôležitú feature, znova trénuje, opakuje až do požadovaného počtu features. Výsledkom je ranking features podľa toho kedy vypadli.

```python
from sklearn.feature_selection import RFE, RFECV

rfe = RFECV(estimator=LogisticRegression(), cv=5, scoring='roc_auc')
rfe.fit(X_train_scaled, y_train)
selected = X.columns[rfe.support_]
```

Odporúča sa `RFECV` (RFE s cross-validation) ktorý automaticky určí optimálny počet features na základe výkonu — nie je potrebné manuálne nastaviť počet features.

Výpočtovo náročnejšie ako embedded metódy — každý krok vyžaduje retréning modelu.

---

## 6.4 — Porovnanie metód

Pre každú metódu zdokumentovať:
- Koľko features zostalo
- Ktoré features boli vyradené
- Ktoré features sú v top 10 podľa každej metódy

Výsledky zhrnúť do tabuľky:

| Feature | LASSO koef. | RF importance | RFE rank | Zachovaná? |
|---------|-------------|---------------|----------|------------|
| t1_gold_mean | | | | |
| t1_kills_sum | | | | |
| ... | | | | |

### Kľúčové otázky pre diskusiu
- Zhodujú sa metódy na tom ktoré features sú najdôležitejšie? Ak áno, silný dôkaz.
- Ak sa líšia — je to kvôli tomu že každá metóda je viazaná na iný model (lineárny vs. tree-based)?
- Potvrdzujú výsledky hypotézy z EDA? (napr. je `gold_per_minute` dôležitejší ako `kills`?)
- Zostáva `vision_score` signifikantný po zarátaní ekonomických metrík?

---

## 6.5 — Výstup: redukovaný feature set

Na základe porovnania metód vybrať finálny redukovaný feature set — zdôvodniť výber v notebooku. Tento feature set sa použije v Scenári A kde sa porovná výkon modelov na plnom vs. redukovanom sete.

Očakávané otázky:
- Zlepšil sa ROC-AUC po redukcii features?
- Ktoré modely profitujú z redukcie najviac? (KNN — curse of dimensionality, LogReg — multikolinearita)
- Koľko features stačí na dobrý výkon?

---

## 6.6 — Prepojenie so Scenárom A

Po dokončení Scenára B zopakovať všetky modely zo Scenára A na redukovanom feature sete a doplniť do porovnávacej tabuľky:

| Model | Features | ROC-AUC | Accuracy |
|-------|----------|---------|----------|
| LogReg | všetky | | |
| LogReg | redukované | | |
| KNN | všetky | | |
| KNN | redukované | | |
| Random Forest | všetky | | |
| Random Forest | redukované | | |
| ... | ... | | |
