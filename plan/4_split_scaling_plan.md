# Fáza 4 — Train/Test Split a Scaling
### LoL Match Outcome Prediction

---

## 4.1 — Train/test split

Rozdeliť agregovaný dataset na trénovaciu a testovaciu sadu pred akýmkoľvek fitovaním — scaling, feature selection aj modely sa fitujú výlučne na train sete.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

- **Pomer:** 80/20 — pri ~2 450 riadkoch to dáva ~1 960 train a ~490 test
- **Stratifikácia:** podľa targetu `t1_win` — zachová class balance v oboch setoch
- **random_state:** nastaviť fixnú hodnotu pre reprodukovateľnosť (požiadavka zadania)

Overiť class balance po splite:
```python
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
```

---

## 4.2 — Scaling

Numerické features sa štandardizujú pomocou `StandardScaler` (z-score normalizácia).

**Kľúčový princíp:** scaler sa fituje **iba na train sete** a následne transformuje oba sety. Fitovanie na celom datasete by spôsobilo data leakage — informácia z test setu by ovplyvnila trénovanie.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Ktoré modely scaling potrebujú
- **Áno:** LogReg, LDA, KNN — citlivé na scale features
- **Nie:** Random Forest, Gradient Boosting — tree-based modely sú invariantné voči scale

Pre jednoduchosť pipeline scalovať všetko a použiť scaled verziu pre všetky modely.

### Prečo je scaling kritický pre KNN
KNN počíta euklidovskú vzdialenosť medzi bodmi. Feature s rozsahom 0–50 000 (napr. `gold_earned`) by dominovala nad featurou s rozsahom 0–10 (napr. `kills`) — model by prakticky ignoroval malé features. Po `StandardScaler` má každá feature priemer 0 a štandardnú odchýlku 1, takže všetky features prispievajú rovnomerne k vzdialenosti. Toto zdôvodniť explicitne v notebooku.

---

## 4.3 — Cross-validation

Jednoduchý train/test split môže byť nestabilný pri menšom datasete. Po agregácii máme ~2 450 riadkov čo nie je veľa — zvážiť k-fold cross-validation pre robustnejšie odhady výkonu.

Odporúčaný prístup:
- **Výber a ladenie modelov:** k-fold CV na train sete (napr. k=5 alebo k=10)
- **Finálne hodnotenie:** na test sete ktorý sa počas celého procesu nepoužíval

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
```

---

## 4.4 — Dôležité pre Scenár B

Feature selection (LASSO, RFE) sa musí fitovať **iba na train sete** — rovnaký princíp ako scaling. Vyberať features na základe celého datasetu vrátane test setu by znamenalo leakage.

Správne poradie v pipeline:
1. Train/test split
2. Scaling (fit na train, transform oba)
3. Feature selection (fit na train)
4. Trénovanie modelov (na train)
5. Vyhodnotenie (na test)

Odporúča sa použiť `sklearn.pipeline.Pipeline` ktorý toto poradie vynúti automaticky a zabráni náhodnej chybe.

---

## 4.5 — Kontrola výsledku

```python
print(X_train.shape, X_test.shape)    # správny pomer
print(y_train.mean(), y_test.mean())  # class balance ~0.5 v oboch
```

Zdokumentovať finálne rozmery train a test setu pred pokračovaním na scenáre.

> Nasleduje **Fáza 5 — Scenár A** a **Fáza 6 — Scenár B**.
