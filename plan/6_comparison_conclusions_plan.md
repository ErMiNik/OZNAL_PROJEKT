# Fáza 7 — Porovnanie a Závery
### LoL Match Outcome Prediction

---

## Cieľ

Prepojiť výsledky EDA, Scenára A a Scenára B do konzistentných záverov. Nejde len o tabuľku čísel — cieľom je vysvetliť čo výsledky znamenajú, prečo niektoré modely fungujú lepšie a čo to hovorí o dátach a doméne.

---

## 7.1 — Overenie hypotéz z EDA

Pre každú hypotézu z EDA zdokumentovať či bola potvrdená alebo vyvrátená na základe výsledkov modelovania:

| Hypotéza | Potvrdená? | Dôkaz |
|----------|------------|-------|
| H1: gold_per_minute > kills ako prediktor | | Feature importance Scenár B, koeficienty LogReg |
| H2: vision_score signifikantný po kontrolovaní gold | | Zostal vo feature sete po LASSO/RFE? |
| H3: damage stráca signifikantnosť po zarátaní gold | | Feature importance, LASSO koeficienty |
| H4: baron+dragon > kills | | Feature importance |
| H5: deaths koreluje silnejšie s prehrou ako kills s výhrou | | Koeficienty LogReg, RF importance |

Ak hypotéza nebola potvrdená — diskutovať prečo. Prekvapivé výsledky sú rovnako hodnotné ako potvrdené hypotézy.

---

## 7.2 — Porovnanie modelov: výkon

Záverečná porovnávacia tabuľka všetkých modelov na plnom aj redukovanom feature sete:

| Model | Typ | Features | Accuracy | ROC-AUC | Train AUC | Overfit? |
|-------|-----|----------|----------|---------|-----------|---------|
| LogReg | param. | všetky | | | | |
| LogReg | param. | redukované | | | | |
| LDA/QDA | param. | všetky | | | | |
| LDA/QDA | param. | redukované | | | | |
| Naive Bayes | param. | všetky | | | | |
| Naive Bayes | param. | redukované | | | | |
| KNN | neparam. | všetky | | | | |
| KNN | neparam. | redukované | | | | |
| Random Forest | neparam. | všetky | | | | |
| Random Forest | neparam. | redukované | | | | |
| Gradient Boosting | neparam. | všetky | | | | |
| Gradient Boosting | neparam. | redukované | | | | |

---

## 7.3 — Kľúčové diskusné body

### Výkon vs. explainability
Porovnať najlepší parametrický a najlepší neparametrický model. Je rozdiel v ROC-AUC prakticky relevantný? Napríklad rozdiel 0.02 v AUC pravdepodobne neospravedlňuje stratu interpretovateľnosti. Diskutovať kedy by sme uprednostnili jeden prístup pred druhým.

### Predpoklady vs. realita
Splnili parametrické modely svoje predpoklady (normalita pre LDA, linearita pre LogReg)? Ak nie — bol výkon napriek tomu dobrý? Niektoré modely sú robustné voči porušeniu predpokladov — zdôvodniť prečo.

### Overfitting
Porovnať train vs. test AUC pre každý model. Mali neparametrické modely väčší rozdiel? Pomohla regularizácia (LASSO, Ridge) alebo redukcia features (Scenár B) znížiť overfitting?

### Efekt feature selection
Zlepšil sa výkon po redukcii features? Ktoré modely profitovali najviac? Očakávame že KNN profituje viac ako Random Forest — potvrdiť alebo vyvrátiť. Koľko features stačí na dobrý výkon?

### Zhoda metód feature selection
Ak sa LASSO, RF importance a RFE zhodujú na rovnakých top features, môžeme ich označiť za robustné prediktory výhry nezávisle od metodiky. Ak sa líšia — diskutovať prečo (lineárny vs. nelineárny model, multikolinearita).

---

## 7.4 — Odporúčaný model

Na základe výsledkov odporučiť jeden model pre tento problém a zdôvodniť výber. Zvážiť:
- Výkon (ROC-AUC)
- Interpretovateľnosť (sú koeficienty / importance zmysluplné?)
- Robustnosť (stabilita cez CV, overfitting)
- Praktická použiteľnosť (rýchlosť, jednoduchosť implementácie)

---

## 7.5 — Limitácie projektu

Zdokumentovať čo projekt neobsahuje a čo by výsledky mohlo ovplyvniť:
- Dataset pokrýva len konkrétne časové obdobie — patch verzia hry sa mení, meta sa vyvíja
- Pracujeme s end-of-game dátami — model nie je použiteľný pre predikciu počas hry
- Veľkosť datasetu po agregácii (~2 450 zápasov) — väčší dataset by mohol zlepšiť stabilitu výsledkov
- Chýbajúce features ktoré by mohli byť dôležité — napr. rank hráčov, server región

---

## 7.6 — Záver z pohľadu domény

Čo výsledky hovoria o League of Legends z herného hľadiska — toto je časť ktorá robí projekt zaujímavým aj pre niekoho kto štatistiku neovláda a je dôležitá pre one-pager.

Príklady záverov ktoré môžu vyplynúť:
- Je ekonomika (gold) skutočne dôležitejšia ako combat (kills)?
- Nakoľko dôležitý je vision control pre výhru?
- Sú objekty (baron, dragon) silnejšími prediktormi ako individuálny výkon?

Tieto závery tvoria základ one-pagera a executive summary.
