# Fáza 3 — Agregácia dát
### LoL Match Outcome Prediction

---

## Cieľ

Transformácia long formátu (1 riadok = 1 hráč) na formát zápasu (1 riadok = 1 zápas).
Každý zápas má 10 hráčov — 5 v tíme 100 a 5 v tíme 200. Výsledok: ~N/10 riadkov.

---

## 3.1 — Typy agregácií

Pred samotnou agregáciou roztriediť každý stĺpec do jednej z nasledujúcich kategórií:

### Aditívne metriky — `sum()`
Metriky kde celková hodnota tímu je súčet hodnôt hráčov. Má zmysel agregovať súčtom.
Príklad: `kills` — tím celkovo zabil X hráčov.

### Rate metriky — `mean()`
Metriky vyjadrené ako rate alebo priemer — sumovanie by nedávalo zmysel.
Príklad: `gold_per_minute` — priemerný GPM tímu.

### Konštantné v rámci skupiny — `first()`
Metriky ktoré majú rovnakú hodnotu pre všetkých hráčov v tíme alebo v zápase — stačí vziať prvú hodnotu.
Príklad: `game_duration`, `win`, `queue_id`.

### Distribučné metriky — `std()`
Selektívne pridať štandardnú odchýlku pre metriky kde je rozptyl v rámci tímu informatívny — zachytáva či má tím jedného silného carry hráča alebo rovnomerné rozdelenie výkonu.
Príklad: `gold_earned_std` — vysoké std znamená že jeden hráč dominuje ekonomike tímu.

Nie pre každú metriku má std zmysel — vynechať pre konštantné metriky (std = 0) a metriky s prirodzene malými hodnotami kde std nenesie informáciu.

---

## 3.2 — Štruktúra výsledného datasetu

### Prefixovanie stĺpcov
Features každého tímu dostanú prefix podľa team_id:
- Tím 100 → `t1_`
- Tím 200 → `t2_`

Príklad: `t1_kills_sum`, `t2_kills_sum`, `t1_gold_per_minute_mean`, `t1_gold_earned_std`.

### Target
`win` je konštantný v rámci tímu — vziať `first()` pre tím 1. Target bude `t1_win` (0/1).
Tím 2 výsledok je inverzia — nie je potrebné ho duplikovať ako extra stĺpec.

### Symetria
Dôležité konzistentne definovať kto je t1 a kto t2 — model sa musí naučiť že vysoké `t1_kills` a nízke `t2_kills` znamená výhodu pre t1. Náhodné alebo nekonzistentné priradenie by model zmiatlo.

---

## 3.3 — Absolútne hodnoty vs. diferencie

Ponechávame **absolútne hodnoty** oboch tímov — nie diferenčné features.

Dôvod: diferencia `t1_gold - t2_gold` je lineárna kombinácia `t1_gold` a `t2_gold` — pridanie diferencie by vytvorilo multikolinearitu bez pridanej informácie. Model s absolútnymi hodnotami oboch tímov dokáže sám naučiť relatívnu výhodu cez koeficienty. Navyše absolútne hodnoty zachovávajú kontext zápasu — zápas kde oba tímy majú 60k gold je iný ako zápas kde oba tímy majú 30k gold, aj keď diferencia je rovnaká.

---

## 3.4 — Stĺpce ktoré sa neagregujú

Niektoré stĺpce sa pri agregácii vyhadzujú pretože stratia zmysel na úrovni zápasu alebo sú po čistení (Fáza 2) už vyhodené:
- Identifikátory (`match_id` — vyhodený v Fáze 2)
- Pozičné stĺpce (`team_position`, `role`, `lane`) — každý tím má vždy jedného hráča na každej pozícii, agregácia nedáva zmysel
- `team_id` — stáva sa prefixom, nie featurou
- Stĺpce vyhodené pre leakage alebo nulovú varianciu (Fáza 2)

---

## 3.5 — Kontrola výsledku

Po agregácii overiť:
```python
df_agg.shape                          # očakávaný počet riadkov ~N/10
df_agg.isnull().sum().sum() == 0      # žiadne chýbajúce hodnoty
df_agg['t1_win'].value_counts()       # class balance ~50/50
df_agg.head()                         # vizuálna kontrola štruktúry
```

Zdokumentovať finálny počet features a riadkov pred tým ako pokračujeme na Fázu 4 (train/test split a scaling).

> Nasleduje **Fáza 4 — Train/test split a scaling**.
