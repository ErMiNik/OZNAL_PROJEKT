# Fáza 2 — Čistenie a Encoding
### LoL Match Outcome Prediction

---

## 2.1 — Stĺpce na vyhodenie

### Leakage stĺpce
Identifikované v EDA — stĺpce ktoré deterministicky alebo takmer deterministicky kódujú výsledok zápasu. Každé rozhodnutie zdôvodniť v notebooku.

### Identifikátory
Stĺpce ktoré slúžia len ako ID záznamu a nemajú predikčnú hodnotu (napr. `match_id`). Skontrolovať všetky stĺpce s vysokým počtom unikátnych hodnôt.

### Stĺpce s nulovou / takmer nulovou varianciou
Skontrolovať programaticky — stĺpce kde takmer všetky hodnoty sú rovnaké nenesú informáciu pre model.
```python
df.var(numeric_only=True).sort_values()
df.nunique().sort_values()
```
Príklad: `unreal_kills` — overiť či má nenulové hodnoty.

### Redundantné odvodené stĺpce
Stĺpce ktoré sú lineárnou kombináciou iných stĺpcov spôsobujú multikolinearitu. Ponechať buď pôvodné alebo odvodené, nie oboje. Identifikovať pomocou koreláčnej matice z EDA.

Príklad: `kda` je odvodené z `kills`, `deaths`, `assists` — ak máme pôvodné, `kda` je redundantné.

### Duplicitné stĺpce
Overiť stĺpce ktoré môžu merať tú istú vec v iných jednotkách alebo za iný časový úsek.
```python
df.corr() — stĺpce s koreláciou = 1.0 sú duplicitné
```

### Stĺpce ktoré stratia zmysel po agregácii
Niektoré stĺpce majú zmysel na úrovni hráča ale nie na úrovni zápasu — identifikovať pred agregáciou a rozhodnúť či vyhodiť alebo agregovať inak.

---

## 2.2 — Chýbajúce hodnoty

Najprv zmapovať rozsah problému:
```python
df.isnull().sum().sort_values(ascending=False)
df.isnull().mean()  # podiel chýbajúcich hodnôt per stĺpec
```

Stratégia závisí od podielu chýbajúcich hodnôt a príčiny:
- **Malý podiel (<1–5%)** — vyhodiť riadky alebo imputovať
- **Väčší podiel** — imputácia medianom (numerické) alebo modusom (kategorické), prípadne vyhodiť stĺpec
- **Štruktúrálne chýbajúce hodnoty** — hodnota chýba zo zmysluplného dôvodu (napr. žiadny ban), imputovať sentinel hodnotou a zdôvodniť

Príklad: ban stĺpce môžu mať chýbajúce hodnoty ak zápas nemal plný draft.

Každé rozhodnutie zdôvodniť v notebooku — prečo imputujeme a nie vyhadzujeme, alebo naopak.

---

## 2.3 — Ďalšie všeobecné situácie pri čistení

### Nekonzistentné dátové typy
Stĺpce ktoré by mali byť numerické ale sú uložené ako string, alebo boolean uložený ako 0/1 vs. True/False. Skontrolovať `df.dtypes` a opraviť.

### Outliere
Rozhodnutie z EDA — ak sme identifikovali extrémne hodnoty, tu ich riešime. Možnosti: vyhodiť riadky, capping (nahradiť percentilom), ponechať a zdôvodniť prečo sú reálne.

### Nekonzistentné kategórie
Kategorické stĺpce môžu mať rôzne varianty toho istého (napr. medzery, rôzne veľké písmená). Skontrolovať `df['col'].unique()` pre každý kategorický stĺpec.

### Duplikátne riadky
```python
df.duplicated().sum()
```
Ak existujú, rozhodnúť sa či sú chybou alebo reálnymi záznami.

---

## 2.4 — Encoding

### Boolean stĺpce
Overiť že sú v správnom formáte 0/1 a správnom dátovom type. Príklad: `first_blood_kill`.

### Kategorické stĺpce s nízkym počtom kategórií
One-Hot Encoding — vhodné pre stĺpce s malým počtom unikátnych hodnôt (do ~10–15). Pozor na dummy variable trap — použiť `drop_first=True`.

### Kategorické stĺpce s vysokým počtom kategórií
Priamy OHE by vytvoril príliš veľký feature space. Možnosti:
- Target encoding
- Vyhodiť stĺpec ak neobsahuje relevantné info pre hypotézy
- Príklad: `champion_name` má 150+ hodnôt — zvážiť vyhodenie

### Ordinálne stĺpce
Ak existujú stĺpce s prirodzeným poradím, použiť ordinálny encoding namiesto OHE.

### Target
`win` — ponechať ako 0/1, neencódovať.

---

## 2.5 — Kontrola výsledku

Po čistení a encodingu overiť pred pokračovaním na agregáciu:
```python
df.isnull().sum().sum() == 0      # žiadne chýbajúce hodnoty
df.duplicated().sum() == 0        # žiadne duplikáty
df.var(numeric_only=True) > 0     # žiadne konštantné stĺpce
df.dtypes                         # správne typy
df.shape                          # zdokumentovať počet zostávajúcich stĺpcov
```

Zdokumentovať finálny zoznam stĺpcov ktoré idú do agregácie — zdôvodniť každé rozhodnutie.

> Nasleduje **Fáza 3 — Agregácia** kde sa vyčistený long format transformuje na formát zápasu.
