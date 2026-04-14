# EDA Plán — League of Legends Match Dataset

## Kontext a cieľ

Dataset obsahuje ~24 600 riadkov v **long formáte** (1 riadok = 1 hráč v zápase).
Cieľom projektu je **binárna klasifikácia**: predikcia výsledku zápasu (`win = True/False`)
na základe **end-of-game štatistík** — teda modelujeme post-game výkon, nie pre-game predikciu.

Výstupom EDA je:
- čistý dataset pripravený na modelovanie
- odôvodnený zoznam vylúčených stĺpcov
- potvrdené / vyvrátené hypotézy
- motivácia pre výber features v Scenári A a B

---

## Framing — čo predikujeme

> **Dôležitá poznámka do notebooku:** Pracujeme s end-of-game dátami. Model sa teda neučí
> predikciu výsledku *pred* zápasom, ale identifikuje **ktoré metriky výkonu počas hry
> korelujú s víťazstvom**. Stĺpce ako `gold_earned`, `kills`, `damage` nie sú data leakage
> — sú to legitímne výkonnostné metriky. Kauzalitu nevyvodzujeme.

---

## Blok 1 — Základné preskúmanie datasetu

### Čo robiť
- `df.shape`, `df.dtypes`, `df.head()`, `df.describe()`
- Počet unikátnych zápasov: `df['match_id'].nunique()`
- Overiť že každý zápas má presne 10 riadkov (5+5 hráčov)
- Distribúcia `queue_id` — ponecháme **iba ranked solo queue (queue_id = 420)**
- Počet unikátnych championov, distribúcia pozícií (`team_position`)
- Skontrolovať chýbajúce hodnoty: `df.isnull().sum()`

### Čo napísať do notebooku
- Stručný popis datasetu (zdroj, počet zápasov, obdobie)
- Zdôvodnenie filtrovania len na queue_id = 420
- Komentár k formátu dát (long format, 1 riadok = 1 hráč)

---

## Blok 2 — Leakage check

### Čo robiť
Identifikovať stĺpce ktoré **deterministicky alebo takmer deterministicky** kódujú výsledok.

#### Priamy leakage — vyhodiť
| Stĺpec | Dôvod |
|--------|-------|
| `team_early_surrendered` | Ak tím surrenderoval, prehral — priama informácia o výsledku |
| `game_ended_in_early_surrender` | Priamo kóduje spôsob ukončenia hry |

#### Hraniční kandidáti — ponechať, ale komentovať
| Stĺpec | Poznámka |
|--------|----------|
| `nexus_kills` | Takmer leakage — ale pri surrenderi môže byť 0 aj u víťaza |
| `inhibitor_kills` | Nie je leakage — môžeš zničiť inhibitor a stále prehrať |
| `turret_kills` | Nie je leakage — silný prediktor, ale nie deterministický |
| `game_ended_in_surrender` | Hovorí len *či* sa surrenderoval, nie *kto* — nie leakage |

#### Identifikátory — vyhodiť (nie features)
- `match_id`, `champion_skin_id`, `summoner_1_id`, `summoner_2_id`

### Čo napísať do notebooku
- Explicitne zdôvodniť každý vylúčený stĺpec
- Vysvetliť rozdiel medzi priamym leakage a silným prediktorom
- **Toto je sekcia na ktorú sa tutori pýtajú na skúške**

---

## Blok 3 — Univariátna analýza

### Čo robiť
- Histogramy + boxploty pre kľúčové numerické features:
  `gold_earned`, `kills`, `deaths`, `assists`, `total_damage_dealt_to_champions`,
  `vision_score`, `kda`, `gold_per_minute`, `dragon_kills`, `baron_kills`
- Identifikácia outlierov (napr. hráč s 34 kills — reálny, ale extrémny)
- Rozhodnutie: outliere necháme alebo caps? → zdôvodniť v notebooku
- **Class balance**: `df['win'].value_counts(normalize=True)`
  → očakávame ~50/50 (každý zápas má jedného víťaza a jedného porazeného)
- Distribúcia kategorických premenných: `team_position`, `champion_name`

### Čo napísať do notebooku
- Komentár k distribúciám — sú normálne? zošikmené?
- Komentár k outlierom a rozhodnutie ako s nimi naložiť
- Potvrdenie class balance → nie je potrebný oversampling

---

## Blok 4 — Bivariátna analýza

### Čo robiť

#### Korelácia features s targetom
- Pearsonova korelácia každej numerickej premennej s `win`
- Heatmapa top 20 najkorelovanejších features
- Point-biserial korelácia pre binárny target

#### Skupinové porovnanie (win vs. loss)
- Boxploty / violin ploty pre kľúčové features rozdelené podľa `win`
- Porovnanie: `gold_per_minute`, `vision_score`, `kills`, `dragon_kills`, `baron_kills`
- T-test alebo Mann-Whitney U test pre štatistickú signifikantnosť rozdielov

#### Multikolinearita medzi features
- Koreláčná matica numerických features
- Identifikácia silne korelovaných párov (napr. `kills` ↔ `kda`, `gold_earned` ↔ `gold_per_minute`)
- Relevantné pre Scenár B — feature selection

### Čo napísať do notebooku
- Interpretácia heatmapy — ktoré features sú najsilnejšie korelované s výhrou
- Komentár k multikolinearite — dôsledky pre modelovanie
- Prepojenie na hypotézy (potvrdenie / vyvrátenie)

---

## Blok 5 — Hypotézy a ich vyhodnotenie

Každú hypotézu formulujeme pred analýzou, overíme štatisticky a výsledok okomentujeme.

---

### H1 — Ekonomika > combat
> Priemerný `gold_per_minute` tímu je silnejší prediktor výhry ako priemerný `kills`.

**Ako overiť:**
- Porovnaj Pearsonovu koreláciu `gold_per_minute` vs. `kills` s targetom `win`
- Skupinový boxplot oboch features podľa `win`
- V modeloch: porovnaj feature importance

**Očakávaný výsledok:** gold_per_minute bude silnejšie korelovaný

---

### H2 — Vision rozhoduje
> Tímy s vyšším celkovým `vision_score` vyhrávajú častejšie, aj po kontrolovaní gold a kills.

**Ako overiť:**
- Korelácia `vision_score` s `win`
- Parciálna korelácia po kontrolovaní `gold_per_minute`
- Logistická regresia len s `vision_score` ako feature → baseline AUC

**Očakávaný výsledok:** vision_score bude signifikantný aj po kontrolovaní ekonomiky

---

### H3 — Damage je slabší prediktor ako sa zdá
> `total_damage_dealt_to_champions` nie je signifikantný prediktor po zarátaní `gold_per_minute`.

**Ako overiť:**
- Porovnaj koreláciu `damage` vs. `gold_per_minute` s `win`
- VIF (Variance Inflation Factor) na odhalenie multikolinearity medzi nimi
- Feature importance v modeloch zo Scenára A

**Očakávaný výsledok:** damage bude menej dôležitý keď gold je v modeli

---

### H4 — Objekty > kills
> `dragon_kills` + `baron_kills` tímu predikujú výhru lepšie ako celkový počet kills.

**Ako overiť:**
- Porovnaj korelácie: `dragon_kills`, `baron_kills`, `kills` s `win`
- Skupinové porovnanie — ako často vyhráva tím s viac baronmi?
- Logistická regresia: porovnaj AUC modelu len s objectives vs. len s kills

**Očakávaný výsledok:** baron_kills bude silnejší prediktor ako kills

---

### H5 — Asymetria deaths vs. kills
> Priemerný `deaths` tímu koreluje silnejšie s prehrou ako priemerný `kills` koreluje s výhrou.

**Ako overiť:**
- Porovnaj absolútne hodnoty korelácie: `|corr(deaths, win)|` vs. `|corr(kills, win)|`
- Vizualizácia: distribúcia deaths u víťazov vs. porazených
- Interpretácia: death = stratený gold + čas respawnu

**Očakávaný výsledok:** deaths budú silnejšie (negatívne) korelované s win

---

## Čo robiť s výstupmi EDA

Po dokončení EDA by malo byť jasné:

1. **Zoznam features na vyhodenie** (leakage + identifikátory) — odôvodnený
2. **Zoznam features na encoding** (`team_position`, prípadne `champion_name`)
3. **Zoznam features na scaling** (všetky numerické pred modelmi citlivými na scale: LogReg, KNN, LDA)
4. **Potvrdené / vyvrátené hypotézy** — každá s konkrétnym číslom (korelácia, p-value)
5. **Motivácia pre feature selection** v Scenári B — ktoré skupiny features sa zdajú redundantné

---

## Poznámka k formátu dát

EDA prebieha na **long formáte** (1 hráč = 1 riadok).
Pred Scenármi A a B nasleduje **agregácia na úroveň tímov** (1 zápas = 1 riadok),
kde sa vypočítajú tímové súčty/priemery. Baseline model sa trénuje ešte pred agregáciou
na pôvodnom long formáte pre porovnanie.
