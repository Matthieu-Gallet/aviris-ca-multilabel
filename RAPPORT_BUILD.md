# Rapport de construction — AVIRIS-C Californie, multilabel (2026-11-25, version à 155 bandes)

Release locale : `BERTHA:~/projets/aviris-ca-multilabel-release` (6,6 Gio / 7,1 Go, 10 shards
Parquet de 0,69 à 0,74 Go). Rien n'a été publié ni écrit sous `/DATA`.

## Commande exécutée

```bash
OMP_NUM_THREADS=1 nice -n 10 uv run aviris-ml build --stage all \
  --from-existing /DATA/DATASET/HYPERSPECTRAL/AVIRIS-C_CA_TS14_MIX \
  --headers /DATA/_staging/aviris_v2_build/hdr \
  --cdl-cache ~/projets/aviris-ca-multilabel-work/existing/cdl_counts_mix2034.npz \
  --work ~/projets/aviris-ca-multilabel-work --release ~/projets/aviris-ca-multilabel-release --workers 3
```

Durées : passe QC sur les 2034 cubes en 75 s ; export en ~2 min (3 processus).

**Reproductibilité.** La même commande, réduite à `--from-existing`, a été relancée dans un
répertoire de travail vide. Elle utilise alors les fichiers de référence embarqués dans
`src/aviris_ml/data/` : table des bandes, comptes CDL et lignes de vol. Les 24 fichiers de la
release obtenue sont **identiques octet par octet** à ceux de la release livrée, shards Parquet
compris. Cette copie de test a ensuite été supprimée.
`cdl_counts_mix2034.npz` est extrait de `phase2/work/cand_fine.npz` (comptes CDL 2008-2021 pour
40 groupes, cache de l'ancienne construction). Il reproduit exactement `props_by_year.npz`
(écart max 3e-8).

## Preuve du chemin réseau complet (2 fenêtres, 5 dates)

* Téléchargement : ORNL DAAC, une requête Range par date, 2 requêtes en parallèle, 153 Mo lus
  en 16 s. Les 5 cas couvrent uint16 BIP 2013, int16 BIP 2014 et float32 BIL 2016/2018, ainsi
  que des lignes tournées de 28 à 38°.
* Comparaison avec MIX (`tools/compare_with_source.py`) : les 5 couples fenêtre-date sont
  **identiques au bit près**. La ligne de vol et le motif de NaN sont les mêmes, et l'écart max
  vaut 0.
* CDL : relu sur Planetary Computer pour 2 fenêtres et 14 ans. Les comptes sont identiques au cache.
* Correctif fait pendant la preuve : la première version renvoyait des 401, car l'en-tête
  d'authentification n'était pas réappliqué à la redirection URS. On utilise maintenant la
  gestion standard de `requests` avec `.netrc`. Les 401, 403 et 404 ne sont plus retentés.

## Bandes

* Masque de base : 365-414, 1303-1473, 1782-2028 et 2447-2496 nm retirés, soit 166 bandes. C'est
  identique au `clean` de MIX.
* Statistique de pic : écart d'une bande isolée par rapport à ses deux voisines dans le spectre
  moyen de la fenêtre, après retrait de la médiane temporelle de la fenêtre. Un pic est compté
  si cet écart dépasse 10 % de max(réflectance, 0,05). Mesure faite sur 24 211 couples QC-ok.
* Médiane du taux : 0,0006. Seuil : 20 × médiane = 0,012. **9 bandes retirées** (taux entre
  parenthèses) :
  * 424 nm (0,119)
  * 2038 nm (0,062)
  * 938 nm (0,059)
  * 1483 nm (0,047)
  * 2048 nm (0,030)
  * 2417 nm (0,022)
  * 2407 nm (0,017)
  * 947 nm (0,013)
  * 1742 nm (0,013)
* Le critère de pic seul laisse 157 bandes. On retire ensuite explicitement les ailes de la bande
  de vapeur d'eau à 940 nm (plage 920-965 nm, soit 928 et 957 nm en plus de 938 et 947 nm).
* **Résultat final (validé) : 155 bandes.** La colonne `removed_by` de `bands.csv` donne la raison
  du retrait : `base_mask`, `spike` ou `water_vapour_940`.
* Pic à 1750 nm : il porte sur une seule bande, 1742 nm. Des valeurs jusqu'à 2·10^4 y
  apparaissent sur ~1,2 % des couples, surtout sur les vols 2017-2019. Elle est retirée.
* Zone 2350-2370 nm : taux de 0,007 (2357 nm), dû surtout aux vols du 2020-10-13. Elle reste
  sous le seuil et ces bandes sont gardées.
* Après nettoyage (155 bandes), la statistique recalculée sur le jeu final de bandes a un max de
  0,027 à 434 nm, puis 0,012 à 2427 nm. Avant le retrait explicite des ailes à 940 nm, ce max
  montait à 0,088 à 928 nm et à 0,048 à 957 nm : c'est ce qui a motivé ce retrait.
* Sensibilité au facteur de seuil (nombre de bandes gardées) :

  | facteur | bandes gardées | effet |
  |---|---|---|
  | ×10 | 152 | retire en plus 434, 655, 1263, 2357 et 2437 nm |
  | ×15 | 154 | |
  | ×20 | 157 | choix retenu, puis 155 après le retrait des ailes à 940 nm |
  | ×25 | 159 | 947 et **1742 nm** sont gardées |
  | ×30 | 160 | |

  Le retrait de 1742 nm (le pic vers 1750 nm) ne tient qu'à ~5 % au-dessus du seuil. Il dépend
  donc du facteur choisi.
* Valeurs (bandes gardées, couples QC-ok) :
  * 1,2 % des valeurs sont < 0 ;
  * 0,025 % sont > 1 ;
  * 641 couples ont au moins une valeur hors [-0,2 ; 2]. Ce sont surtout des bandes SWIR
    2130-2390 nm sur les vols f180625, f140604, f140818 et f180816.
  * Colonne `outlier_frac` : 4,2 % des dates QC-ok ont ≥ 1 pixel hors plage. Ces valeurs ne sont
    pas modifiées.
* NaN :
  * 3,8 % des pixels sur les dates acquises ;
  * 0,05 % sur les dates QC-ok ;
  * 4,6 % en comptant le bourrage (171 fenêtres ont moins de 14 dates).

## QC

Ce sont les règles de MIX, réimplémentées : **0 écart** avec `meta/date_mask.npz`
(24 211 couples ok sur 28 245 acquis). Motifs, sachant qu'un même couple peut en cumuler
plusieurs :

* plus de 20 % de NaN : 1155 ;
* neige : 525 ;
* scène très claire : 876 ;
* spectre nul : 1918.

## Étiquettes (seuil 10 %, 13 classes)

* Par date : le CDL de l'année d'acquisition. Les dates de bourrage valent -1.
* Par fenêtre : moyenne des proportions sur les années distinctes des dates de la fenêtre.
  * Comparaison avec `p_*` de MIX (moyenne sur toutes les années de vol) : écart moyen 0,0002,
    et seulement 8 fenêtres ont un ensemble d'étiquettes différent.
  * Comparaison avec l'union des étiquettes par date : 531 fenêtres diffèrent (621 bascules).
  * 766 fenêtres voient au moins une classe changer d'une date à l'autre, surtout des proportions
    proches de 10 %.
* Étiquettes par fenêtre : 1 → 333, 2 → 876, 3 → 670, 4 → 147, 5 → 8 (moyenne 2,32). Aucune
  fenêtre n'est sans étiquette.
* Positifs par fenêtre :

  | classe | positifs |
  |---|---|
  | almond | 162 |
  | shrubland | 587 |
  | water | 195 |
  | evergreen_forest | 402 |
  | mixed_forest | 206 |
  | grassland_pasture | 376 |
  | barren | 148 |
  | developed_high | 249 |
  | developed_low | 656 |
  | developed_medium | 651 |
  | developed_open | 599 |
  | grapes | 194 |
  | other | 298 |

* Co-occurrences principales :
  * developed_low + developed_medium : 485 ;
  * low + open : 401 ;
  * medium + open : 278 ;
  * shrubland + evergreen_forest : 244 ;
  * high + medium : 241, soit 97 % des fenêtres developed_high.
  * 7 paires de classes n'apparaissent jamais ensemble.

## Plis

* 616 blocs de 10,2 km répartis en 5 plis de 413, 402, 405, 398 et 416 fenêtres.
* Aucun bloc ni aucune fenêtre n'appartient à deux plis. Les shards contiennent les 2034 fenêtres,
  chacune une seule fois.
* Équilibre des classes : l'écart relatif max entre plis est de 10 % (evergreen_forest).
* Proximité entre plis : 1,0 % des fenêtres ont une voisine d'un autre pli à 600 m, et 2,8 % à
  moins de 1 km. La distance médiane vaut 4,8 km.
* Accord avec `fold5` de MIX : 24 %. C'est une nouvelle répartition, équilibrée sur les
  multilabels.

## Vérifications du chargement

* `load_dataset(<release>)` donne les splits fold_0 à fold_4. Les configurations
  `train_val_test`, `windows`, `dates` et `cdl_by_year` se résolvent aussi.
* `scripts/sanity_check.py . 8` renvoie OK : formes (14, 155, 32, 32), comptes conformes à
  `classes.csv`, et 98/98 covariances de Ledoit-Wolf 155 × 155 SPD (rétrécissement médian 0,0026).
* `scripts/load_example.py` renvoie OK.
* `pytest` : 17 tests passent sur données synthétiques.
* `checks.json` :
  * le minimum de pixels CDL cartographiés sur les années utilisées vaut 323 (seuil 320) ;
  * aucune date acquise n'a d'étiquette manquante ;
  * les bandes déclarées dans les shards correspondent à `bands.csv` (155), et la forme du cube
    est (14, 155, 32, 32).

## Décisions (validées le 2026-11-25)

1. **Étiquette fenêtre** : moyenne des proportions sur les années distinctes des dates (conservée).
2. **Bandes** : 155. Règle : masque de base, puis critère de pic ×20, puis retrait explicite de la
   bande de vapeur d'eau à 940 nm (920-965 nm). 2357-2367 nm est gardé.
3. **Valeurs aberrantes** : conservées telles quelles et signalées (`outlier_frac`, colonnes de
   `bands.csv`).
4. **Licences** : jeu de données sous CC BY 4.0 (`license: cc-by-4.0`). `LICENSE.md` exige de citer
   NASA/JPL AVIRIS-Classic L2 via l'ORNL DAAC et l'USDA NASS CDL. Code sous MIT (`LICENSE`).
   La citation BibTeX reste à compléter.
5. **Plis** : la nouvelle répartition équilibrée sur les multilabels est conservée.
