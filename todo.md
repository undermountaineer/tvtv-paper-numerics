# TODO — tvtv-paper-numerics

Revisió de tot el projecte, 19-11-2026. Els canvis de la ronda anterior
(resolucions en paral·lel; solver amb residus gratuïts, sense legacy, hash AST)
són a la branca `solver-free-residuals`; els tests passen.

**Cal optimitzar-lo més?** Ara no, primer cal mesurar. Les dues palanques grans
(nuclis en paral·lel i el cost dels residus) ja hi són. El que queda al codi
dona ≤ 1.5–2×. El cost de fons és el nombre d'iteracions (fins a 128–256
externes, i el bloc $v$ a 512), i això és una qüestió algorítmica i científica,
no de codi.

Etiquetes:
- **[cache]** toca `SOURCE_FILES` → noves claus → es recalculen totes les resolucions. Agrupar-los.
- **[token]** només canvia el directori de resultats → cal tornar a fer `prepare` (barat); les resolucions es reutilitzen.

## 0. Bloquejant (abans de llançar res)

- [x] Passar els tests: `python -m unittest discover -s tests -v` (tots passen, 19-11-2026).
- [x] `@lru_cache` a `scientific_code_id()` (`checkpoint_cache.py:49`). **Regressió meva:** `solve_identity` (línia 66) el crida per cada fila de `plan`/`report`, i ara això vol dir parsejar l'AST de 6 fitxers cada vegada. [cache]
- [x] `stop()` (`run_original_1024.py:56`) ha de tolerar `ProcessLookupError`. Amb diversos workers, un Ctrl+C mata diversos processos, i algun pot acabar entre el `poll()` i el `killpg()`.
- [x] Fer el commit en una branca i deixar escrit al missatge que la memòria cau queda invalidada (`solver-free-residuals`, 24-11-2026).

## 1. Validar els canvis del solver

- [x] Benchmark abans/després: `scripts/benchmark_performance.py --resolution 128 --case optimized` a `ed52042` (en un worktree) i a la versió nova. S'espera ~1.5–1.7× per iteració interna i gairebé el mateix nombre d'iteracions.
  **Resultat (24-11-2026):** només 1.09× (32.2 s → 29.5 s; 19.2 → 17.5 ms per iteració interna). Iteracions idèntiques (35 externes; 617/1065 internes) i mateixa pèrdua (diferència relativa 1e-7). Fitxers a `bds/bench/`. Hipòtesi: a 128 domina la sobrecàrrega de numpy; es comprova a 256 (punt següent).
  **PC de càlcul (i5-12400, 6 nuclis, 32 GB; 26-11-2026):** 12.2 s (7.3 ms per iteració interna) amb el `.venv` (Python 3.12, numpy 2.5.5). Mateixes iteracions i pèrdua igual fins a 8e-8 relatiu; $v_0$ ja difereix en els últims bits. Els dos benchmarks del portàtil van córrer amb el Python 3.14/numpy 2.4.6 del sistema, no amb el `.venv`, així que el 2.4× barreja maquinari i versions.
- [ ] Una resolució completa a 256 ($\alpha=0.000875$) per comparar $u$, $v$, les iteracions externes i la pèrdua amb el resultat antic convergit, si encara és a `results/`.
- [ ] `--workers 2` i `--workers 4` a 256: temps real per etapa. A 512, mesurar la memòria (RSS) per worker; encara no està mesurada. Fer-ho al PC (6 nuclis físics: `--workers` 5–6).

## 2. Optimització (només si, després de mesurar, encara cal)

- [ ] Perfilar abans de tocar res: `python -m cProfile -o prof.out scripts/benchmark_performance.py --resolution 512 --fixed-inner 50 --case optimized --output results/prof`.
- [ ] Operadors amb menys temporals. Dins d'un bloc, $v$ és fix, així que es poden precalcular $c_x=v_x/(2h_x)$ i $c_y$ contigus (ara `v[..., 0]` té stride). També `out=` i multiplicar per $1/h$ en lloc de dividir. Estimat: 1.3–2×. [cache]
- [ ] `plan`/`report` fan feina repetida per cada fila:
  - `inspect` crida `problem()` dues vegades (`workflow.py:157` via `key`, i `:176`), i cada crida descomprimeix l'npz de la llavor.
  - `solve_identity` fa un sha256 de ~55 MB per fila a 512.
  - Primer cal mesurar el temps de `plan` a 512 (el límit és 120 s). Després, una cache per llavor i per clau de feina, dins del directori del token. [token]
- [ ] Controlador: omplir els slots que queden lliures en lloc d'esperar que acabi el lot sencer (comentari `ponytail` a `run_original_1024.py:107`).
- [ ] Checkpoint per temps (cada ~60 s i en aturar) en lloc de fer-ne un a cada iteració externa. A 512 són ~90 MB amb `fsync`. [cache]
- [ ] Canvis algorítmics, que alteren la trajectòria i són una decisió científica:
  - PDHG adaptatiu (Goldstein–Li–Yuan 2015) amb els residus que ja es calculen.
  - Resolucions internes inexactes a les primeres iteracions externes.
  - Warm start des de l'alpha veïna: canvia el mínim local i fa dependre el resultat de l'ordre de la cerca.
  - numba: dependència nova. CuPy, no: cap de les dues màquines té GPU NVIDIA.
  - float32, no: les toleràncies ($2.5\cdot10^{-7}$) són massa a prop de l'eps.

## 3. Neteja del codi

- [ ] Esborrar el pipeline legacy (abans, posar un `git tag`):
  - `workflow.Experiment`, `smoke`, `handoff.py`, `figures.py`
  - `scripts/run.py`, `run_meeting.py`, `meeting_report.py`, `resume_stopping_fix.py` (amb el codi nou es nega a executar-se)
  - `notebooks/proof_of_concept.ipynb`, `configs/proof_of_concept.json`
  - `data.make_sequence` (`data.py:26`) i `minimal_phantom.scene` (`:21`), que només fa servir el legacy
  - [token]
- [ ] Treure l'herència `Original1024(Experiment)`. Només aprofita `inspect`, `solve`, `key` i `input_path`; cal moure'ls a `original_1024.py`. [token]
- [ ] `report_experiment.py:45-52`: treure la referència a `STOPPING_FIX_CONTINUATION.json`.
- [ ] Hi ha tres versions de "4 fills espacials" (`workflow.spatial_children`, `original_1024.children` i la que ja s'ha esborrat de `DyadicCell`). Quedar-se només amb `original_1024.children`.
- [ ] Configs:
  - `meeting_512.json` és idèntic a `synthetic_512.json`, i `meeting_256.json` només canvia el límit d'iteracions externes.
  - `hx`/`hy` del bloc `solver` se sobreescriuen a `original_1024.py:150`.
  - `kappa` hi és dues vegades (`model` i `solver`).
  - El bloc `tighter` repeteix tot el bloc `local`.
  - Les claus no canvien, perquè la configuració efectiva és la mateixa. [token]
- [ ] `WeightedTVTVConfig.update_order` té per defecte `"v_then_u"` (`weighted_solver.py:58`), però tots els configs fan servir `"u_then_v"`. [cache]
- [ ] Format de la resta de fitxers (`operators.py`, `metrics.py`, `dyadic.py`, `weighted_energy.py`, `proximal.py`), que ara tenen un argument per línia. Amb el hash AST no invalida res, sempre que no es toquin els docstrings.
- [ ] `pyproject.toml`: el nom (`tvtv-dyadic-numerics`) i la descripció ("Minimal proof of concept") estan desactualitzats.

## 4. Tests que falten

- [ ] `spatial_norm_squared` contra el valor propi màxim de la matriu explícita en una malla petita. És el que garanteix la condició de pas.
- [ ] La fita $\|T(v)\|\le 2/h_t+\max|v_x|/h_x+\max|v_y|/h_y$, també amb la matriu explícita.
- [ ] PDHG explícit de referència per al bloc $v$. Ara només el bloc $u$ té test de referència (`test_performance.py`).
- [ ] Controlador: deduplicació per clau i aturada quan una feina no és elegible. Primer cal extreure la selecció de feines a una funció pura.

## 5. Reproductibilitat

- [x] Fixar versions: `environment.yml` no en fixa cap. Els resultats poden canviar amb numpy/scipy, i la clau de la memòria cau no inclou les versions. Generar un lock (`conda env export --no-builds` o conda-lock).
  **Fet (26-11-2026) amb uv:** `requirements-lock.txt` (`uv pip freeze` del `.venv` del portàtil). El PC s'instal·la amb `uv pip install -r requirements-lock.txt` i `uv pip install -e . --no-deps`.
- [ ] Al portàtil, `python` no és el del `.venv` (els benchmarks del 24-11 van córrer amb el Python 3.14 del sistema). Esbrinar amb quin Python es van calcular les resolucions de 256 de `results/`: si no va ser el 3.12 del `.venv`, la memòria cau no es reutilitzarà (vegeu `ast.dump`, més avall).
- [ ] Guardar les versions de Python, numpy i scipy a les metadades de cada `result.npz` (`checkpoint_cache.py:125`).
- [ ] `ast.dump` depèn de la versió menor de Python. Si es passa a 3.13, totes les claus canvien.

## 6. Documentació desactualitzada

- [ ] `README.md:124`: "The neutral 256 configuration ... identifies the same saved experiment" ja no és cert.
- [ ] README, "Installation": descriu conda i `environment.yml`, però l'entorn real és uv + `requirements-lock.txt`.
- [ ] `docs/reproducibility.md`: els àlies amb `meeting_256_stopping_fixed.json` són falsos. Cal explicar l'execució en paral·lel i `--workers`.
- [ ] `docs/performance_pass.md`, `stopping_fix.md`, `meeting_protocol.md` i `SOURCE_PROVENANCE.json` són històrics. Moure'ls a `docs/history/` o esborrar-los. `performance_pass.md` explica com aplicar un arxiu de pedaç que no existeix i conté rutes personals de Windows.
- [ ] `docs/model.md`, apartat del solver (opcional): explicar que els residus fan servir $Ku$ i $K^*p$ de la iteració anterior.

## 7. Experiments pendents (protocol)

- [ ] 256: `prepare` → `baseline` → `refine` → `compare` → `validate`.
- [ ] 512, amb pressupostos més grans (vegeu el README).
- [ ] Sensibilitat del guany a la tolerància (`local` contra `tighter`). Els guanys històrics són d'~1% (`docs/status.md`), és a dir, de la mateixa mida que l'error numèric possible.
- [ ] Comprovar si alguna $\alpha$ queda al límit de la cerca. A la versió de 128, les cel·les de dalt hi quedaven.

## 8. Toy: identificabilitat del moviment (25/26-11-2026)

Scripts: `scripts/toy_heterogeneous_noise.py` (escena 64×64, mateix quadrat i disc a cada meitat, σ=0.05 / 0.2) i `scripts/toy_v_block.py` (només el bloc $v$, amb tolerància estreta). Resultats a `results/toy_heterogeneous_noise/` i `results/toy_v_block/` (al PC).

Resultats:
- Soroll heterogeni: la meitat sorollosa tria una $\alpha$ el doble de gran, i el mapa guanya un 2–5% de pèrdua, tot a la meitat neta.
- Moviment diferent a cada meitat (`split`): la resolució conjunta es queda a la $v_0$ de l'ajust global (0.53 px d'error) per a totes les $\alpha$. Arrencant del moviment real, l'energia final és un 2.5% més baixa i la pèrdua d'imatge un 40% més baixa.
- Bloc $v$ amb la imatge neta: el mínim no depén de l'arrencada, però no és el moviment real. L'error és ≈$d$ (85–99%) tant a 0.75 com a 0.25 px per fotograma, així que no és la linealització.
- Mecanisme (vora abrupta en 1D): la derivada temporal ocupa un píxel i la diferència centrada en reparteix el gradient en tres. El cost L1 és constant per a qualsevol velocitat normal en $[0,2d]$, i un pic $v_k=2d$, $v_{k\pm1}=0$ anul·la el residu amb un cost TV de $4d\beta$ per fila. Predicció $E_{\min}\approx\beta h^2\cdot4d\cdot N_{\text{files}}\cdot N_t$: 1.25e-3 i 3.75e-3, observat 1.21e-3 i 3.61e-3.
- Conseqüència: les vores abruptes no aporten informació de velocitat al model discret. La $v$ final la decideix l'arrencada, i $u$ s'hi adapta. L'experiment principal funciona perquè allí $v_0$ és exacta per construcció; no posa a prova l'estimació del moviment.

Pendent:
- [ ] Vores suaus: `toy_v_block.py --blur 1.5` (i 0 per comparar). Mirar els perfils de `v_block.png`: pics a les vores sense blur, i el moviment real amb blur?
- [ ] Si el blur ho confirma, decidir la discretització del transport per als experiments del paper: pre-suavitzat dins del terme de transport, penalització L2/Huber del residu (identificable però esbiaixada) o *warping* amb interpolació.
- [ ] Al paper: dir que l'experiment principal no posa a prova l'estimació del moviment, o canviar l'escena (moviment no uniforme) i l'arrencada.

## 9. Paper: correccions de la revisió

Informe: `bds/overleaf/Review report BDS1.md` (23-11-2026). Ordre: primer B1, que és l'única llacuna matemàtica.

- [ ] B1: dir que $g_Q\in L^{r'}(L)=L^{n+1}(L)$ a la discussió després de `P: nontrivial canonical parameters`, perquè $\int_L g_Q\varphi$ siga finita per a $\varphi\in BV(L)$.
- [ ] C1: una línia a `E: counterexample` que mostre que cap minimitzador té $u$ constant.
- [ ] D1: la cota de l'envolupant en tot punt ($\sup_z$), no q.p. (prova de `L: weighted spatial variation stability`).
- [ ] D2: per què el límit $u$ és a $L^1(0,T;L^{n'})$ (pas 2 de `P: compactness and stability`).
- [ ] D3: citar `L: weighted spatial variation stability` a la prova de Level 3.
- [ ] D4: la constant intermèdia de l'absorció (pas 3 de `P: local rigidity`).
- [ ] D5: una línia per a les cel·les dins de $P$ (prova de `T: existence of an optimal partition`).
- [ ] D6: una altra lletra per al nombre de mostres a `R: several samples` ($m$ ja és l'índex de les successions).
- [ ] Llegibilitat: comprovar si els punts #4, #6 i #9 de `bds/notes/readability_suggestions_main_1.md` estan fets.

## 10. Paper: treure el nivell mínim N0 (TV)

Idea: `bds/latex/bds-1-tgv/bds-1-tgv.tex`, Remark 4.5(d). Amb els moments espacials de cada llesca en lloc de la mitjana espai-temps, el terme de la divergència no necessita absorció. Va abans de §11, perquè canvia la prova que la secció de TGV estén.

- [ ] Comprovar l'argument per a TV amb el competidor $(B(t_Q),\boldsymbol 0)$.
- [ ] Si funciona: reescriure `P: local rigidity` sense $N_0$ i ajustar `R: rigidity regimes`, `C: fine-cell parameter independence`, `R: explicit depth` i la intro.
- [ ] Si no funciona: escriure el perquè a la nota i llevar la Remark 4.5(d).

## 11. Paper: cas TGV

Nota: `bds/latex/bds-1-tgv/bds-1-tgv.tex` (26-11-2026). La llista del que cal comprovar és a la secció 7 de la nota.

- [ ] Revisar la nota, sobretot les constants del Lemma 4.3 i la tria de $C_R$ a la Proposició 4.4.
- [ ] Afegir els números d'enunciat de Temam (Korn–Poincaré i compacitat de $BD$) i compilar la nota.
- [ ] Integrar-la al paper com una secció curta dins de Complements: TGV definit per llesques, rigidesa i l'exemple de les rampes que giren. La bona posició, resumida.
- [ ] Actualitzar la intro (la frase de TGV, el paràgraf de resultats i el d'organització) i les paraules clau.

## 12. Paper: revista i numèrica

- [ ] Decidir la revista amb Elisa (tasca #83 de Vikunja, 04-12-2026): AMO sense numèrica, o JMIV amb una il·lustració numèrica.
- [ ] Si és JMIV: triar l'experiment a partir de §7 i §8 i escriure'n la secció, dient en què difereix el model numèric de la teoria:
  - transport penalitzat ($\kappa$) en lloc de la restricció exacta;
  - només $\varrho=0$;
  - cel·les només espacials, en tot l'interval de temps;
  - només s'aprén $\alpha$.

## 13. Paper: abans d'enviar

- [x] A2: títol nou, "Optimal dyadic partitions for parameter learning in motion-compensated denoising and inpainting" (26-11-2026).
- [ ] Escriure el resum (ara és buit).
- [ ] A1: llevar `showlabels`, el text en taronja i els `\vi`, i omplir els agraïments (finançament de l'SFB?).
- [ ] A3: activar les adreces dels autors (el bloc `\iffalse … \fi`).
- [ ] A4: any i edició de `evans2025measure` a `references.bib`.
- [ ] Decidir si es manté `MarchRiey2017` a la bibliografia.
- [ ] Compilar, comprovar les referències i fer l'última lectura amb Tobias.
