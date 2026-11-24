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
- [ ] Fer el commit en una branca i deixar escrit al missatge que la memòria cau queda invalidada.

## 1. Validar els canvis del solver

- [ ] Benchmark abans/després: `scripts/benchmark_performance.py --resolution 128 --case optimized` a `ed52042` (en un worktree) i a la versió nova. S'espera ~1.5–1.7× per iteració interna i gairebé el mateix nombre d'iteracions.
- [ ] Una resolució completa a 256 ($\alpha=0.000875$) per comparar $u$, $v$, les iteracions externes i la pèrdua amb el resultat antic convergit, si encara és a `results/`.
- [ ] `--workers 2` i `--workers 4` a 256: temps real per etapa. A 512, mesurar la memòria (RSS) per worker; encara no està mesurada.

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
  - numba o CuPy: dependència nova.
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

- [ ] Fixar versions: `environment.yml` no en fixa cap. Els resultats poden canviar amb numpy/scipy, i la clau de la memòria cau no inclou les versions. Generar un lock (`conda env export --no-builds` o conda-lock).
- [ ] Guardar les versions de Python, numpy i scipy a les metadades de cada `result.npz` (`checkpoint_cache.py:125`).
- [ ] `ast.dump` depèn de la versió menor de Python. Si es passa a 3.13, totes les claus canvien.

## 6. Documentació desactualitzada

- [ ] `README.md:124`: "The neutral 256 configuration ... identifies the same saved experiment" ja no és cert.
- [ ] `docs/reproducibility.md`: els àlies amb `meeting_256_stopping_fixed.json` són falsos. Cal explicar l'execució en paral·lel i `--workers`.
- [ ] `docs/performance_pass.md`, `stopping_fix.md`, `meeting_protocol.md` i `SOURCE_PROVENANCE.json` són històrics. Moure'ls a `docs/history/` o esborrar-los. `performance_pass.md` explica com aplicar un arxiu de pedaç que no existeix i conté rutes personals de Windows.
- [ ] `docs/model.md`, apartat del solver (opcional): explicar que els residus fan servir $Ku$ i $K^*p$ de la iteració anterior.

## 7. Experiments pendents (protocol)

- [ ] 256: `prepare` → `baseline` → `refine` → `compare` → `validate`.
- [ ] 512, amb pressupostos més grans (vegeu el README).
- [ ] Sensibilitat del guany a la tolerància (`local` contra `tighter`). Els guanys històrics són d'~1% (`docs/status.md`), és a dir, de la mateixa mida que l'error numèric possible.
- [ ] Comprovar si alguna $\alpha$ queda al límit de la cerca. A la versió de 128, les cel·les de dalt hi quedaven.
