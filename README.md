# TDA Filtration Explorer

Interactive visualization of persistent homology on a range of manifolds,
real datasets (ancient/modern human DNA via Global25, the HK97 viral
capsid, the Drosophila optic lobe connectome, ubiquitin at atomic
resolution, …) and structured point clouds (Menger sponge, Sierpinski,
Weaire–Phelan foam, dense sphere packing).

The page runs entirely in the browser: synthetic-manifold sampling and
persistent-homology computation are done in [Pyodide][1] (Python +
NumPy + SciPy compiled to WebAssembly). No backend required for the
deployed site.

[1]: https://pyodide.org

## Running locally

Any static web server will do. Two options:

```bash
# Option A — static only (matches GitHub Pages exactly)
python3 -m http.server 8765
# then open http://localhost:8765/index.html
```

```bash
# Option B — Flask + GUDHI backend (faster compute for power users)
bash run.sh
# then open http://127.0.0.1:5000
```

Option A is the production setup. Option B keeps the original Flask
server around for local development; with this path /sample and
/compute are no-ops because the frontend always routes them through
Pyodide. You can still hit those endpoints directly from a Python
shell or curl if you want to validate the pure-Python implementation
against GUDHI (see `smoke_pyodide.py`).

## Deploying to GitHub Pages

1. Create an empty GitHub repository.
2. In this directory:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin git@github.com:<your-user>/<repo>.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source:
   Deploy from a branch → Branch: main / (root) → Save**.
4. After a minute, the site is live at
   `https://<your-user>.github.io/<repo>/`.

Notes on file sizes — `hk97_data.json` (64 MB), `menger_data.json`
(27 MB), and `foam_data.json` (16 MB) exceed GitHub's recommended
25 MB but fall under the 100 MB hard limit, so the push works (with
a warning) and Pages serves them fine. Repo clones are slow because
of these.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | The whole frontend (sliders, plots, all the prose). |
| `tda_core.py` | Pure-Python samplers + persistence, loaded into Pyodide at runtime. |
| `server.py` | Original Flask backend — reference implementation against GUDHI. |
| `smoke_pyodide.py` | Parity test: `tda_core` vs GUDHI. Run `python3 smoke_pyodide.py`. |
| `requirements.txt` | Dependencies for server.py / build scripts. |
| `run.sh` | Boots Flask on port 5000. |
| `build_*.py` | Offline jobs that produced the precomputed `*_data.json` files. |
| `*_data.json` | Precomputed persistence / Betti / cocycle data served as static assets. |
| `images/` | Static images used in the article (optic lobe press images, etc.). |

## How the Pyodide bridge works

`index.html` loads `pyodide.js` from the jsdelivr CDN, then lazily
initializes a runtime the first time a synthetic manifold is sampled
(precomputed-data sections like G25 / HK97 don't pay this cost). On
first call:

1. `loadPyodide()` boots the WASM runtime (~5 MB download).
2. `loadPackage(['numpy', 'scipy'])` fetches those wheels (~15 MB).
3. `tda_core.py` is fetched and imported.

After that, `/sample` and `/compute` calls are routed in-process via
the `window.tdaCore` shim (search `TDACORE_ROUTING` in index.html).
Results are JSON-serialized in Python and `JSON.parse`'d on the JS
side; that side-steps Pyodide's default dict→JS-Map conversion which
makes nested `obj.death` come back as `undefined`.

## Algorithm notes

The pure-Python persistence engine in `tda_core.py` implements:

- **Rips** — incremental clique expansion with full pairwise distance
  matrix. Faster than GUDHI's general code for tiny N, much slower
  for large N or dense graphs. Capped by the slider's `max_r`.
- **Alpha** — Delaunay (via `scipy.spatial.Delaunay`) with GUDHI's
  filtration rule: every simplex inherits the min α² of its cofaces
  unless its own circumsphere is empty of other Delaunay vertices, in
  which case its circumradius wins. Squared values are sqrt'd before
  output.
- **Čech (Delaunay–Čech)** — same skeleton as Alpha but using
  Welzl-style smallest-enclosing-ball radius per simplex instead of
  circumradius. Slower than Alpha because the per-simplex miniball is
  more work than a single linear solve.
- **Persistent homology** — standard left-to-right column reduction
  on the boundary matrix over F_p, with `min_persistence > 0` to
  match GUDHI's defaults. Drops dim-`max_dim` essentials (artifacts
  of the capped simplex tree).

`smoke_pyodide.py` validates that pair counts and Betti signatures
match GUDHI on a handful of fixtures (S^1, S^2, T^2, Klein) within a
couple of pairs (boundary cases in the Gabriel test).

## Limitations vs the Flask version

- Higher ambient dimensions (≥ 5) are slow: SciPy's `Delaunay` and the
  pure-Python column reduction don't have GUDHI's C++ inner loop.
  Pragmatic ceiling for interactive use: N ≤ ~250, max_dim ≤ 2.
- First sample click pays a ~5–10 second one-time cost while Pyodide
  bootstraps. Subsequent calls are instant.
