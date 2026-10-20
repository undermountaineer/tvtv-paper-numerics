# Synthetic sequence

The tracked file `data/reference/synthetic_original_1024/reference_1024.npz`
contains a clean five-frame 1024×1024 sequence, frame times, physical spacings
and reference translation. Its SHA-256 checksum is
`99f2e56dc8bfc22ee9dca3fa7a23a08b4eac76f7c6a3cc7212060c50620ce834`.
The adjacent `validation.json` describes the master generation and checks.

The scene combines smooth areas, sharp interfaces and a textured region. Its
physical translation per frame is `(0.003, 0.0015)` in domain units. Pixel
displacement therefore depends on the working resolution. The reference motion
is used for evaluation; initialization uses the noisy observations only.

For working resolution N, average each nonoverlapping `(1024/N)×(1024/N)` clean
block. The working grid has spacing `hx=hy=1/N` and frame spacing `ht=1`. Add
independent unclipped Gaussian noise after averaging. Reconstruction and image
loss evaluation use this working grid and its clean reference.

The current N=256 and N=512 protocols use noise standard deviation 0.1 and the
same declared design/validation seed numbers. Their observations differ in
dimension and sampling; results are not interchangeable across resolutions.
The alpha search is performed at each resolution. The spatial partition of
alpha is independent of image resolution; all local crops retain the original
pixel spacing and full time interval.

The earlier `proof_of_concept.json` experiment directly renders a different
working-grid reference. Its historical results are documented in `status.md`.
They cannot be adopted as solved candidates for the master-based experiments.
