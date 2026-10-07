# Single-contiguous-interval ground contact (lookup schema v10)

The contact lookup describes the sole touching flat ground \(y=0\) (upward normal
\(n_g=(0,1)\)) on **one contiguous interval of bottom nodes**. Every such interval
is precomputed from the existing coupled foam–plate FEM compliance blocks; the
runtime only contracts stored responses with scalar coefficients. There is no
runtime FEM solve, no separate contact or plate solver, and no prescribed
toe-angle formula.

The implementation supports one contiguous contact interval. If the normal reactions become tensile inside an otherwise active interval, or if two separated sole regions simultaneously contact the ground with a free region between them, a multi-interval contact model is required.

## Interval records

With \(N_b\) bottom nodes sorted by increasing \(x\), an interval is

\[
\mathcal I_{ij}=\{i,i+1,\dots,j\},\qquad 0\le i\le j<N_b .
\]

All \(N_b(N_b+1)/2\) intervals are enumerated (row order: \(i\) ascending, then
\(j\) ascending; `lookup.row_of(i, j)` maps back). The topology label is derived
from the endpoints and is a filter only, never a separate model:

| Label | Condition | Count |
|-------|-----------|-------|
| `heel` | \(i=0,\ j<N_b-1\) | \(N_b-1\) |
| `toe` | \(i>0,\ j=N_b-1\) | \(N_b-1\) |
| `full` | \(i=0,\ j=N_b-1\) | 1 |
| `interior` | \(0<i\le j<N_b-1\) | \((N_b-1)(N_b-2)/2\) |

Per record the lookup stores `contact_start_index` \(i\), `contact_end_index`
\(j\), `contact_start_x` \(l_h=x_i\), `contact_end_x` \(l_t=x_j\), the anchor, the
adjacent free node ids (`heel_adjacent_free_node_id` \(=i-1\),
`toe_adjacent_free_node_id` \(=j+1\), or \(-1\) beyond an end of the sole),
`contact_mask`, a `status` (`ok`, `single_contact_node`, `rejected`) and a
structured `rejection_reason`. One-node intervals (\(i=j\)) are valid records:
the heel and toe edge are the same node and its reaction is reported once,
never summed twice.

Records are rejected (never silently dropped) when the boundary matrix is
numerically rank deficient (`rank_deficient_boundary_matrix`, decided from the
LU reciprocal condition estimate and confirmed by an SVD), the solve fails, the
solve / boundary-condition / equilibrium residual exceeds its tolerance, the
response is non-finite, or the plate recovery residual is too large.
`metadata["rejection_counts"]` and `n_intervals_valid` summarize them.

## Anchor and curved-sole closure

The interval anchor is a **numerical decomposition point**, not a physical
contact edge (`contact_anchor_definition = "interval_midpoint"`):

\[
x_a=\tfrac12(x_i+x_j),\qquad y_a=y_b(x_a)\ \text{(linear interpolation of the reference bottom profile)}.
\]

Every active node is prescribed the exact rotating-frame displacement that puts
it on the ground at its reference abscissa (perfect sticking):

\[
\mathbf d_k^{\mathcal T}=\mathbf d_a^{\mathcal T}+\bigl(Q(\varphi)^T-I\bigr)
\begin{pmatrix}x_k-x_a\\0\end{pmatrix}
+\begin{pmatrix}0\\-(y_k-y_a)\end{pmatrix}.
\]

The last term is the **curved-sole closure**. It is independent of the load and
of \(\varphi\), so it is stored as affine column 0 of every response tensor
with coefficient fixed at exactly 1 (the top is clamped for that column). On a
flat sole it is identically zero. Every runtime quantity is

\[
z = z_{\mathrm{closure}}+\sum_{k=0}^{4}\gamma_k z_k,\qquad
\boldsymbol\gamma=[\tan\theta,\ d_{a,x},\ d_{a,y},\ \cos\varphi-1,\ -\sin\varphi],
\]

with true \(\sin\varphi\), \(\cos\varphi\) (no small-angle expansion). Response
tensors therefore have six columns: `curved_sole_closure`, `top_shape_alpha`,
`contact_translation_x`, `contact_translation_y`, `contact_rotation_x`,
`contact_rotation_y`.

## Generation

For each interval the free/contact partition, anchor and closure are formed;
the full-domain compliance blocks are reused, never rebuilt. The dense boundary
saddle system is LU-factored **once** and all six right-hand sides are solved
against that factorization (no explicit inverse). Per record the solve
produces:

- bottom displacement bases `bottom_u_basis`, `bottom_v_basis` and contact
  reaction bases `reaction_x_basis`, `reaction_y_basis` — shape `(n_rec, 6, N_b)`;
- top force bases `top_force_x/y_basis` — `(n_rec, 6, n_t)`;
- scalars `scalar_lookup` `(n_rec, 6, 6)` = `[Fx, Fy, Mv, Mz, M_toe, M_toe_vertical]`;
- `edge_responses`: heel/toe edge local reactions and heel/toe adjacent-free
  displacements, `(n_rec, 6)`, NaN when the adjacent node does not exist;
- `Q_alpha_shoe_on_foot_basis` `(n_rec, 6)` for the passive toe spring;
- `kf_matrix` `(n_rec, 2, 2)` and its singular values / condition number;
- residual diagnostics, numerical rank, matrix size, condition estimate;
- for layered results, the six-column plate bases (`plate_u/v/rotation_local_basis`,
  multiplier, axial force) recovered from the same factorized FEM.

### Stored scalars, on-demand nodal fields

By default only the per-record scalars are kept (`scalar_lookup`,
`edge_responses`, `Q_alpha_shoe_on_foot_basis`, `kf_matrix`, rigid amplitudes,
residuals, interval topology). The per-node bases above (and the plate bases)
are **not** stored; they are recomputed on demand by
`ContactLookupResult.record_fields(rows, plate=False)`, which re-runs the
record's boundary solve from the stored prepared compliance matrix
(`field_solver_compliance`, \((2n_t+2N_b)^2\)) and, for layered/measured soles,
the plate influence matrices (`plate_influence_*`). The re-solve uses the same
code path and inputs as generation, so the fields are **bitwise identical** to
stored ones. Solved records are kept in a thread-safe LRU cache
(`FIELD_CACHE_ROWS = 2048`).

The per-node bases grow like \(N_b^3\) (one \(6\times N_b\) block per each of the
\(N_b(N_b+1)/2\) intervals), so this is what keeps the measured 270 mm lookup at
about 20 MB instead of 377 MB. Pass `generate_contact_lookup(..., store_fields=True)`,
`python -m compliance_fem.contact_lookup_cli --store-fields`, or set
`lookup.store_nodal_fields: true` in a measured config to keep them; selections
are identical either way. `lookup.nodal_fields_stored` reports the mode.

At runtime, `reconstruct_rows(..., exact="auto")` evaluates every row from the
stored scalars and edge data, then fetches nodal fields only for rows that pass
an exact necessary-condition prefilter (force reproduced, finite scalars,
non-negative heel/toe edge normal reactions and adjacent free gaps within
tolerance). A row failing the prefilter is provably inadmissible, so
admissibility is exact. For non-prefiltered rows the result carries
`fields_exact = False`, the per-node output arrays are NaN, and the violation
score, gap/reaction penalties and maximum free penetration are edge-only **lower
bounds** (scaled by \(1-10^{-9}\)); minimum free gap and minimum contact
reaction are upper bounds. Wherever these rows are ranked (least-violating
fallback, passive-toe and wrench-control shortlists), `refine_top_k` makes the
leading candidates exact before trusting the order, so the chosen row and its
reported values match a fully stored lookup. `exact="all"` forces exact fields
for every row. On a stored lookup every evaluated row is exact.

One difference remains: with a compact lookup the fallback message's
interior-tension (`disconnected_contact_warning`) note only considers rows that
were evaluated exactly. The selected row's own flags are unaffected.

## Runtime force control

For each evaluable record (no FEM solve):

1. \(F^{\mathcal T\star}=Q(\varphi)^T F^{\mathcal F\star}\).
2. \(K_F\,[d_{a,x},d_{a,y}]^T=F^{\mathcal T\star}-F_{\mathrm{closure}}-\alpha F_\alpha-r_xF_{r_x}-r_yF_{r_y}\)
   (2×2 solve; singular \(K_F\) → record not evaluable; ill-conditioned → flagged).
3. Contract all responses with \([1,\boldsymbol\gamma]\).
4. Transform the bottom to the fixed frame about \(r_a^{\mathcal F}=(x_a,0)\) and form

\[
g_k=\sin\varphi\,\Delta x_k+\cos\varphi\,\Delta y_k,\qquad
R_{n,k}=\sin\varphi\,R_{x,k}^{\mathcal T}+\cos\varphi\,R_{y,k}^{\mathcal T},
\]

with \(\Delta x_k=(x_k-x_a)+u_k-d_{a,x}\), \(\Delta y_k=(y_k-y_a)+v_k-d_{a,y}\).

With the passive toe spring, step 2 is replaced by the exact per-interval toe
equilibrium of [passive_toe_spring.md](passive_toe_spring.md): each interval
solves its own \(Q_\alpha(\alpha)+\partial U/\partial\alpha=0\) with the exact
\(\arctan\) spring; roots are never shared or averaged between intervals.

## Admissibility (all nodes)

Tolerances are mesh- and magnitude-scaled (\(h\) = median bottom spacing,
\(R_{\mathrm{ref}}=\max(|F^\star|,F_{\mathrm{floor}})\,h/L\)):

\[
\tau_{g,\mathrm{eff}}=\tau_g+\texttt{gap\_rel\_tol}\,h,\qquad
\tau_{R,\mathrm{eff}}=\tau_R+\texttt{reaction\_rel\_tol}\,R_{\mathrm{ref}}.
\]

A record is admissible when it is finite, reproduces \(F^\star\), **every** free
node has \(g_k\ge-\tau_{g,\mathrm{eff}}\) and **every** contact node — both edges
and all interior nodes — has \(R_{n,k}\ge-\tau_{R,\mathrm{eff}}\).

For linear elements the bottom gap is piecewise linear between nodes, so its
minimum over a free segment is attained at a node and the nodal check is exact.
For quadratic elements the per-segment quadratic minimum
(`_quadratic_segment_minimum`) is also checked.

Interior tension on an otherwise compressive interval sets
`disconnected_contact_warning` and the message recommends a multi-interval
model; such a candidate is never admissible.

Edge diagnostics: heel/toe edge reactions (local, fixed, normal), adjacent free
gaps (NaN when absent), and a dimensionless complementarity score
\(|R_{n,\mathrm{edge}}\,g_{\mathrm{adj}}|/(R_{\mathrm{ref}}h)\) per side.

The violation score is
\(V=w_g\sum_{\mathrm{free}}\max(0,(-g-\tau_g)/h)^2+w_R\sum_{\mathrm{contact}}\max(0,(-R_n-\tau_R)/R_{\mathrm{ref}})^2+w_F(\|F-F^\star\|/F_s)^2\)
plus flag penalties (non-finite, force mismatch, ill-conditioned \(K_F\), failed
or unstable toe root).

## Selection

Modes: **Auto** (all intervals), **Heel** / **Toe** (attached to that end;
the full interval is included in both), **Interior**, **Full** (forced), and
**Specific** (forced \((i,j)\), validated). Legacy GUI names `Auto`,
`Heel contact`, `Full contact`, `Toe contact` map onto these filters.

Search labels (`candidate_search_method`):

| Label | Meaning |
|-------|---------|
| `local_interval_search` | admissible within \(|i-i_p|+|j-j_p|\le\) `local_radius` of the previous interval |
| `expanded_interval_search` | admissible within the next `expansion_radii` |
| `global_interval_search` | admissible anywhere in the mode filter |
| `forced_interval` | Full / Specific mode |
| `least_violating_fallback` | nothing admissible; minimum \(V\), marked approximate |

Local and expanded stages only run with `SelectionConfig.use_temporal_continuity`
and a previous interval. Among admissible candidates the previous interval is
kept if it is still admissible; otherwise the minimum complementarity score
(plus the optional `topology_change_penalty`, applied to admissible candidates
only) wins, preferring well-conditioned \(K_F\), with ties broken by interval
distance and then row. Continuity is therefore only a tie-breaker among
physically acceptable candidates: it can never keep a penetrating or tensile
interval. With continuity disabled the result does not depend on evaluation
order.

## GUI

The contact-mode control offers Auto / Heel / Interior / Toe / Full / Specific;
Specific exposes start and end node indices. The fixed-frame plot draws the
contact interval as a solid curve on the ground and the free bottom dashed,
marks the heel edge (`triangle-right`) and toe edge (`triangle-left`) at
\(l_h\) and \(l_t\), marks the adjacent free nodes with open diamonds, and draws the interval anchor as a grey open cross labelled
as a numerical anchor (it is not a contact boundary). The diagnostics panel
shows \(i\), \(j\), \(l_h\), \(l_t\), edge normal reactions, adjacent free gaps,
complementarity, minimum free gap, minimum contact reaction, search method and
admissible count; absent values display **N/A**. The rendered internal plate is
contracted from the selected interval's own stored plate response. Changing the
mode or interval only re-contracts stored data.

## Schema and compatibility

`schema_version = 10`, `contact_set_model = "single_contiguous_interval"`,
`contact_anchor_definition = "interval_midpoint"`. v10 adds per-record
`rigid_alpha_basis` and an optional measured-sole geometry section
([measured_sole.md](measured_sole.md)); v9 rectangle and layered files are
migrated on load. Files with schema v8 or
older are rejected with a regenerate message: they store only \(2N_b-1\)
heel/toe/full records with contact-edge or \(L/2\) anchors and no curved-sole
closure, so they cannot represent arbitrary curved-sole intervals. Missing
interval fields, a different anchor definition or truncated record arrays are
also rejected. Compact (default) and full-field v10 files share the schema;
the saved `nodal_fields_stored` flag tells them apart, and older full-field v10
files load unchanged.

## Layered rocker bottom boundary

The layered mesh previously selected bottom facets with a midpoint test that
fails on a curved (rocker) bottom profile, leaving no bottom boundary. Bottom
facets are now the boundary facets whose nodes all lie on
`config.y_bottom(x)`, so curved layered soles build interval lookups whose plate
response matches direct FEM.

## Performance

See the "Interval contact performance" table in the README for record counts,
file size, build time and runtime per sample of the shipped lookups.
