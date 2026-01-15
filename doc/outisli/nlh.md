# The Analytical Pair Potential of Zone Bridging

*How the coefficients of DPA4/DPA4C's `nlh` bridging mode were obtained: what the term has to do, why the published Nordlund-Lehtola-Hobler table cannot be used as published, the constrained refit that replaces it, and what the refit costs. Record of the design work of 20 September 2026.*

This document is the methods record of a table the project fits itself. Coefficients taken from a paper carry that paper's methods section as their justification; ours carry only what is written here, so the fit is specified completely enough to be reimplemented and every quoted number names the script that produced it. Statements that are inferences rather than measurements say so in the sentence that makes them. The shipped behaviour is documented with the code in `doc/model/dpa4c.md` and `doc/model/dpa4.md`; everything below — including the alternatives that were tried and rejected — stays here.

## Contents

- 0\. Summary
- 1\. What the analytical term has to do
- 2\. Why the published table cannot be shipped as published
- 3\. The paper's error metric, and its validation
- 4\. The refit: objective, constraints, solver
- 5\. What the constraints guarantee
- 6\. The structural obstruction
- 7\. The operating range, and the fit floor it justifies
- 8\. Pareto I: the fit range
- 9\. Pareto II: the tail bound
- 10\. Honest comparison
- 11\. B-Ne
- 12\. Coverage above Z = 92
- 13\. Runtime packaging
- 14\. Provenance and licensing
- 15\. Rejected alternatives and negative results
- 16\. What was not verified
- Appendix A — scripts, and how to run them
- Appendix B — data

## 0. Summary

DPA4C's bridging term gains a second mode, `nlh`, whose coefficients are a refit of the same reference data the NLH authors used, under three conditions their fit does not carry: non-negative amplitudes, unit amplitude sum, and a per-pair bound on the long-range energy. `zbl` remains the default; the choice of default is the user's and is not settled by this work.

The published NLH coefficients are not shipped. 603 of their 4278 pairs leave a long-range tail above 1 meV at 5 Å and 450 above 0.1 meV at 6 Å, worst case 1.62 eV at 6 Å, and the term is added on every edge inside the cutoff with no switching function. That is a measured fact, and its cause is structural rather than accidental: the publication's objective is undefined below 10 eV, so a slowly decaying term costs it nothing.

The refit meets the accuracy requirement and the tail requirement simultaneously; there is no compromise to negotiate between them. Against the published table it is better in the energy range where the analytical term is the only description (median relative error 1.43 % above 100 eV against 1.63 %, winning 90 % of the 4278 pairs), its worst case is an order of magnitude better (6.9 % against 86.9 %, setting aside B-Ne, which §10 explains), and it is positive and strictly decreasing at every radius as a theorem rather than as an observation. It is worse in two places, both below the range where the term ever acts alone; §10 states them plainly and §6 explains why they are a consequence of the constraints rather than a defect of the fit.

The functional form is unchanged, so the fused CPU and CUDA kernels need no change: they already consume `[A_1..A_4, c_1..c_4]` generically. Verified by execution, not by reading (§13).

## 1. What the analytical term has to do

The bridged model is `E = E_learned + sum_pairs V(r_ij)`, with V a purely analytical, purely repulsive pair potential added unconditionally on every edge inside the cutoff, which is 6 Å in every DPA4 and DPA4C preset. There is deliberately no switching function on V: a complementary switch `(1 - w) V` contributes a cross term `-w' V` to the force, which the user rejected. The learned part is muted instead, by a switch on the descriptor's edge envelope that closes at a per-pair inner radius and is fully open at a per-pair outer radius, and training frames containing a pair closer than the midpoint of that window are filtered out. Inside the inner radius V alone is responsible for the physics.

Two requirements follow.

The first is accuracy in the high-energy repulsive regime, because that is where V is the only description. §7 turns this into a number.

The second is that V must be negligible at long range. The decisive mechanism is not that a small energy is nice to have: without a switch, a pair crossing the cutoff changes the total energy by exactly `V(rcut)` and the force on the two atoms by exactly `|dV/dr|(rcut)`, discontinuously. That is a genuine discontinuity in the model, a source of energy drift in MD, and it makes the total energy depend on the neighbour-list radius. `V(rcut)` is therefore the binding number and `V(rcut - 1 Å)` the companion that keeps the approach to it tame. Stated so that it transfers if the cutoff ever changes, the criterion adopted is

```
V(rcut) <= 0.1 meV        |dV/dr|(rcut) <= 1 meV/A        V(rcut - 1 A) <= 1 meV
```

which for `rcut = 6 Å` is a bound of 1 meV at 5 Å and 0.1 meV at 6 Å. Since V is monotone (§5), a bound at those two radii extends to every larger radius by itself.

For scale, measured by `nlh_evaluate.py compare` and the force table of §10: the ZBL universal potential, which is what the `zbl` mode uses, has a worst case over all 118 elements of 6.16 meV at 5 Å (Sc-Sc), 1.05 meV at 6 Å (Ne-Ne) and 1.78 meV/Å at 6 Å (P-P). The criterion is therefore six to ten times tighter than what the package currently tolerates; every one of ZBL's pairs fails it. That is deliberate. The point of the exercise is to be able to add the term without thinking about it again, and §9 shows the criterion costs almost nothing.

The bridging window is set by covalent radii and is independent of the analytical potential. Nothing about the table moves it, and §7 reads the window only to find out which energies matter.

## 2. Why the published table cannot be shipped as published

### 2.1 What the published table gets right

The NLH potential is `V = k_e Z1 Z2 / r * phi(r)` with `phi(r) = sum_{i=1..3} a_i exp(-b_i r)` and unscaled r — exactly the shape of the existing pair table, using three of its four exponential slots. Its advantage over ZBL is real and large: at each pair's DFT equilibrium bond length the residual repulsion has median 1.12 eV for ZBL and 0.29 eV for NLH, NLH is smaller for 95 % of pairs, and O-O at 1.208 Å gives 11.65 eV for ZBL against 1.55 eV for NLH (`nlh_evaluate.py compare`). That is why the form was adopted at all.

### 2.2 The disqualifying defect

24 of the 4278 published rows have `b_3 = 0` with `a_3 > 0`, so `phi -> a_3` as `r -> infinity` and V decays as a bare Coulomb tail. Further rows have `b_3` small but nonzero, which inside a 6 Å cutoff is the same thing. Measured over all 4278 pairs by `nlh_evaluate.py compare`:

| quantity                          | ZBL universal | published NLH      |
| --------------------------------- | ------------- | ------------------ |
| pairs with V(5 Å) above 1 meV     | 4278          | 603                |
| pairs with V(6 Å) above 0.1 meV   | 4278          | 450                |
| worst V(5 Å)                      | 6.16e-3 eV    | 1.94 eV (B-Mg)     |
| worst V(6 Å)                      | 1.05e-3 eV    | 1.62 eV (B-Mg)     |
| worst \|dV/dr\|(6 Å)              | 1.78e-3 eV/Å  | 0.279 eV/Å (Be-Be) |
| worst residual at the bond length | 17.97 eV      | 3.89 eV            |

A force discontinuity of 0.279 eV/Å at the neighbour-list boundary is 157 times ZBL's. Al-Al gives 2.97 eV at its 2.86 Å bond length, worse than ZBL's 0.32 eV, so for the very pairs whose tail is broken NLH also loses its advantage at the bond.

To make the magnitude concrete, `nlh_evaluate.py` integrates the pair energy over a neighbour shell at uniform density. Summed over pairs between 4.5 and 6 Å, the spurious energy the learned part must absorb is 31 to 157 meV/atom for ZBL, 0.001 to 20349 meV/atom for the published NLH (the large values are Al and Mg), and 0.0001 to 11 meV/atom for the refit.

### 2.3 The root cause, established twice

The paper's objective, its Eq. (14), sums only over grid points with `V >= 10 eV`: a relative error in the screening function above 30 eV, and below 30 eV an absolute error scaled by the screening function at 30 eV. Data points below 10 eV "were not considered in the fits". Nothing in the objective refers to any radius beyond `r(V = 10 eV)`, which has median 1.64 Å, and the 10-30 eV band is a median 14 % of the fit's radial range. Everything beyond that radius — the entire region in which the term is added to our model — is unconstrained, and a slowly decaying term there costs the objective nothing while buying flexibility inside the window.

Two independent measurements establish that this, and not a poor optimiser run, is the cause. Both come from `nlh_evaluate.py pathology`.

First, the defective rows are *better than typical* fits in their own validated range. The 24 zero-exponent rows have a published E10 median of 1.97 % against 3.53 % overall, the 25th percentile — better than 75 % of all rows; their E30 median is 1.68 % against 1.80 %, the 43rd percentile. The same holds for the wider set of 450 rows that exceed 0.1 meV at 6 Å. Under the published objective a slow term is not a symptom of failure; it is a reward.

Second, refitting the same DMol data with the same objective, the same four-exponential form and no tail bound reproduces the pathology and makes it worse. A term counts as *surviving* when its amplitude exceeds 1e-4: the amplitudes of a row sum to one, so a term below that carries less than a ten-thousandth of the screening function at contact and cannot move the tail.

|                                         | published NLH | refit, no tail bound | refit, tail bound |
| --------------------------------------- | ------------- | -------------------- | ----------------- |
| slowest surviving decay rate, minimum   | 0.0000 Å⁻¹    | 0.0500 Å⁻¹           | 0.9204 Å⁻¹        |
| pairs with a surviving rate below 1 Å⁻¹ | 40            | 596                  | 1                 |
| pairs above 1 meV at 5 Å                | 603           | 1418                 | 0                 |
| worst V(6 Å)                            | 1.617 eV      | 3.408 eV             | 1.000e-4 eV       |

The 0.0500 Å⁻¹ is the lower bound of the refit's own search range, so the unconstrained fit drives the slowest term all the way to whatever floor it is given. The remedy is therefore not a patch on 24 rows; it is a constraint the objective should have carried.

The one shipped row that keeps a surviving term below 1 Å⁻¹ is Li-Ge, at 0.9204 Å⁻¹ with an amplitude of 1.1e-4; the next slowest surviving term anywhere in the table sits at 1.1788 Å⁻¹. That row is not an escape, because the constraint is on the energy rather than on the rates and it binds every one of the 7021 rows: the worst `V(5 Å)` is 1.000e-3 eV and the worst `V(6 Å)` is 1.000e-4 eV, both exactly at the bound.

### 2.4 The erratum, and what the current package does and does not fix

There is an erratum: K. Nordlund, S. Lehtola and G. Hobler, *Erratum: Repulsive interatomic potentials calculated at three levels of theory*, Phys. Rev. A **112**, 059901 (2025), doi 10.1103/cdrk-x7my. Its abstract states that it provides new parameters for the Na-O screening function, "that had an unphysical value at the limit of infinite distance in the original data set". Na-O (Z1 = 8, Z2 = 11) is one of the 24 zero-exponent rows, so the authors recognise the defect.

The current open-data package is Zenodo record 10.5281/zenodo.22092149, version v4 of 25 August 2026, file `nlh_potentials_opendata_v4.tar.gz`, md5 `89186315fbc9a34ec376d36a8802f52b`, 20.6 MB. Diffing its coefficient file against v1.0 (`nlh_evaluate.py erratum`) shows **exactly one changed line**, the Na-O row, and leaves **23 rows with the identical defect**. v4 still has 602 pairs above 1 meV at 5 Å, 449 above 0.1 meV at 6 Å, and the same worst case of 1.6171 eV at 6 Å for B-Mg.

The corrected Na-O row also exhibits the trade this work quantifies: replacing `b_3 = 0` with `b_3 = 2.70748` moved its printed E10 from 2.27 % to 10.45 %. The authors paid four and a half times the fit error at the bottom of their window to buy a decaying tail on one pair.

v4 adds an `nlhlin/` directory holding a *finite-range* NLH variant, cut to exactly zero at the distance where the DMol data turns negative, for a separate paper. That is a switching function in a different guise, it does not fit the four-exponential table, and it is not usable here.

## 3. The paper's error metric, and its validation

The published per-pair error columns E30 and E10 cannot be reproduced from the obvious reading of the paper. Sampling the native DMol grid gives a median E30 of 1.43 % against their printed 1.80 %, a bias of -0.46 pp. Several point-set conventions were tried, and the one that reproduces both columns is

> E_T is the RMS relative deviation of the fitted screening function from the reference screening function, sampled on a grid **uniform in r** from the innermost data point (0.002 Å) out to the radius at which the reference pair energy equals T eV.

with the crossing radius obtained by log-linear interpolation between the two bracketing grid points. Measured over the 4277 DMol-fitted pairs by `nlh_evaluate.py metric`, 400 sample points: E30 median 1.74 % against the printed 1.80 % (mean bias -0.02 pp, median absolute difference 0.048 pp) and E10 median 3.66 % against 3.53 % (bias +0.23 pp, median absolute difference 0.090 pp, correlation 0.955). That is close enough to treat the convention as identified; the residual difference from the authors' exact grid was not chased further.

Every E-number in this document uses that convention, applied identically to ZBL, to the published table and to every refit, so the comparison is internally consistent whatever the residual difference. The effort was worth making for one reason beyond bookkeeping: it lets the refit be reported in the paper's own currency, so "better than the published table above 100 eV" is not a claim about a metric chosen to favour it.

## 4. The refit: objective, constraints, solver

### 4.1 Form

The form is the publication's, so the table layout is unchanged:

```
phi(r) = sum_{k=1..4} a_k exp(-b_k r)        V(r) = k_e Z1 Z2 phi(r) / r
A_k    = k_e Z1 Z2 a_k   [eV A]              c_k = b_k   [1/A]
```

`k_e = 14.3996454687 eV Å` in the fit, the value the authors' own reference evaluator builds from CODATA constants; DeePMD-kit's table build uses the repository's 14.3996, a relative difference of 3.2e-6 that is four orders below the fit's own accuracy.

### 4.2 Objective

Let `phi_ref` be the reference screening function, `r_lo` the innermost reference radius (0.002 Å), `r(T)` the radius at which the reference pair energy equals `T`, and `G` the set of `N = 200` points spread uniformly in r over `[r_lo, r(lo_eV)]`. Then

```
chi^2(a, b) = sum_{r in G} [ w(r) ( phi_fit(r) - phi_ref(r) ) ]^2

w(r) = 1 / phi_ref(r)             for r <= r(cap_eV)
     = 1 / phi_ref(r(cap_eV))     for r >  r(cap_eV)

lo_eV = 10        cap_eV = 100
```

`phi_ref` is obtained at arbitrary radii by linear interpolation of `ln phi_ref` against r on the reference tabulation, restricted to the repulsive branch — the reference turns attractive near the bonding minimum and the screening function is only defined where the pair energy is positive.

The weight is the reciprocal of the reference, so the residual is a relative deviation; below `cap_eV` the weight stops growing, which prevents the outermost points, where the reference bends towards the bonding minimum, from dominating the sum.

Two things differ from the publication's Eq. (14).

The **weight cap moves from 30 eV to 100 eV**. The paper's cap is set by its own use case, bridging into a many-body potential in the 10-100 eV range. Ours is set by the bridging window, which hands the analytical term the whole potential only above 19 eV for the worst pair and above 150 eV for 95 % of them (§7). The floor stays at the paper's 10 eV, so no data they validated is discarded and the deviation is exactly one number.

The **objective is evaluated on the same uniform-in-r grid as the metric**, whereas the paper appears to minimise on the native DMol grid. The native grid places 19 of about 45 in-range points below 0.2 Å, where `phi` is within a few percent of 1 and every reasonable fit is already accurate to better than 1 %; minimising there spends the fit's freedom where none is needed. That the paper minimises on the native grid is an *inference* from the metric mismatch of §3, not something the paper states.

### 4.3 Constraints

**C1, unit sum: `sum_k a_k = 1`.** This fixes `phi(0) = 1`, the exact unscreened Coulomb limit. The published table already obeys it, so it is not a change; it is retained because it pins the deep-inner region for free — measured deep-inner accuracy is better than the published table's, 0.54 % against 0.78 % above 10⁴ eV and 0.31 % against 0.51 % above 10⁵ eV (`nlh_evaluate.py compare`) — and because without it the relative-error weighting is meaningless as `r -> 0`.

**C2, non-negative amplitudes: `a_k >= 0`.** This is new. It is what turns positivity and monotonicity from observations into theorems (§5), and it removes catastrophic cancellation from the float32 table: the published H-H row is `a = (-9, +10)`, two terms of opposite sign that nearly cancel, whereas a sum of positive terms cannot lose digits.

**C3, the tail bound, written as an energy bound.** The requirement is `V(r_t) <= eps_t`. Substituting the form,

```
k_e Z1 Z2 / r_t * sum_k a_k exp(-b_k r_t) <= eps_t
```

and dividing through by the positive prefactor gives, at fixed `b`, **one linear inequality in the amplitudes**:

```
sum_k a_k exp(-b_k r_t) <= eps_t r_t / (k_e Z1 Z2)          r_t in {5 A, 6 A},  eps = {1 meV, 0.1 meV}
```

Writing it this way rather than as a floor `b_k >= b_min` matters twice over. A floor is merely sufficient, and it forbids a slow term with a small amplitude — exactly the term that carries the 1-10 eV region. And the requirement is Z-dependent in the direction that makes heavy pairs binding: at `Z1 Z2 = 92^2` the prefactor `k_e Z1 Z2 / 6` is about 20300 eV, so the 0.1 meV bound at 6 Å requires `sum_k a_k exp(-6 b_k) < 5e-9`, whereas for H-H the same bound requires only `< 4e-5`. A single hard floor would have to be set by the heaviest pair and would then be absurdly strict for hydrogen.

### 4.4 Number of terms

Four, matching the four exponential slots the fused kernels evaluate unconditionally, so the fourth slot costs nothing at runtime. Under C2 the fit is a projection onto a convex cone, so more terms cannot introduce oscillation and cannot make the curve non-monotone or negative; the usual over-fitting argument for keeping a model small does not apply.

Measured against a clean three-term run at the same floor, cap and tail bound (`nlh_evaluate.py terms`): the chi-squared ratio three-term over four-term has median 1.0000, p90 1.373, p99 2.50, max 5.46; the fourth term reduces chi-squared by more than 1 % for 1480 pairs and by more than 10 % for 904 of 4278. In the shipped table 4, 215, 2378 and 4424 rows use one, two, three and four slots — the last figure includes the ZBL-derived rows above Z = 92, which use all four. The benefit inside the fitted set is modest but free.

### 4.5 Solver

The amplitudes enter linearly and the decay rates do not, so the fit is solved by **variable projection**: the inner problem in `a` is solved exactly at fixed `b`, and the reduced objective `f(b) = min_a chi^2(a, b)` is minimised over four variables.

**Inner problem.** Minimise `||M a - y||^2` subject to `1'a = 1`, `a >= 0` and `C a <= d`, where `M[j,k] = w(r_j) exp(-b_k r_j)`, `y_j = w(r_j) phi_ref(r_j)`, `C[t,k] = exp(-b_k r_t)` and `d_t = eps_t r_t / (k_e Z1 Z2)`. This is a convex quadratic program in at most four variables, so any point satisfying the Karush-Kuhn-Tucker conditions is its global minimum. The solver:

1. tries the equality-only solution (unit sum alone) and returns it when it is already feasible, which is the common case;
1. otherwise loops over the 2^T subsets of tail rows held as equalities, and for each subset runs a Lawson-Hanson style active-set loop on the sign constraints — an index leaves the free set while its amplitude is negative and re-enters while its reduced gradient is;
1. certifies each candidate against the KKT conditions: the multipliers of the active tail rows must be non-negative, the reduced gradient must vanish on the free set and must be non-negative on the held set;
1. falls back, if no candidate is certified, to an exhaustive enumeration of active sets over supports and tail subsets.

Validated against brute-force enumeration of every active set, on 3000 random problems drawn from the same family as the real ones: one mismatch out of 3000, relative excess 6.7e-5 in the objective. That is the accuracy of the inner solve, measured on the shipped implementation.

**Outer problem.** `f(ln b)` is minimised by Nelder-Mead followed by a Powell polish, from fifteen starting rate sets — the ZBL rates of the pair, four geometric ladders scaled by the width of the fit window, and ten random sets drawn from a generator seeded by the element pair — of which the best three by `f` are refined. Rates are bounded to `[0.05, 1000] Å⁻¹`.

**Determinism.** Every random start comes from a generator seeded by `(Z1, Z2)` alone, and results are stored by pair index, so the worker count does not affect the output and a rerun reproduces the table.

**Robustness.** Checked on Fe-Fe: identical chi-squared from two, three and four slots, from three and eight retained starts, and from two random seeds. That is one pair, not a survey.

**Cost.** 7021 element pairs take about eleven minutes on sixty worker processes, roughly ten CPU-hours.

## 5. What the constraints guarantee

Three properties follow analytically from C1 and C2. The numerical confirmations below are confirmations, not the evidence.

**Positivity.** With `a_k >= 0` and at least one `a_k > 0`, `phi(r) = sum_k a_k exp(-b_k r) > 0` for every finite r, because each term is positive and at least one is nonzero. Hence `V = k_e Z1 Z2 phi(r) / r > 0` for `r > 0`, and the potential is purely repulsive everywhere without any case analysis.

**Strict monotonicity.** `phi'(r) = -sum_k a_k b_k exp(-b_k r) <= 0`, and it is strictly negative whenever some `a_k b_k > 0`, which C1 guarantees since the amplitudes sum to one and every live rate is positive. Then `V'(r) = k_e Z1 Z2 (r phi'(r) - phi(r)) / r^2 < 0`, because `phi' <= 0` and `phi > 0` make the numerator strictly negative. So V is strictly decreasing on `(0, infinity)`: no spurious minimum, no barrier, no flat region.

**The tail bound extends outwards.** Because V is strictly decreasing, `V(r) <= V(6 Å) <= 0.1 meV` for every `r >= 6 Å`, and `V(r) <= V(5 Å) <= 1 meV` for every `r >= 5 Å`. Imposing the bound at two radii therefore bounds the whole tail, and no constraint is needed beyond the cutoff.

**Confirmation.** `test_nlh_is_positive_and_strictly_decreasing` and `test_nlh_tail_bound_holds_for_every_pair` in `source/tests/common/dpmodel/test_inner_potential.py` check every row of a 118-element type map on a dense grid over `(1e-3, 20] Å`: positive and strictly falling everywhere, and the amplitudes summing to `k_e Z_a Z_b` to a relative 1e-12. The generator re-checks the same conditions over all 7021 rows before it writes the file, and refuses to write if any fails.

**One caveat, measured.** A row on which a tail bound is active lands on it to the conditioning of its own least-squares system. Over the whole table the largest relative excess is 2.47e-12 at 5 Å and 9.45e-8 at 6 Å — that is `V(6 Å) = 1.0000001e-4 eV` for one pair, an absolute excess of 9.5e-12 eV, which is smaller than one float32 ulp of the bound itself. The bound is met to the precision at which the table is stored and evaluated, and the verification in `nlh_refit.py` asserts it to a relative tolerance of 1e-6 for that reason.

## 6. The structural obstruction

This section exists because without it two rows of §10 read as defects rather than as consequences.

**Lemma.** For a positive mixture of exponentials `phi(r) = sum_k a_k exp(-b_k r)` with `a_k >= 0`, the local decay rate

```
beta(r) = -d ln phi / dr = sum_k a_k b_k exp(-b_k r) / sum_k a_k exp(-b_k r)
```

is non-increasing in r.

**Proof.** `beta(r)` is the mean of the `b_k` under the probability weights `p_k(r) = a_k exp(-b_k r) / phi(r)`. Differentiating, `dbeta/dr = -(sum_k p_k b_k^2 - beta^2) = -Var_p(b)`, which is at most zero. ∎

**The reference does the opposite at the bottom of the fit window.** Measured local decay rates across their own windows: H-H rises from 3.6 to 10.5 Å⁻¹, O-O goes 6.9 to 3.95 to 12.8, Fe-Fe goes 9.5 to 4.3 to 8.3. The steepening at the outer end is where bonding attraction begins to pull the pair energy down.

**Consequence.** A non-negative mixture cannot follow that steepening. It is exactly the region where the published fits spend their negative amplitudes, which is *why* they use them and why an unconstrained fit is so much better inside the window and so much worse outside it. The two places where the refit loses to the published table — E10 and the median residual at the bond length — are this lemma, not a failure of the optimiser or of the objective. Raising the weight cap moves the conflict out of the range that matters to us (§7), and tightening the tail bound reduces the bond-length residual as a side effect (§9), but neither removes the obstruction.

This is an inference drawn from a proved lemma plus measured reference curves; it was not tested by, for instance, fitting with five or six positive terms.

## 7. The operating range, and the fit floor it justifies

The bridging window mutes the learned part below an inner radius `r_in` and restores it fully above an outer radius `r_out`, each a fraction of the pair's own covalent bond length — 0.26 and 0.80 of the sum of the two Pyykkö single-bond covalent radii, the rule in force on 20 September 2026. Below `r_in` the analytical term is the whole potential. Mapping those radii onto the DMol pair energies over all 4278 fitted pairs (`nlh_evaluate.py window`):

| quantity                                  | value                                              |
| ----------------------------------------- | -------------------------------------------------- |
| inner radius                              | 0.166 to 1.206 Å, median 0.723                     |
| pair energy at the inner radius           | min 19.0 eV, p5 150 eV, median 638 eV, max 2697 eV |
| pairs whose inner radius sits below 10 eV | 0 of 4278                                          |
| pairs whose inner radius sits below 30 eV | 2 of 4278                                          |
| pair energy at the outer radius           | min 0.10 meV, median 1.04 eV, max 26.5 eV          |

So the range in which V is ever the sole description begins at 19 eV for the single worst pair, at 150 eV for 95 % of pairs, and at 638 eV for the median. **E100 is the metric closest to our requirement, E30 is margin, and E10 is below the operating range of every pair** — it is allowed to degrade.

That is the measured justification for moving the weight cap to 100 eV while leaving the floor at 10 eV: the cap decides where full relative weight applies, and the floor decides only how far out the fit is anchored at all. §8 shows what each choice costs.

## 8. Pareto I: the fit range

Sweep over the fit floor and the weight cap, all with the same constraints, four terms and the same tail bound; medians over 4278 pairs with p99 in parentheses (`nlh_refit.py fit` per variant, scored by `nlh_evaluate.py compare`).

| floor / cap (eV)       | E10 %        | E30 %       | E100 %     | E1000 %    | V(r_eq) eV   | max V(r_eq) eV | wins vs NLH: E30 / E100 |
| ---------------------- | ------------ | ----------- | ---------- | ---------- | ------------ | -------------- | ----------------------- |
| published NLH          | 3.66 (18.2)  | 1.74 (6.6)  | 1.63 (6.3) | 1.22 (3.7) | 0.286 (2.01) | 3.89           | —                       |
| 10 / 30 (the paper's)  | 3.69 (15.6)  | 2.08 (8.4)  | 1.86 (7.8) | 1.32 (4.8) | 0.271 (1.57) | 2.27           | 21 % / 22 %             |
| **10 / 100 (adopted)** | 4.79 (20.8)  | 1.66 (7.7)  | 1.43 (4.5) | 1.10 (3.1) | 0.298 (2.02) | 3.18           | 77 % / 90 %             |
| 20 / 60                | 5.12 (19.6)  | 1.57 (7.5)  | 1.45 (5.2) | 1.11 (3.4) | 0.310 (1.96) | 3.04           | 88 % / 86 %             |
| 30 / 100               | 6.32 (23.1)  | 1.71 (8.3)  | 1.24 (4.0) | 1.02 (2.7) | 0.329 (2.31) | 3.65           | 59 % / 92 %             |
| 10 / 300               | 5.71 (25.5)  | 1.91 (9.5)  | 1.23 (3.8) | 0.92 (2.2) | 0.317 (2.35) | 3.77           | 43 % / 93 %             |
| 100 / 300              | 10.11 (35.8) | 3.75 (13.2) | 1.04 (3.3) | 0.79 (1.7) | 0.398 (3.54) | 9.63           | 9 % / 92 %              |

`V(r_eq)` is the analytical energy left at the pair's DMol equilibrium bond length — what the learned part must cancel at the bond.

The adopted setting is **floor 10 eV, weight cap 100 eV**. The reasoning, in order: it keeps the paper's own fit floor, so the deviation from their procedure is one number; it is the only setting that beats the published table on the E30, E100 and E1000 medians *and* on the per-pair win rates at the same time; and its residual at the bond length, 0.298 eV, is level with the published table's 0.286 eV. This is a recommendation, and the alternative worth naming is floor 30 / cap 100, which buys 13 % in the E100 median (1.24 % against 1.43 %) for a 10 % larger bond-length residual. Within this range the choice is a mild lever and every row of it beats the published medians; the difference between the four middle rows is not consequential.

## 9. Pareto II: the tail bound

Sweep over the tail bound alone, at fixed floor 30 / cap 100 so the bound is the only variable.

| bound V(5 Å) / V(6 Å)     | E30 % | E100 % | V(r_eq) eV | max V(5 Å) | max V(6 Å) | chi² vs unbounded: med / p90 / p99 |
| ------------------------- | ----- | ------ | ---------- | ---------- | ---------- | ---------------------------------- |
| none                      | 1.45  | 1.12   | 0.610      | 4.30 eV    | 3.41 eV    | 1.00 / 1.00 / 1.00                 |
| 0.1 / 0.01 eV             | 1.47  | 1.14   | 0.439      | 66.9 meV   | 10 meV     | 1.000 / 1.10 / 2.0                 |
| **1 / 0.1 meV (adopted)** | 1.71  | 1.24   | 0.329      | 1.00 meV   | 0.10 meV   | 1.000 / 2.20 / 42.5                |
| 10 / 1 µeV                | 6.19  | 3.40   | 0.128      | 10.0 µeV   | 1.0 µeV    | 12.0 / 361 / 2231                  |
| 0.1 / 0.01 µeV            | 29.67 | 20.14  | 0.016      | 0.12 µeV   | 5.0 neV    | 364 / 2407 / 9477                  |

**The knee is at the adopted bound.** Loosening by two decades buys 8 % in the E100 median; tightening by one decade costs a factor 2.7 in E100 and 3.6 in E30. The bound binds for 1441 of 4278 pairs and is free for the median pair — the median chi-squared ratio is 1.0000.

One structural observation simplifies the picture: the tail bound and the bond-length residual pull the *same* way, because both are "make V decay faster". Tightening the bound from 1 meV to 10 µeV lowers the median residual from 0.329 to 0.128 eV. There is therefore no conflict between the two requirements of §1; the single real axis is decay rate against high-energy accuracy, and the adopted point sits at its knee.

A bound at 4 Å was also measured, on a 306-pair sample. Adding `V(4 Å) <= 10 meV` costs 8 % in the E100 median (1.47 % to 1.58 %) and halves the worst 4 Å energy (23.4 to 10.0 meV); a 3 meV bound costs 33 %. It was **not adopted**: at 4 Å the *true* DMol pair energy has median -631 meV (p1 -2156, p99 +39), so the network is already learning about 0.6 eV of attraction there and the refit's 3.4 meV median and 25.4 meV worst repulsive offset is a half-percent to four-percent perturbation of what it learns anyway — and already fifteen times smaller than ZBL's 50 meV median. The full-scale 4 Å runs were killed before completing; the sample is the only evidence.

## 10. Honest comparison

Medians over all 4278 pairs with Z1 ≤ Z2 ≤ 92, p99 in parentheses, measured on the shipped table itself by `nlh_evaluate.py compare`.

|                                     | ZBL universal | published NLH    | refit (shipped)                             |
| ----------------------------------- | ------------- | ---------------- | ------------------------------------------- |
| E10 %                               | 24.83 (101.3) | **3.66** (18.2)  | 4.79 (20.8)                                 |
| E30 %                               | 14.98 (54.3)  | 1.74 (6.6)       | **1.66** (7.7)                              |
| E100 %                              | 9.48 (29.2)   | 1.63 (6.3)       | **1.43** (4.5)                              |
| E1000 %                             | 3.94 (14.9)   | 1.22 (3.7)       | **1.10** (3.1)                              |
| V at the DFT bond length, eV        | 1.124 (10.50) | **0.286** (2.01) | 0.298 (2.02)                                |
| worst E30 %                         | 62.7          | 78.5             | **15.8**                                    |
| worst E100 %                        | 70.1          | 86.9             | 17.6, or **6.9** excluding B-Ne             |
| worst V(r_eq), eV                   | 17.97         | 3.89             | **3.18**                                    |
| worst V(4 Å), eV                    | 5.4e-2        | 2.43             | **2.5e-2**                                  |
| worst V(5 Å), eV                    | 6.2e-3        | 1.94             | **1.0e-3**                                  |
| worst V(6 Å), eV                    | 1.1e-3        | 1.62             | **1.0e-4**                                  |
| worst \|dV/dr\|(6 Å), eV/Å          | 1.778e-3      | 0.279            | **2.287e-4**                                |
| pairs failing V(5 Å) ≤ 1 meV        | 4278          | 603              | **0**                                       |
| pairs failing V(6 Å) ≤ 0.1 meV      | 4278          | 450              | **0**                                       |
| pairs with V ≤ 0 or non-monotone    | 0             | 0                | **0**                                       |
| per-pair wins against published NLH | 0.2 % / 2.8 % | —                | 77.0 % (E30), 89.9 % (E100), 75.1 % (E1000) |

Worst pairs by E100: ZBL has Xe-Pb at 70 %; the published table has Xe-Pb at 87 %, B-Ne at 18 % and Xe-Pr at 12 %; the refit has B-Ne at 17.6 %, then Fr-Fr at 6.9 %, Fr-Ra at 6.6 %, N-N at 6.5 % and the remaining Fr pairs at 6.2-6.4 %.

The refit's worst case needs a word, because the table above scores every pair against the DMol reference and B-Ne is deliberately fitted to MP2 instead (§11). Measured against the reference it is actually fitted to, B-Ne has an E100 of 3.06 %; its 17.6 % here is the measured distance between the two references for that pair, not an error of the fit. Excluding it, the refit's worst E100 is 6.91 % (Fr-Fr) against the published table's 86.94 %.

Worst spurious pair force over all pairs, in eV/Å:

| r   | ZBL     | published NLH | refit  |
| --- | ------- | ------------- | ------ |
| 3 Å | 2.16    | 6.46          | 2.30   |
| 4 Å | 0.136   | 0.646         | 0.083  |
| 5 Å | 1.25e-2 | 0.408         | 3.2e-3 |
| 6 Å | 1.78e-3 | 0.279         | 2.3e-4 |

The 3 Å row is not a spurious force: its worst cases are Th, Ac, Fr and Ra pairs whose equilibrium bond length is 2.7 to 5 Å, so 3 Å is inside their real repulsive wall, and `V(3 Å) = 0.69 eV` for the refit against ZBL's 0.77 eV.

### Where the refit is worse, and why that trade is right

**E10, 4.79 % against 3.66 %**, and **the per-pair bond-length residual**, where the refit wins only 18 % of pairs: the median is 0.298 eV against 0.286 eV and 636 pairs exceed 1 eV against 553. Both are the lemma of §6 — a non-negative mixture cannot follow the steepening of the reference at the bottom of the fit window.

The trade is right for two reasons. The window analysis of §7 puts the operating range above 19 eV for every pair and above 150 eV for 95 % of them, so E10 is below the region where V is ever alone. And the residual at the bond measures how hard the network has to work, not correctness, since the learned part is trained with V present and absorbs it; at 0.298 eV it is 3.8 times better than what the `zbl` mode gives, and its *worst* case, 3.18 eV, is better than the published table's 3.89 eV. The second half of that sentence — that a smaller residual makes the learned part's job easier — is an inference from the shape of the problem, not a measured training result (§16).

Named pairs, V at the equilibrium bond length in eV:

| pair  | r_eq / Å | ZBL   | published NLH | refit |
| ----- | -------- | ----- | ------------- | ----- |
| O-O   | 1.208    | 11.66 | 1.55          | 2.04  |
| N-N   | 1.098    | 13.84 | 2.24          | 2.99  |
| Si-O  | 1.620    | 4.83  | 0.98          | 1.33  |
| C-C   | 1.540    | 3.02  | 0.38          | 0.49  |
| Fe-Fe | 2.480    | 1.34  | 0.017         | 0.017 |
| W-W   | 2.460    | 3.13  | 0.159         | 0.180 |
| Al-Al | 2.860    | 0.32  | **2.97**      | 0.19  |
| Mg-Mg | 3.197    | 0.15  | **2.60**      | 0.075 |
| B-Mg  | 3.150    | 0.11  | **3.08**      | 0.067 |

The refit is 25 to 35 % higher than the published table on light covalent pairs and fifteen to forty times lower on exactly the rows whose tails are broken.

If the residual mattered more than the high-energy accuracy, floor 10 / cap 30 would give a median of 0.271 eV and a worst case of 2.27 eV — better than the published table on both — at an E100 of 1.86 % instead of 1.43 %. That setting was not adopted because it loses on the metric that governs the region V actually owns.

### Why not patch only the bad rows

Considered and rejected. The defect is continuous, not binary: 603 pairs exceed 1 meV at 5 Å and the distribution has no natural cut. Fixing a row requires refitting it anyway. A table in which 600 rows came from a different procedure than the other 3678 is harder to defend and harder to document than one uniform fit. And taking the v4 package instead fixes one row and leaves 602 pairs above 1 meV at 5 Å (§2.4).

## 11. B-Ne

The authors state that they fitted B-Ne (Z1 = 5, Z2 = 10) to MP2 rather than DMol because of a problem with the DMol result for that pair. `nlh_evaluate.py bne` confirms this and shows it is a genuine outlier rather than a judgement call. Comparing the DMol and MP2 screening functions over the pure-repulsion range (V ≥ 100 eV) for the 324 pairs that have MP2 data, B-Ne deviates by up to 27.5 %, the next worst pair by 7.7 %, and the median over all of them is 3.0 %. The DMol curve is anomalously low at 0.05-0.3 Å — `phi = 0.459` against MP2's 0.556 at 0.1 Å — while the neighbouring pairs B-F, B-Na, Be-Ne and C-Ne agree with MP2 to within 0.3-2.6 % there.

The shipped table therefore fits B-Ne to the MP2 screening function, under the same objective and the same constraints. That gives E30 3.18 % and E100 3.06 % against MP2, against the published table's 3.36 % and 3.75 %, with `V(5 Å) = 3.0e-7 eV`. Fitting it to DMol instead gives 10.9 % and 12.0 % against MP2, which is clearly the wrong reference. The exception is recorded in the data file's metadata and in the module that loads it, so it is not silently lost.

Na-O needs no special handling. The defect there was in the *fit*, not the data, so refitting from the unchanged DMol curve subsumes the erratum: the shipped row gives E30 1.71 % with `V(5 Å) = 6.9e-4 eV`.

## 12. Coverage above Z = 92

The reference data covers Z ≤ 92; the preset type map covers all 118 elements, so 2743 of the 7021 unordered pairs have no quantum-chemical reference at all.

**Extrapolating the coefficients in Z was measured and rejected.** Holding out every pair with a charge in 89-92 (362 pairs), building the model from pairs with both charges ≤ 88 (3916 pairs), and scoring the prediction against the DMol data with the same metric (`nlh_evaluate.py zext`):

| source of the row                                           | E30 median % | E30 p99 % | E100 median % | E1000 median % |
| ----------------------------------------------------------- | ------------ | --------- | ------------- | -------------- |
| ZBL universal                                               | 15.17        | 51.8      | 11.04         | 3.61           |
| extrapolated in Z, symmetric polynomial degree 1 in Z^(1/3) | 14.48        | 49.5      | 9.68          | 3.60           |
| degree 2                                                    | 16.94        | 41.9      | 12.31         | 4.58           |
| degree 3                                                    | 16.48        | 43.6      | 11.72         | 4.38           |
| nearest pair in Z                                           | 15.76        | 60.3      | 9.97          | 2.79           |
| the actual fitted row                                       | **2.01**     | **8.7**   | **1.30**      | **0.98**       |

The best variant closes about 5 % of the gap between ZBL and the truth, and half the variants are worse than ZBL. The machinery was not built.

**What ships instead: those rows are ZBL in substance.** Every pair touching Z > 92 is the ZBL universal curve put through the same constrained fit. Measured fidelity to the ZBL curve itself, over all 2743 rows (`nlh_refit.py fit`): E30 median 0.717 % with a maximum of 1.062 %, E100 median 0.525 %, E1000 median 0.433 %. ZBL's own error against the DFT reference is about 15 % (the first row of the table above), so a 0.7 % refit error is noise on top of it. A user selecting `nlh` for a system containing a transuranic element is getting ZBL for that pair, and the public documentation says so.

**The reason for refitting rather than pasting ZBL coefficients is the tail bound, not accuracy.** Plain ZBL rows do not meet it: Og-Og gives 2.8 meV at 5 Å and 0.18 meV at 6 Å, O-Og gives 3.8 meV and 0.43 meV, against bounds of 1 meV and 0.1 meV. Refitting them gives one uniform table format, one code path, and the same hard guarantee on every row of the table rather than on 61 % of it. The coefficients vary smoothly with Z across the refitted range — the four rates of the homonuclear rows rise monotonically in every slot, from 60.27, 14.55, 5.89, 2.85 Å⁻¹ at Np-Np to 63.67, 15.06, 6.02, 2.91 Å⁻¹ at Og-Og — so nothing pathological appears.

This is consistent with the standing decision that elements absent from the training data are not an acceptance criterion; the user's own words on accepting it were that these elements are absent from OMat24 and the other large public datasets anyway, so nobody will use them.

**A first version of this experiment was wrong and is recorded as a warning.** It compared the curves on a grid of *reduced* radii `x = r/a_screen` spanning 0.02 to 12, which for U-U reaches only 0.99 Å — far short of `r(V = 30 eV) = 1.53 Å` — and so concluded that ZBL was accurate to 0.2-2 % for the held-out pairs. The reduced radius corresponding to a fixed energy grows with Z, so one reduced grid cannot serve both ends of the periodic table. The corrected experiment scores in physical units and is the one tabulated above.

## 13. Runtime packaging

The coefficients ship as a data file, `deepmd/dpmodel/atomic_model/nlh_coefficients.npz`, holding the element pair, the four unitless amplitudes and the four rates in Å⁻¹ for all 7021 pairs, plus a provenance string. Computing them at model-build time is not an option: the fit needs the 10 MB reference package and takes about ten CPU-hours.

Measured by `nlh_evaluate.py runtime`: the file is 389 KiB and loads in about 2.5 ms, once. Building the `((T+1)^2, 8)` kernel table from it by gather takes 1.20 ms for the 118-element preset map, giving 443 KiB in float32, and 0.02 ms for a two-element map — once per model build. There is precedent in the package: `deepmd/dpmodel/utils/lebedev_rules.npz` at 203 KiB is already shipped and packaged by the existing wheel configuration. The file stores `a_k` rather than `A_k`, and the `k_e Z1 Z2` factor is applied at table-build time from the repository's own constant.

The float64 table is kept alongside the float32 one, because the closed-form (non-fused) evaluation route reads it and the existing cross-check against the analytic ZBL formula holds to 1e-10; the float32 cast the kernels require would only hold to about 1e-7. Both are declared config-derived so neither lands in a checkpoint.

**The fused kernels need no change.** `source/op/pt/dpa4c/*` consume `[A_1..A_4, c_1..c_4]` generically — `Z`, the screening length, `0.88534`, `0.23` and the ZBL coefficients appear nowhere under `source/op/`. Verified by execution: a refitted table packed into the existing layout and evaluated through the repository's own `_reference_pair_energy` reproduces the intended `V(r)` to within 1.1e-6 relative over 0.3-6 Å for H-H, H-O and Fe-Fe. The C++ and LAMMPS paths contain no reference to the table at all; `pair_table` is a non-persistent buffer that is lifted as a tensor constant into the exported artifact, and the mode string survives in the archive's `model.json`.

Two implementation notes worth keeping. The `mode` string already round-trips through serialize and deserialize, and the pt_expt build path *constructs* its wrapper by that round-trip, so a mode that failed to serialize would be silently dropped back to the default at build time rather than at reload. And the CPU kernel clamps `c_k r` at 80 while CUDA does not; the shipped table reaches 1000 Å⁻¹ on a few hundred pairs against ZBL's 38.6, but the clamp changes `phi` by exactly zero over all pairs for r in [0.02, 1] Å, because a clamped term contributes below 1e-35 while float32 resolves 1.2e-7.

`mode` gains a second value and no new serialized key, so no `@version` bump is needed; `deepmd/utils/argcheck.py` declares `mode` and `bridging_method` as plain strings with no choice list, so no schema change is needed either.

## 14. Provenance and licensing

**Reference data.** K. Nordlund, G. Hobler and S. Lehtola, *Data sets for the publication "Repulsive interatomic potentials calculated at three levels of theory"*, Zenodo, version 1.0, released 16 November 2024, doi [10.5281/zenodo.14172633](https://doi.org/10.5281/zenodo.14172633), archive `nlh_potentials_opendata.tar.gz`, md5 `ee14b690a86a37ee6a55a181561930c9`, 10.6 MB. Concept doi (always the newest version) [10.5281/zenodo.14172632](https://doi.org/10.5281/zenodo.14172632); current version v4 of 25 August 2026, doi [10.5281/zenodo.22092149](https://doi.org/10.5281/zenodo.22092149), md5 `89186315fbc9a34ec376d36a8802f52b`. Licensed CC BY 4.0. The fit uses v1.0, and §2.4 records that v1.0 and v4 differ in exactly one coefficient row, which the refit does not use.

**Functional form.** K. Nordlund, S. Lehtola and G. Hobler, *Repulsive interatomic potentials calculated at three levels of theory*, Phys. Rev. A **111**, 032818 (2025), doi [10.1103/PhysRevA.111.032818](https://doi.org/10.1103/PhysRevA.111.032818), published by APS under CC BY 4.0; erratum Phys. Rev. A **112**, 059901 (2025), doi [10.1103/cdrk-x7my](https://doi.org/10.1103/cdrk-x7my), same licence.

**Which rows come from where**, in the shipped table of 7021 rows:

| rows                             | count | reference                                                   |
| -------------------------------- | ----- | ----------------------------------------------------------- |
| Z1 ≤ 92 and Z2 ≤ 92, except B-Ne | 4277  | the self-consistent DFT (DMol) pair energies of the package |
| Z1 = 5, Z2 = 10 (B-Ne)           | 1     | the Hartree-Fock MP2 screening function of the package      |
| Z1 > 92 or Z2 > 92               | 2743  | the ZBL universal potential                                 |

**What CC BY 4.0 requires of us**, as a reading of the licence text and not as legal advice: attribution with licence identification, an explicit indication that the material is modified (§3(a)(1)(B)), no implication of endorsement, and nothing that would restrict downstream recipients. CC BY 4.0 has no ShareAlike clause, so it imposes nothing on the surrounding LGPL code: the coefficients remain CC BY while the source stays LGPL.

Section 3(a)(2) permits the attribution to be given in any manner reasonable to the medium, including by pointing at a resource that carries it. It is carried in three places in the published tree, each complete on its own, and none of them points at this document or at any private path:

1. a provenance block at the top of `deepmd/dpmodel/atomic_model/inner_potential.py`, the module that loads the table — the fuller analogue of the one-line source comment that attributes `lebedev_rules.npz` in `deepmd/dpmodel/utils/lebedev.py`;
1. the `meta` entry inside `nlh_coefficients.npz` itself, so the attribution travels with the numbers if the file is copied out of the tree, readable through `nlh_provenance()`;
1. a note in `doc/model/dpa4c.md` and `doc/model/dpa4.md`.

Each states the authors, the article and dataset DOIs with the dataset version, the erratum, the licence and its URI, that the shipped numbers are this project's own refit and not the published NLH coefficients, and the two exceptions. A per-directory licence file was considered and rejected: the project has none anywhere in the Python package and attributes bundled data by source comment, and naming a third-party attribution in `pyproject.toml`'s `license-files` would misstate the wheel metadata, since that field declares the licence the *project* is offered under.

Raw numerical data may not attract copyright in every jurisdiction, but attribution is given regardless, because the licence is stated and the authors ask for it.

## 15. Rejected alternatives and negative results

Recorded so they are not re-tried.

**A floor on the exponents (`b_k >= b_min`).** Sufficient for the tail bound and simpler to state, but merely sufficient: it forbids the slow small-amplitude term that carries the 1-10 eV region, and one global floor would have to be set by the heaviest pair. Superseded by the energy bound of §4.3, which is the actual requirement. The per-pair variant of the floor is also strictly more restrictive than the inequality, because only the aggregate matters at the check radii.

**Mixed-sign amplitudes with explicit shape constraints.** A free-sign fit with `phi > 0`, V decreasing and the tail bound imposed on a dense grid out to 12 Å was tested on eight pairs. It was better than the non-negative fit on four of them and much worse on two, where SLSQP failed to converge. Rejected because it trades a structural guarantee for a grid-checked one, reintroduces float32 cancellation, and showed no consistent gain. The conclusion rests on eight pairs.

**Unconstrained-sign fits without shape constraints.** Superb inside the window and catastrophic outside it: H-H reaches an E30 of 0.03 % and then gives `V(5 Å) = -0.114 eV`, an *attractive* tail; Fe-Fe -0.476 eV; W-W -1.36 eV. The mirror image of the published pathology, from the same cause.

**Extrapolating coefficients in Z above 92.** Measured and worth nothing (§12). Separately, the NLH coefficients are a non-unique parametrisation — different `(a, b)` sets give nearly the same `phi` — so extrapolating them directly is unsound in principle as well; that part is an inference, the measurement is the held-out table.

**A log-energy objective.** To first order it is the same objective, since `(phi_fit - phi_ref)/phi_ref` and `ln(phi_fit/phi_ref)` agree at the 1-10 % error level the fits reach. Measured on six pairs with an eight-parameter SLSQP fit under identical constraints: better by 0.2-1.0 pp at E30 and *worse* by 0.1-1.7 pp at E100. Rejected mainly because it is nonlinear in `a` and destroys the variable-projection structure that makes the constrained fit globally solvable for 7021 pairs.

**A bound at 4 Å.** Measured on 306 pairs; costs 8 % of the E100 median for a factor two in the worst 4 Å energy, and is unnecessary because the true pair energy there is a hundred times larger and attractive (§9).

**Patching only the defective published rows, or taking the v4 package.** §10.

**The first Z-extrapolation experiment** used a reduced-radius grid that did not reach the energies being scored and gave a misleading answer (§12).

**The first three-term and two-term sweeps** left the unused slots pinned at the top of the rate range with free amplitudes, which is a contact term rather than a reduced fit; they were rerun after the fitter was corrected (§4.4).

## 16. What was not verified

No training run was done. Everything here is a property of the potential, not of a trained model. In particular, the claim that a smaller residual at the bond length makes the learned part's job easier is an inference from the shape of the problem, not a measured training result, and the `nlh` mode has not been shown to train better than `zbl`.

The DMol curves for the Fr-containing pairs are the refit's worst cases at 6-7 % E100; whether the reference data itself is noisy there was not investigated.

The mixed-sign alternative was tested on eight pairs, the log objective on six, and the 4 Å bound on 306.

The reproduction of the paper's error columns is close but not exact (§3); the residual difference from their grid convention was not chased.

The robustness of the outer search was checked on one pair (§4.5), not surveyed.

Fits with five or more positive terms were not tried, so the lemma of §6 was not probed for how much more freedom would relieve it — though the kernel offers only four slots, so it would be academic.

## Appendix A — scripts, and how to run them

Both live in `debug/nlh/` and are private; nothing in the published tree imports from or depends on them, and no public document references them.

`nlh_refit.py` is self-contained, deterministic and release-grade. It takes the unpacked Zenodo package and produces exactly the shipped table, including the B-Ne and Z > 92 exceptions, verifying the constraints before it writes:

```bash
python debug/nlh/nlh_refit.py fit \
    --data /path/to/nlh_potentials_opendata \
    --out deepmd/dpmodel/atomic_model/nlh_coefficients.npz \
    --jobs 60
```

About eleven minutes on sixty processes. `--lo-ev`, `--cap-ev`, `--terms`, `--eps5` and `--eps6` expose the objective and the tail bound, which is how the sweeps of §8 and §9 are regenerated; the defaults are the shipped values. `nlh_refit.py report --data ... --table ...` recomputes a table's accuracy and tail figures on its own.

`nlh_evaluate.py` reproduces the measurements this document quotes, in named sections — `metric`, `window`, `pathology`, `erratum`, `compare`, `terms`, `bne`, `zext`, `runtime`:

```bash
python debug/nlh/nlh_evaluate.py \
    --data /path/to/nlh_potentials_opendata \
    --table deepmd/dpmodel/atomic_model/nlh_coefficients.npz \
    --v4 /path/to/nlh_potentials_opendata_v4 \
    --alt three_term=/path/to/three_term.npz
```

with no section name it runs every section that has the data it needs. `--alt name=path` adds a table to the `compare` section, which is how the Pareto tables are assembled: generate each variant with `nlh_refit.py fit` and its own settings, then pass them all in. `--stride` scores a subset when a quick answer is enough. `compare` takes about a minute per table and `zext` about three minutes; the rest are seconds. Run it from `debug/nlh/`, since it imports `nlh_refit`.

## Appendix B — data

The open-data package `nlh_potentials_opendata.tar.gz` (Zenodo 10.5281/zenodo.14172633 v1.0), and for §2.4 the current `nlh_potentials_opendata_v4.tar.gz` (Zenodo 10.5281/zenodo.22092149 v4). Within them:

- `dmol/original_data/energies.Z1.Z2` — the self-consistent DFT pair energies, r in Å and energy in eV with the energy at 1000 Å set to zero. 4278 files, 53 to 107 grid points each, from 0.002 Å outward. The fit's reference.
- `mp2/screening_mp2_Z1_Z2.dat` — the Hartree-Fock/MP2 screening functions, available for Z1 + Z2 ≤ 36. The B-Ne reference.
- `nlh/nlh_coeffs.dat` — the published coefficients, `Z1 Z2 a1 b1 a2 b2 a3 b3 E30 E10`. Used only for comparison.
- `nlh/nlhpot.c` — the authors' reference evaluator, which fixes the Coulomb constant at `k_e = 14.3996454687 eV Å`.
- `zbl/`, `zblspec/`, `zbluniv/` — the ZBL pair-specific and universal data, used only as context; the ZBL-derived rows of the shipped table are fitted to the universal formula evaluated directly, not to these files.
