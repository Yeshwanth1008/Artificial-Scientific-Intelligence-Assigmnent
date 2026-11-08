# Theory paper — full working (T1–T5)

Notation follows the paper. I write $\cos^+(a,b)=\max(\cos(a,b),0)$.
Every number below is recomputed by `python theory/check_numbers.py` (32 values plus randomized checks of the identities).
Final results only are collected in [answer_sheets.md](answer_sheets.md).

---

## T1 — Identity, scope and limits of one vector

### A.1

**(i)** ($\Leftarrow$) If $\mathcal A(q_1)\subseteq\mathcal A(q_2)$, every complete answer to $q_1$ is already a complete answer to $q_2$, which is the definition of $q_1\succeq q_2$.
($\Rightarrow$) If $q_1\succeq q_2$, each $a\in\mathcal A(q_1)$ yields (entails) a complete answer $b\in\mathcal A(q_2)$. Since $\mathcal A(q_2)$ is closed upwards under entailment, $a\in\mathcal A(q_2)$. So $\mathcal A(q_1)\subseteq\mathcal A(q_2)$. $\square$

**(ii)** $\subseteq$ is reflexive and transitive, so $\succeq$ is a preorder. $q_1\equiv q_2$ means each answers the other, i.e. $\mathcal A(q_1)=\mathcal A(q_2)$; set equality is reflexive, symmetric and transitive, so $\equiv$ is an equivalence relation. The symmetric part of $\succeq$ is $\mathcal A(q_1)\subseteq\mathcal A(q_2)\wedge\mathcal A(q_2)\subseteq\mathcal A(q_1)\iff\mathcal A(q_1)=\mathcal A(q_2)\iff q_1\equiv q_2$. $\square$

**(iii)** $q_1$ = "How does weight-decay strength affect the generalisation gap of transformer language models across sizes from 10M to 10B parameters?"; $q_2$ = "How does weight-decay strength affect the generalisation gap of a 125M-parameter transformer language model?"
A full answer to $q_1$ covers every size in the range, including 125M, so it answers $q_2$. A full answer at 125M says nothing about 10B, so it does not answer $q_1$. Hence $q_1\succeq q_2$ but not $q_2\succeq q_1$.

**(iv)** "Can X improve Y?" is existential: one positive example settles it. "Colour jitter improves contrastive pre-training" answers $q_2$, and because colour jitter *is* a data augmentation it also answers $q_1$. So $\mathcal A(q_2)\subseteq\mathcal A(q_1)$: the topically *narrower* $q_2$ subsumes $q_1$. The converse fails — "random crops help" answers $q_1$ but says nothing about colour jitter. For existential questions, topical breadth and answer-set breadth point in opposite directions, so "the broader topic subsumes the narrower one" is simply wrong here; scope has to be read off the answer set, not the topic.

### A.2

Take angles $0^\circ,30^\circ,60^\circ$: $u=(1,0)$, $v=(0.866,0.5)$, $w=(0.5,0.866)$.
$\cos(u,v)=\cos(v,w)=0.866\ge0.8$, but $\cos(u,w)=0.5<0.8$. "Same at threshold $\tau$" is not transitive.

- **Connected components** take the transitive closure (single-linkage). Chains $u\!-\!v\!-\!w\!-\!\dots$ glue together questions that are clearly different. In a pool of 5 000 generated questions on a few topics this produces a handful of giant components: recall looks great, precision collapses.
- **Cliques** avoid chaining but are not well defined as a partition. A question can sit in several maximal cliques, the result depends on the order of a greedy cover, maximum clique is NP-hard, and the groups jump around under small changes of $\tau$.
- **What I would do instead:** use cosine only to *retrieve* candidates (top-k), decide "same question?" per pair with a calibrated pairwise model (a cross-encoder / reranker), and cluster the resulting signed graph with a method that tolerates intransitive decisions — correlation clustering, or average/complete linkage, or a "leader" scheme where each member must be judged the same as the cluster's canonical question. Then check cluster purity on a hand-labelled sample.

### B.1

The token sequence of "How does Y affect X?" is a permutation of that of "How does X affect Y?" (moving whole spans when X, Y are multi-token). Addition is commutative, so $\sum_i h(w_i)$ is identical and $f(q)=g(\sum_i h(w_i))=f(q')$. No permutation-invariant encoder can separate them. $\square$

| Near-miss type | Invisible? | Reason |
|---|---|---|
| Role swap | **yes** | pure permutation of tokens |
| Constraint flip | no | token multiset changes ("without" → "with") |
| Goal swap | no | a content token changes (accuracy → speed) |
| Inquiry change | no | the question word changes (how → why / can) |
| Direction flip | no | the verb changes (improve → harm) |

"Visible" only means the sum changes — by one token out of ~20, so these edits are visible but weakly. Any near-miss made purely by reordering (e.g. "train on A, test on B" vs "train on B, test on A", or moving a negation to another clause) is also invisible.

### B.2

**(i)** No. A mean of token vectors is dominated by the content tokens, which are the same in both sentences; the position-dependent part of each contextual vector is a small perturbation, and averaging over all tokens shrinks it further. Typical contrastive training (paraphrase pairs, QA pairs, citation pairs, in-batch negatives) rewards putting topically equal texts together and almost never shows a role-swapped hard negative, so confusing $A$ with its role swap costs nothing in the loss. I'd expect $\cos(A,C)$ to be about as high as a paraphrase.

**(ii)** A cross-encoder lets every token of one question attend to every token of the other, so it can align X with X and check that X has the *same role* in both. It computes an arbitrary function of the pair, including asymmetric relations such as "broader than". The cosine of two separately pooled vectors is symmetric, close to transitive on the sphere, and fixed before either text sees the other — it cannot express "same words, different roles".

**(iii)** Literature search wants $A$ and its role swap $C$ *close* (same topic, same papers to read). Identity wants them *far* (different question). The pair $(A,C)$ is pulled together by the topical objective and pushed apart by the identity objective, so a single cosine must trade the two.
*Experiment:* fine-tune one embedder with $L=\lambda L_\text{identity}+(1-\lambda)L_\text{topic}$, where $L_\text{identity}$ uses near-miss hard negatives and paraphrase positives and $L_\text{topic}$ uses citation / co-topic positives. Sweep $\lambda\in\{0,0.1,\dots,1\}$ with 3 seeds. For each $\lambda$ plot topical retrieval quality (nDCG@10 on a citation-retrieval set) on the x-axis against $1-$near-miss error (from E1) on the y-axis. The curve is the trade-off. A two-head model (one topic vector, one identity vector) should sit above the curve, which is the point of C.3.

### C.1

If in training the broader question is usually shorter, then $s(q)=\alpha\,\mathrm{len}(q)+\beta$ with $\alpha>0$ satisfies (nearly) every constraint $s(q_1)<s(q_2)$ while knowing nothing about scope.
**Control set:** length-flipped pairs, where the broader question is the *longer* one (plus length-matched pairs for reference).
**Statistic:** scope-order accuracy on the length-flipped pairs. A length model scores near 0 %; a model that knows scope scores well above 50 %.

### C.2

**(i)** $\dfrac{P(B\subseteq A)}{P(A\subseteq B)}=\dfrac{\mathrm{Vol}(A\cap B)/\mathrm{Vol}(B)}{\mathrm{Vol}(A\cap B)/\mathrm{Vol}(A)}=\dfrac{\mathrm{Vol}(A)}{\mathrm{Vol}(B)}$. $\square$

**(ii)** The *direction* is decided by the volumes alone: whichever box is larger always has the larger containment probability, no matter how little they overlap. The *magnitude* $P(B\subseteq A)$ carries the overlap — how much of $B$ actually lies inside $A$, i.e. whether the two questions are about the same thing at all.

**(iii)** In $\mathbb R^2$: $A=[0,10]\times[0,1]$ (Vol 10), $B=[9.9,10.9]\times[0,1]$ (Vol 1), $A\cap B=[9.9,10]\times[0,1]$ (Vol 0.1). So $\mathrm{Vol}(A)=10\,\mathrm{Vol}(B)$ and $P(B\subseteq A)=0.1$ (and $P(A\subseteq B)=0.01$).
Interpretation: $A$ is far broader and "wins" the direction, yet 90 % of $B$ lies outside it — these are mostly different questions that barely touch. Reading subsumption needs *both* the ratio (which way) and a large magnitude (whether at all).

### C.3

$\Phi_I(q)=(u,\,v,\,\mathrm{Box}(q))$:

1. $u\in S^{d-1}$ — **topic**. A frozen scientific sentence embedding. $\cos(u_1,u_2)$ answers "same topic?" and drives retrieval.
2. $v\in S^{d'-1}$ — **identity**. A second head on the same encoder, trained with near-miss hard negatives and paraphrase positives. $\cos(v_1,v_2)\ge\tau_I$ pre-filters "same question?", which a cross-encoder confirms on the shortlist.
3. $\mathrm{Box}(q)$ — **scope**. One interval block per facet (purpose, mechanism, evaluation, domain); breadth $=\log\mathrm{Vol}$; $P(B\subseteq A)$ answers "is $A$ broader than $B$?".

Same question $\iff$ $\cos(v_1,v_2)$ high *and* both containment probabilities $\ge\gamma$. $A$ broader than $B$ $\iff$ $P(B\subseteq A)\ge\gamma>P(A\subseteq B)$.

**Falsifiable prediction:** a one-facet narrowing (e.g. adding "on ImageNet", a domain edit) shrinks *that facet's* block more than the other three, and does so equally on length-flipped items.
**Test:** on the E1 narrower/broader items and one-facet edits, measure $\Delta\log$width per facet. The geometry passes if the edited facet shrinks most in $\ge60\%$ of edits (chance is 25 %) and scope accuracy on length-flipped items is within 5 points of length-matched items. If the domain block shrinks just as much under a mechanism edit, the facet split is not real and the prediction fails.

---

## T2 — What does "novel" mean?

### A.1

I group the ten measures by the question about the world each one actually answers.

| Group — the question it answers | Measures | What each would wrongly reward |
|---|---|---|
| **G1** How far is $x$, in content or wording, from the closest thing already written? | 1, 9 | (1) off-topic or garbled text is maximally "novel"; (9) new wording, typos and jargon make new bigrams without a new idea |
| **G2** Is $x$ in a sparsely populated region relative to a reference population? | 2, 6 | (2) isolated fringe topics where nobody works, which is not the same as originality; (6) whatever happens to be rare in the chosen population — change the population, change the answer |
| **G3** Does $x$ reach outside, or join together, what the field already spans? | 3, 4 | (3) style/noise directions outside the topic span; (4) incoherent mash-ups of distant concepts get long tours |
| **G4** Has exactly this idea been done before? | 5 | trivial one-facet edits ("same method, new dataset") count as novel |
| **G5** Would a reader or reviewer *call* it new? | 8 | confident, long, buzzword-heavy writing; the judge's memory of the paper; halo from quality |
| **G6** Did the field later go where $x$ went? | 7 | obvious next steps and questions that were already being asked; it rewards predictability, not originality |
| **G7** Did independent people build on $x$? | 10 | popularity, famous authors, hype and venue, all of which drive uptake without novelty |

G6 and G7 both look forward in time, but they are different questions: G6 asks "was this ahead of its time?", G7 asks "did it matter to others?". A re-asked obvious question scores high on G6 and can score zero on G7.

### A.2

"Distance from prior work measures novelty" is a claim about G1 (and is validated by G1-type tests such as pool manipulations). "Distance does not predict reviewers' novelty scores" is a claim about G5. Reviewers answer a different question; their novelty scores are entangled with quality (they correlate 0.70 with the overall recommendation in C.2) and are noisy (reliability ≈ 0.53 in C.1). A measure can track G1 perfectly and still correlate weakly with G5. Both claims are true because they are about different groups.

### B.1 — distributed coverage, $P=\{c_1,\dots,c_j\}$

$\cos(x,c_i)=m^{-1/2}$ for every $i$, so $Y(j)=1-1/\sqrt m$, constant in $j$.
$\max_p\cos(c_i,p)=1$ for $i\le j$ and $0$ otherwise, so $N(j)=1-j/m$.
**$N$ satisfies the gradient axiom; $Y$ does not** (it does not move at all as more claims are covered).

### B.2 — concentrated coverage, $p=j^{-1/2}(c_1+\dots+c_j)$

$\cos(x,p)=m^{-1/2}j^{-1/2}\cdot j=\sqrt{j/m}$, so $Y=1-\sqrt{j/m}$.
$\cos(c_i,p)=j^{-1/2}$ for $i\le j$, $0$ otherwise, so $N=1-\frac1m\,j\cdot j^{-1/2}=1-\sqrt j/m$.

### B.3

Concentration needs (concentrated) < (distributed) for $1<j\le m$.

- $Y$: $1-\sqrt{j/m}<1-1/\sqrt m\iff\sqrt j>1\iff j>1$. **Satisfied.**
- $N$: $1-\sqrt j/m<1-j/m\iff\sqrt j>j$, false for $j>1$. **Violated** — under $N$, putting $j$ claims into one paper makes the idea look *more* novel than spreading them over $j$ papers. $\square$

### B.4 — a measure that satisfies both

Following the hint, use one term for "how much of the idea is covered" and one for "how much does a single paper cover":
$$C=\frac1m\sum_{i=1}^m\max_{p\in P}\cos^+(c_i,p),\qquad K=\frac1m\max_{p\in P}\Big(\sum_{i=1}^m\cos^+(c_i,p)\Big)^2,\qquad M=1-\tfrac12(C+K).$$
Both $C,K\in[0,1]$; for $K$, Cauchy–Schwarz gives $(\sum_i\cos^+)^2\le m\sum_i\cos^2\le m$, since the $c_i$ are orthonormal and $p$ is a unit vector.

- Distributed: $C=j/m$, $K=1/m$, so $M_\text{dist}=1-\frac{j+1}{2m}$ — strictly decreasing in $j$. **Gradient holds.**
- Concentrated: $C=\sqrt j/m$, $K=(j\cdot j^{-1/2})^2/m=j/m$, so $M_\text{conc}=1-\frac{j+\sqrt j}{2m}$.
- $M_\text{conc}<M_\text{dist}\iff j+\sqrt j>j+1\iff j>1$. **Concentration holds for all $1<j\le m$.**

*Paraphrase invariance:* $M$ depends on the idea only through the claim embeddings $c_i$, so it is invariant whenever they are.
*Monotonicity:* for $P\subseteq P'$, every max over $P'$ is at least the max over $P$, so $C$ and $K$ cannot decrease and $M$ cannot increase. $\square$

*Where it breaks on a real corpus:* real claims are not orthogonal, and their extraction is noisy and depends on granularity (splitting one claim into two changes $m$ and the score); a claim's cosine with a whole-paper embedding is low even when the paper does contain it; in dense topics everything has high cosine without actually covering anything; $K$ rewards a broad survey that is loosely related to many claims; a new *combination* of old claims scores as not novel; and scores are not comparable across ideas with different $m$.

### C.1

Each half-panel mean has reliability $r=0.36$. Under the parallel-measurement model, the full-panel mean (average of two halves) has reliability by Spearman–Brown
$$\rho=\frac{2r}{1+r}=\frac{0.72}{1.36}=0.529.$$
No predictor can correlate with the observed full-panel mean more than the observed mean correlates with its own true score, which is $\sqrt\rho=0.727$.

### C.2

$$r_{PN\cdot R}=\frac{r_{PN}-r_{PR}\,r_{NR}}{\sqrt{(1-r_{PR}^2)(1-r_{NR}^2)}}=\frac{0.30-0.25\times0.70}{\sqrt{0.9375\times0.51}}=\frac{0.125}{0.6915}=0.181.$$
Once the overall recommendation is held fixed, the predictor keeps only a weak link to novelty — most of its raw 0.30 came from tracking quality.

### C.3

Papers that every reviewer called novel are almost all highly rated, and papers that every reviewer called not novel are almost all poorly rated. On this contrast novelty and quality are perfectly confounded, so *any* quality score separates the two groups and no score can show it is measuring novelty rather than quality.
A contrast that can: hold quality fixed. Compare papers matched on overall recommendation (or soundness) that differ in novelty — in particular the off-diagonal cells, "novel but weak" versus "not novel but strong". Report each measure's gain over a quality-only baseline on that matched contrast.

---

## T3 — Traps in evaluation

### A — Length

Take a battle with lengths $L>L'$. $S$ picks the longer proposal when $\varepsilon'-\varepsilon<L-L'$. The noise difference is symmetric and continuous and the threshold is positive, so this happens with probability $p>\tfrac12$. Agreement with experts is
$$0.65p+0.35(1-p)=0.35+0.30p>0.5\iff p>\tfrac12,$$
which holds. Yet $\varepsilon$ is independent of everything, so given the two lengths $S$ carries no information about which proposal the experts prefer. (Simulation in `check_numbers.py`: 57 % agreement from length plus noise alone.)

Remedies:

1. **Evaluate on length-matched pairs.** Can miss real signal when length genuinely tracks completeness, and the matched subset may not be representative.
2. **Control length statistically** — report against a length-only baseline (B-LEN), or add length as a covariate. Can miss non-linear length effects and other correlates of length (detail, jargon, structure).

### B — Systems versus ideas

Cross-system battles: the score difference is $1+(n_A-n_B)$ with $n_A-n_B\in(-0.8,0.8)$, so $s$ always picks the $\mathcal A$ idea, and agrees with experts 80 % of the time. Within-system battles: the difference is pure noise, so agreement is 50 %.
**Overall** $=0.6\times0.80+0.4\times0.50=0.68$. **Within systems** $=0.50$.

*System-demeaned accuracy:* subtract each system's mean score before comparing ideas (equivalently, fit preference against score difference with a fixed effect per system), so only within-system variation can earn credit. Here it is 0.50. That number — not 0.68 — answers "can $s$ pick the better of two ideas from comparable systems?". The 0.68 only shows that $s$ recognises which system wrote the idea.

### C — Re-asking

$P(\text{realised})=0.3\times0.10+0.7\times0.004=0.0328$.

**(i)** Share of realised questions that had already been asked $=0.030/0.0328=0.915$.

**(ii)** Generator G1 only restates questions already in the literature: 100 % already asked, so 10.0 % are "realised later" and 0 are net-new. Generator G2 is mostly original: 5 % already asked, realised rate $0.05\times0.10+0.95\times0.004=0.88\%$. G1 tops the leaderboard by a factor of 11 while producing nothing new.

**(iii)** *Corrected metric:* the **net-new realisation rate** — among questions *not* already asked at $T_0$, the share later realised by author-disjoint work that does not cite the source. This controls re-asking per question and lineage leakage.
The before/after contrast of [R24] (matches after $T_0$ minus matches before) instead controls the *matching pipeline's base rate*: vague or popular questions match the past as often as the future, and judge leniency, threshold and topic density affect both windows equally. But it does not remove individual re-asked questions and is confounded by literature growth (there are more papers after $T_0$). The per-question correction fixes what the contrast misses and vice versa, so I'd report both.

### D — Who wrote it?

1. **Rewrite both sides through the same model** (a neutral paraphrase that keeps content). If the 85 % preference stays, the scorer is reading content; if it falls to ~50 %, it was reading the author's style.
2. **Same-author-type pairs with quality labels** (human vs human, model vs model). Above-chance agreement with the labels in both settings → quality. Near 50 % within type → it was an authorship detector.
3. **Swap model families and capabilities.** If the scorer ranks strong generators above weak ones in line with expert labels → quality. A uniform penalty on all model text, or one that tracks the model family rather than capability → authorship.
4. **Minimal-edit polish of human questions by a model** (no content change). A large score drop → authorship.

### E — Contamination and power

**(i)** Two tests of memory versus judgement:

1. *Temporal split:* accuracy on papers decided before $T$ versus after $T$, matched on topic and venue. A clear drop after $T$ suggests memory.
2. *Memorisation probe:* can the model complete the title, or name the venue and decision, from the abstract's first sentence? Check whether accuracy concentrates on the papers it can recall, and re-test on anonymised, paraphrased versions.

**(ii)** One-sample test of a proportion, $p_0=0.5$, $p_1=0.54$, two-sided $\alpha=0.05$, power 80 %:
$$n=\left(\frac{z_{0.975}\sqrt{p_0q_0}+z_{0.8}\sqrt{p_1q_1}}{p_1-p_0}\right)^2=\left(\frac{1.960\times0.5+0.8416\times0.4984}{0.04}\right)^2=1224.03\Rightarrow\mathbf{1225}\text{ pairs}$$
(1227 if the null variance is used throughout).
Half-width with 400 pairs: $1.96\sqrt{0.54\times0.46/400}=\mathbf{0.049}$.

Two ways to gain power without new labels, and how each can mislead:

1. *Use all pairs among the labelled items*, not just disjoint pairs. Misleads because the pairs share items and are not independent; the naive interval is too narrow — a cluster bootstrap over items is needed.
2. *Use graded outcomes* (score margins) instead of win/loss, or stratify by topic. Misleads if the graded labels are noisier than the binary ones, or if the strata are chosen after seeing the results.

---

## T4 — The geometry of grounding

### A.1

Let $V$ be the $(k-1)$-dimensional direction space of the affine span. For an isotropic Gaussian, $\delta=\|\delta\|\theta$ with $\theta$ uniform on $S^{d-1}$ and independent of $\|\delta\|$. In a basis aligned with $V$, the in-plane share is $\|P_V\delta\|^2/\|\delta\|^2=\sum_{i=1}^{k-1}\theta_i^2$, and $\mathbb E[\theta_i^2]=1/d$ by symmetry. So
$$\mathbb E[\text{in-plane share}]=\frac{k-1}{d}=\frac{29}{768}=0.0378.$$
(Monte-Carlo in `check_numbers.py`: 0.0376.)

About 96 % of the deviation is off-plane. Depth, barycentric coordinates and convex-hull membership all live in the 4 % in-plane part, so they are dominated by noise. The hull is a $(k-1)$-dimensional set, measure zero in $\mathbb R^{768}$, so a real $q$ essentially never lies in it. With unit-normalised embeddings it is even stricter: any convex combination of distinct unit vectors has norm $<1$, so a unit-norm $q$ cannot lie in the hull at all.

### A.2

Measure the off-plane part instead: its size relative to $\|\delta\|$ and, more usefully, its *direction* — does the residual point towards papers that came later, or towards relevant work missing from $R$? Compare against (a) the anisotropic null $\mathrm{tr}(P_V\Sigma)/\mathrm{tr}\Sigma$, with $\Sigma$ the empirical same-topic covariance rather than isotropic noise; (b) decoy questions (random same-topic questions) scored against the same $R$; (c) the true $q$ scored against a shuffled $R$.

### B

$r_\text{early}=\cos28^\circ=0.8829$, $r_\text{final}=\cos35^\circ=0.8192$, $\cos\theta_\text{obs}=\cos50^\circ=0.6428$.
$$\cos\theta_\text{true}\approx\frac{0.6428}{\sqrt{0.8829\times0.8192}}=\frac{0.6428}{0.8505}=0.7558\Rightarrow\theta_\text{true}\approx\mathbf{40.9^\circ}.$$
So about 9° of the observed 50° drift is extraction noise.

*Assumption:* extraction noise is independent across the two extractions and independent of the true vector (classical test theory).
*How it can fail:* if both extractions share a bias (same model, same house style), self-agreement is inflated and the correction under-corrects. With low reliabilities it can even produce $\cos\theta_\text{true}>1$, which is meaningless.

### C

1. **Writing quality / explicitness.** Strong-venue papers state their question more explicitly and in a more standard style, so the extracted question is cleaner and sits closer to anything. *Control:* match on question length and on extraction self-agreement; rewrite all questions through one model before scoring.
2. **Reference-list density.** Strong-venue papers cite more, more on-topic and more recent work with available abstracts, so $R$ is denser and any $q$ looks closer to it. *Control:* subsample $R$ to a fixed $k$ with abstracts; match on reference count and recency.
3. **Topic mix.** Strong venues over-represent mainstream topics, which are denser in embedding space. *Control:* compare within topic (matched topic clusters) only.

### D

**(i)** Restate the inputs: "How does [method of $r_1$] perform on [task of $r_2$]?" when that result is already reported, a paraphrase of the centroid of $R$, or one input's own question. Proximity is maximal and nothing new is asked.

**(ii)** *Guard:* count proximity only if $q$ (a) is not the same question as any input or earlier paper (cross-encoder over $R$ plus retrieved literature before $t$), (b) is not already answered by $R$, and (c) has an off-plane residual above the decoy baseline — a novelty floor. *Cost:* extra retrieval and judge calls, false rejections of legitimate incremental questions, a threshold to set, and a new target that a generator can learn to game.

**(iii)** Proximity predicts citations partly because safe, incremental questions get cited; novelty pulls the other way. A single combined score lets a generator trade the cheaper axis invisibly (Goodhart), and hides the trade-off from whoever uses it. Separate axes let an agent choose a point on the frontier deliberately, and let each axis be validated against its own target.

---

## T5 — The value of a question

### A.1

$I(H;Y)=H(Y)-H(Y\mid H)$. $P(Y=1)=\pi a+(1-\pi)b$, so $H(Y)=h(\pi a+(1-\pi)b)$. Given $H=1$, $Y\sim\text{Bern}(a)$; given $H=0$, $Y\sim\text{Bern}(b)$; so $H(Y\mid H)=\pi h(a)+(1-\pi)h(b)$. Hence $v=h(\pi a+(1-\pi)b)-\pi h(a)-(1-\pi)h(b)$. $\square$

### A.2

| | $a$ | $b$ | $\pi$ | $v$ (bits) | $H(Y)$ (bits) |
|---|---|---|---|---|---|
| $Q_1$ | 0.9 | 0.1 | 0.5 | **0.531** | **1.000** |
| $Q_2$ | 0.5 | 0.5 | 0.5 | **0.000** | **1.000** |
| $Q_3$ | 0.9 | 0.1 | 0.99 | **0.025** | **0.494** |

$Q_1$: $h(0.5)-h(0.9)=1-0.469$. $Q_3$: $P(Y=1)=0.892$, $h(0.892)=0.494$, minus $h(0.9)=0.469$.

### A.3

$v=H(Y)-H(Y\mid H)\le H(Y)$ and, by symmetry of mutual information, $v=H(H)-H(H\mid Y)\le H(H)$, since conditional entropies are non-negative. So $v\le\min\{H(H),H(Y)\}$. $\square$

$Q_2$ is the **unpredictable** question: $H(Y)=1$ bit, the maximum, but $a=b$ so the outcome ignores $H$ and $v=0$. $Q_3$ is **already answered**: $H(H)=h(0.99)=0.081$ bits, so $v\le0.081$ however good the measurement. A question is valuable only if the hypothesis is still open *and* the measurement discriminates between the possibilities — surprise alone or a good instrument alone is not enough.

### B.1

Setting: the existing report says "Increasing batch size lowers the optimal augmentation strength in contrastive pre-training." $H$ = "augmentation strength and batch size interact in the opposite direction too".

- $Q_\text{near}$ (a role swap — almost the same words as the report): "Does increasing augmentation strength lower the optimal batch size in contrastive pre-training?" Still open, $\pi=0.5$; the measurement discriminates, $a=0.9$, $b=0.1$. $v=0.531$ bits.
- $Q_\text{far}$ (different wording and topic words): "Does the order in which GPU nodes are allocated change the final accuracy of contrastive pre-training?" Its outcome does not depend on $H$: $a=b=0.6$, $\pi=0.5$. $v=0$.

The embedding-distance score ranks $Q_\text{far}$ as more novel/valuable; the value ranks $Q_\text{near}$ far higher.

*Why this can always happen:* $v$ depends only on $(\pi,a,b)$, and none of these is a function of text distance. $\pi$ lives in what the reports *concluded*, not in their wording, and $(a,b)$ depend on what the measurement is causally connected to. Two near-identical texts can carry any $(\pi,a,b)$, and so can two distant texts, so for any distance ranking one can build a pair with the opposite value ranking.

### B.2

A model that predicts $v$ from (question text + report texts) can learn shortcuts. Evaluation:

1. **Reports-hidden control.** Train and test the same model with the reports removed. If it does as well, it is predicting from priors on the question text and is not reading the reports.
2. **Same question, different reports.** Pairs where the question text is identical but the reports make it open ($\pi\approx0.5$) in one case and settled ($\pi\approx0.99$, or the outcome already reported) in the other. Score the share of pairs ordered correctly; shuffled reports must hurt.
3. **Unseen mechanism family.** Train on one family of measurements (say binary diagnostic tests) and test on a family never seen in training (Gaussian outcomes, multi-hypothesis designs). Report the drop; a large drop means it memorised surface patterns of the training family.

### B.3

With $K$ hypotheses, posterior weights $\pi_k=P(H=k\mid D_R)$ after reading the inputs, and outcome densities $p(y\mid k)$:
$$v(q\mid R)=I(H;Y\mid D_R)=\sum_k\pi_k\int p(y\mid k)\log\frac{p(y\mid k)}{\sum_j\pi_j\,p(y\mid j)}\,dy.$$
"**Already addressed by the inputs**" means $I(H;Y\mid D_R)\approx0$: either $R$ already settles $H$ ($H(H\mid D_R)\approx0$), or $R$ already contains the outcome $Y$ so observing it again adds nothing.

To estimate $v$ for a real research question, a language model would need to output: (1) a set of mutually exclusive hypotheses $\{h_k\}$ the question distinguishes between, (2) calibrated probabilities $\pi_k$ given $R$, and (3) a predictive distribution $p(y\mid h_k)$ for the outcome of the measurement $q$ specifies. $v$ is then computed by Monte Carlo, and the calibration of (2) and (3) has to be checked on outcomes published after the model's cutoff.
