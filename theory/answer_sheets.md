# Theory paper — answer sheets (final results only)

Full working is in [answers.md](answers.md); every number is checked by `python theory/check_numbers.py`.

## T1
- **A.1 (iii)** $q_1$ = "How does weight-decay strength affect the generalisation gap of transformer language models across sizes from 10M to 10B parameters?" $\succeq$ $q_2$ = the same question for a 125M-parameter model.
  **(iv)** For existential "Can X improve Y?" questions one witness for the narrow question also answers the broad one, so the narrower question subsumes the broader one, not the reverse.
- **A.2** Angles $0^\circ,30^\circ,60^\circ$: $u=(1,0)$, $v=(0.866,0.5)$, $w=(0.5,0.866)$; $\cos(u,v)=0.866$, $\cos(v,w)=0.866$, $\cos(u,w)=0.500$.
- **B.1** Invisible: role swap (and any near-miss made purely by reordering tokens). Visible: constraint flip, goal swap, inquiry change, direction flip.
- **C.1** Control set: length-flipped scope pairs (broader question is the longer one). Statistic: scope-order accuracy on those pairs (≈0 % for a length model).
- **C.2** (ii) Direction decided by: the volume ratio $\mathrm{Vol}(A)/\mathrm{Vol}(B)$ alone. (iii) $A=[0,10]\times[0,1]$, $B=[9.9,10.9]\times[0,1]$: $\mathrm{Vol}(A)=10\,\mathrm{Vol}(B)$, $P(B\subseteq A)=0.1$.

## T2
- **A.1** G1 How far is x from the closest thing already written? (1, 9) · G2 Is x in a sparsely populated region of a reference population? (2, 6) · G3 Does x reach outside or join what the field spans? (3, 4) · G4 Has exactly this been done before? (5) · G5 Would a reader call it new? (8) · G6 Did the field later go where x went? (7) · G7 Did independent people build on x? (10)
- **B.1** $Y(j)=1-1/\sqrt m$, $N(j)=1-j/m$; satisfies the gradient axiom: $N$.
- **B.2** $Y=1-\sqrt{j/m}$, $N=1-\sqrt j/m$.
- **B.3** Satisfies: $Y$. Violates: $N$.
- **B.4** $M=1-\tfrac12\Big[\tfrac1m\sum_i\max_{p\in P}\cos^+(c_i,p)+\tfrac1m\max_{p\in P}\big(\sum_i\cos^+(c_i,p)\big)^2\Big]$
- **C.1** Reliability $=0.529$. Largest correlation $=0.727$.
- **C.2** Partial correlation $=0.181$.

## T3
- **A** (1) Evaluate on length-matched pairs. (2) Control length statistically (length-only baseline / covariate).
- **B** Over all battles $=0.68$. Within systems $=0.50$.
- **C(i)** Share already asked $=0.915$.
- **C(ii)** G1: 100 % already asked → 10.0 % realised later, 0 net-new. G2: 5 % already asked → 0.88 % realised later. G1 tops the leaderboard.
- **D** (1) Rewrite both sides through the same model. (2) Same-author-type pairs with quality labels (human–human, model–model). (3) Swap model families and capability levels. (4) Minimal model polish of human questions.
- **E(ii)** Pairs needed $=1225$. Half-width with 400 pairs $=0.049$.

## T4
- **A.1** Expected in-plane share $=(k-1)/d=29/768=0.0378$.
- **B** Estimated true angle $\approx40.9^\circ$.
- **C** (1) Writing quality/explicitness → match on length and extraction self-agreement; rewrite all questions through one model. (2) Reference-list density → subsample $R$ to fixed $k$ with abstracts; match on count and recency. (3) Topic mix → within-topic comparisons.
- **D(ii)** Count proximity only when $q$ is not already asked, not answered by $R$, and has an off-plane residual above the decoy baseline.

## T5
- **A.2** $Q_1$: $v=0.531$, $H(Y)=1.000$ · $Q_2$: $v=0.000$, $H(Y)=1.000$ · $Q_3$: $v=0.025$, $H(Y)=0.494$
- **B.1** $Q_\text{near}$: "Does increasing augmentation strength lower the optimal batch size in contrastive pre-training?" ($\pi=0.5,a=0.9,b=0.1$, $v=0.531$). $Q_\text{far}$: "Does the order in which GPU nodes are allocated change the final accuracy of contrastive pre-training?" ($\pi=0.5,a=b=0.6$, $v=0$). Distance ranks $Q_\text{far}$ higher; value ranks $Q_\text{near}$ higher.
- **B.2** (1) Reports-hidden control. (2) Identical question text with reports that make it open vs settled. (3) Test on a measurement family never seen in training.
