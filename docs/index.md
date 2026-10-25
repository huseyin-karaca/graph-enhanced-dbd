---
template: home.html
title: Graph-Enhanced Transformer for Duplicate Bug Report Detection
---

## The problem

Large bug trackers fill up with reports of the same defect. Roughly a fifth of Eclipse's reports and a third of
Firefox's end up marked as duplicates, and every one of them costs a triager the time it takes to notice. The task
is to decide, for a pair of reports, whether they describe the same underlying bug — hard because two people
rarely choose the same words for the same thing.

Transformer encoders are good at exactly that kind of matching, and a Siamese BERT is the standard answer. But it
only learns from pairs somebody has already labelled, and those are rare: a repository with tens of thousands of
reports may have confirmed duplicate links for a few per cent of them. Worse, the pairs you *could* form grow
quadratically — 20,000 reports make about 200 million pairs — so exhaustive supervision is off the table even if
the labels existed. A purely supervised model therefore trains on a sliver of the corpus and never sees the rest.

<ul class="pg-points">
  <li><b>Labels are scarce</b><span>Only a small fraction of reports carry a confirmed duplicate link, and
  labelling more means paying a developer to read bug reports.</span></li>
  <li><b>Pairs are quadratic</b><span>Supervision has to be sampled, so most of the corpus never appears in a
  training pair at all.</span></li>
  <li><b>The rest is wasted</b><span>Those unpaired reports still carry vocabulary, phrasing and component
  context that a pairwise objective cannot reach.</span></li>
</ul>

## What we did

We kept the pairwise transformer, and added a second path that every report can take. The whole corpus becomes a
graph — one node per report, train, validation and test alike — with edges from title similarity. A two-layer GCN
passes messages over that graph, so a report that is in no training pair still sits in the neighbourhood of
reports that are, and still shifts in the embedding space when they do.

--8<-- "fig_method.md"

The parts worth naming:

- **Pairs.** Reports in the same duplicate group give the positive pairs. Negatives are sampled two ways: against
  each anchor in roughly the number of its positives, plus 5,000 extra pairs drawn at random across groups.
- **Encoder.** A weight-tied BERT runs over each report's title and description; the `[CLS]` vector is projected
  down to 128 dimensions, which is what keeps the thing trainable on one GPU.
- **Graph.** Titles are encoded with `bert-base-uncased`, and here the surprise: cosine similarity on the raw
  768-dimensional vectors is almost useless, because unrelated bugs score high. Reducing to **ten** principal
  components first gives a similarity that actually separates. Edges come from that reduced space, with no
  threshold applied.
- **Propagation.** `gnn3.py` is a plain GCN — self-loops, symmetric normalisation, 128-d out — over node features
  injected from the transformer.
- **Fusion and loss.** The two embeddings are mixed, `z = λ·z_tr + (1−λ)·z_gnn`, and trained with a cosine
  embedding loss: pull duplicates together, push sampled negatives apart beyond a margin.
- **Decision.** A pair is a duplicate when `cos(z_i, z_j) ≥ θ`, with θ grid-searched on validation in steps of
  0.05.

Supervision is applied only to training-split nodes. Validation and test nodes are in the graph so that messages
can flow through them, which makes this a **transductive** setup — a real limitation, and one we come back to
below.

## The two repositories

Eclipse Platform and Mozilla Thunderbird, the standard benchmarks for this task.

<div class="pg-stats">
  <div class="pg-stat"><p class="pg-stat__corpus">Eclipse</p><div class="pg-stat__row"><span class="pg-stat__label">F1, test pairs</span><span class="pg-stat__value">0.924</span></div></div>
  <div class="pg-stat"><p class="pg-stat__corpus">Thunderbird</p><div class="pg-stat__row"><span class="pg-stat__label">F1, test pairs</span><span class="pg-stat__value">0.957</span></div></div>
  <div class="pg-stat"><p class="pg-stat__corpus">Graph size</p><div class="pg-stat__row"><span class="pg-stat__label">Eclipse nodes</span><span class="pg-stat__value">68,124</span></div></div>
  <div class="pg-stat"><p class="pg-stat__corpus">Overhead</p><div class="pg-stat__row"><span class="pg-stat__label">per epoch, vs. BERT</span><span class="pg-stat__value">+12%</span></div></div>
</div>

--8<-- "tbl_dataset.md"

## Results

### It ties the baselines

Against four transformer baselines — BERT, RoBERTa, DistilBERT and CodeBERT, each trained the same way — the
graph-enhanced model lands in the same place.

--8<-- "fig_performance.md"

--8<-- "tbl_performance.md"

On Thunderbird it reaches F1 0.957, the same as BERT, RoBERTa and DistilBERT to three decimals. On Eclipse it
reaches 0.924, level with RoBERTa and CodeBERT and 0.005 behind DistilBERT. The one place it stands out is
**recall on Eclipse, 0.935 — the highest of any model tested** — bought with the lowest precision, which is what
you would expect if the graph is pulling in neighbours a text-only encoder would have missed.

!!! note "What this is and is not"
    These numbers say the graph branch costs nothing in accuracy. They do **not** show it winning. The claim the
    architecture was built for — that propagation pays off when labels are scarce — is not tested here, because
    these runs use about 80% of the training data. Section [What we would fix](#what-we-would-fix) says what the
    missing experiment is.

### Where the errors are

--8<-- "fig_confusion.md"

Both test sets hold 10,000 pairs, but they are not balanced the same way: Eclipse is close to even at 5,390
duplicates, while two-thirds of Thunderbird's pairs are duplicates. Some of the gap between the two F1 columns is
that difference rather than anything about the model, which is worth holding on to when reading them side by
side.

### What it costs

--8<-- "tbl_time.md"

The graph adds 10–12% to the time per epoch, and the overhead holds steady between the 68K-report Eclipse corpus
and the 26K-report Thunderbird one, so it scales with the data rather than against it. The design also allows the
GCN to be dropped at inference and the transformer used alone for retrieval against stored embeddings — which is
what you would actually deploy, since it needs no graph rebuild when a new report arrives.

### Training dynamics

--8<-- "fig_convergence.md"

Loss falls smoothly, training and validation F1 converge towards each other rather than apart, and the extra
capacity does not produce overfitting. It also does not produce a gain: the curves sit where RoBERTa's sit.

### What the embedding space looks like

--8<-- "tbl_similarity.md"

Duplicate pairs end up near 1 and unrelated pairs near 0 or below, which is the separation the cosine objective
is asking for — and which the PCA step was needed to get at all.

## What we learned

<div class="pg-cards pg-cards--3">
  <span class="pg-card"><b>RQ1 · Unlabelled data</b><span>Yes — unlabelled reports can take part as graph nodes,
  and the learned space separates duplicates cleanly. But only after PCA cuts the title embeddings from 768
  dimensions to 10, which ties the method to the training distribution.</span></span>
  <span class="pg-card"><b>RQ2 · Scalability</b><span>Yes, within reason. 10–12% extra training time, steady
  across two corpus sizes, and the GCN can be left out at inference. Holding a whole repository's graph in
  memory is the ceiling.</span></span>
  <span class="pg-card"><b>RQ3 · Competitiveness</b><span>It matches the baselines; it does not beat them. F1
  0.957 on Thunderbird and 0.924 on Eclipse, with the best Eclipse recall of the five models.</span></span>
</div>

The honest summary is that the architecture works and the hypothesis is untested. Putting unlabelled reports in a
graph is cheap, stable and costs nothing in accuracy — so the mechanism is sound. Whether it *helps* is a
question about the label-scarce regime, and these runs were not label-scarce.

## What we would fix

**Run it with few labels.** The experiment the project is missing: retrain with 5% or 10% of the duplicate
labels and see whether the gap to a transformer-only baseline opens up. Everything else is secondary to this.

**Justify the hyperparameters.** PCA *d* = 10, projection *D* = 128, the fusion weight λ, the margin — all fixed
from preliminary runs, none ablated. Dropping from 768 to 10 dimensions is aggressive enough that the trade-off
between denoising and information loss deserves a curve, not an anecdote.

**Build edges from descriptions too.** Titles are short and often vague. Weighting title and description
similarity, or using separate edge types for each, should give a graph with more in it.

**Mine hard negatives.** Random cross-group negatives are mostly easy. Curriculum or similarity-based sampling
would sharpen the decision boundary where it actually matters.

**Test on more than two repositories.** Eclipse and Thunderbird are both large, open-source and long-running.
Smaller projects, proprietary trackers and different reporting cultures are all unverified.

**Say something statistical.** One split, no *k*-fold, no significance test. Margins of 0.005 between models are
not distinguishable from seed noise on this evidence, which is exactly why the results above are described as a
tie and not a win.

## How to run it

Everything is in the repository; the dataset downloads from Google Drive. A single A100 (80 GB) ran the reported
experiments.

=== "Colab"

    ```python
    !git clone https://github.com/huseyin-karaca/graph-enhanced-dbd.git
    %cd graph-enhanced-dbd
    !python colab_setup.py   # deps, dataset, extraction
    ```

    Or open the [prepared notebook](https://colab.research.google.com/drive/1zmkY9WIJXK6_2CytcZt5yTbcehKXdtST).

=== "Local"

    ```bash
    git clone https://github.com/huseyin-karaca/graph-enhanced-dbd.git
    cd graph-enhanced-dbd
    pip install -r requirements.txt
    gdown https://drive.google.com/uc?id=1ef68tggrWtdrUv-Bb2l-E2yUVlo8yKgA
    mkdir -p datasets && tar -xzf dataset.tar.gz -C datasets/
    ```

Then train either side of the comparison. Both scripts take the same flags:

```bash
# the graph-enhanced model
python train-ours.py --device gpu --batch_size 64 --epochs 5 \
    --dataset_name thunderbird

# a transformer baseline
python train-baselines.py --device gpu --batch_size 64 --epochs 5 \
    --dataset_name eclipse --model_name roberta-base
```

Checkpoints, metrics, curves and confusion matrices land in
`checkpoints/{model}_{dataset}/{timestamp}/`. The GCN model needs `graph_adj.npz` and `graph_adj_node_ids.npy`
in the dataset directory; the baselines do not.

## The team

Kerem Bayramoğlu, Hüseyin Karaca, Emirhan Koç, Hasan Erkin Ünlü and Rabia Ela Ünlü — listed alphabetically by
surname, as in the report. Written for **CS 588, Data Science for Software Engineering**, Bilkent University,
Fall 2025–26. The report went through a round of peer review within the course and the version linked here is the
revised one.

## Citation

```bibtex
@misc{bayramoglu2026graphdbrd,
  author       = {Bayramo{\u{g}}lu, Kerem and Karaca, H{\"u}seyin and Ko{\c{c}}, Emirhan and
                  {\"U}nl{\"u}, Hasan Erkin and {\"U}nl{\"u}, Rabia Ela},
  title        = {Graph Enhanced Transformer for Semi-Supervised Duplicate Bug Report Detection},
  year         = {2026},
  note         = {CS 588 Data Science for Software Engineering term project, Bilkent University},
  howpublished = {\url{https://github.com/huseyin-karaca/graph-enhanced-dbd}}
}
```
