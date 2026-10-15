# JEPA Crop Classification Research

## Purpose, End Goal, Research Directions & Experimental Specifications

------------------------------------------------------------------------

## 1. Research Purpose

The purpose of this research is to investigate **Joint Embedding
Predictive Architectures (JEPA)** for self-supervised visual
representation learning in the **crop classification / agricultural
image understanding** domain.

The starting point is **I-JEPA (Image Joint Embedding Predictive
Architecture)** and an existing teammate implementation
(`Crop_JEPA_4.ipynb`) that experiments with crop images.

The goal is **not simply to apply I-JEPA to a crop dataset**. The goal
is to identify a scientifically meaningful modification to the JEPA
learning framework and validate it experimentally.

The research should investigate whether a modified JEPA can improve:

-   crop-classification accuracy;
-   Macro-F1 and class-wise performance;
-   representation quality;
-   label efficiency, especially under limited labeled data;
-   robustness to agricultural visual variation;
-   computational efficiency, where relevant.

The final objective is to turn the strongest validated idea into a
**research-paper contribution** supported by controlled experiments,
ablations, multiple seeds, statistical analysis, and a defensible
explanation of why the method works.

------------------------------------------------------------------------

# 2. End Goal

> **Develop and experimentally validate a novel JEPA-based
> self-supervised learning methodology for agricultural/crop
> representation learning that improves downstream crop classification
> under limited labeled data, while providing a clear and experimentally
> supported explanation for the improvement.**

The final method should ideally be:

1.  **Novel** --- not merely a direct application of an existing JEPA
    variant.
2.  **Agriculture-aware** --- motivated by useful structure in crop
    imagery.
3.  **Self-supervised** --- representation learning should not depend on
    manual class labels.
4.  **Label-efficient** --- strong performance at 1%, 5%, 10%, 25%, etc.
    label availability.
5.  **Computationally reasonable**.
6.  **Ablation-friendly** --- each proposed component can be
    independently tested.
7.  **Publication-worthy** --- the contribution can be stated as a
    hypothesis, method, experiment, and conclusion.

------------------------------------------------------------------------

# 3. Current Starting Point: I-JEPA

I-JEPA predicts the latent representation of a target image region from
a context region rather than reconstructing target pixels.

Conceptually:

``` text
Image
  |
  +-----------------------+
  |                       |
Context region        Target region
  |                       |
Context Encoder       Target Encoder
  |                       |
  +----------+------------+
             |
          Predictor
             |
      Predicted target
             |
             v
      Latent-space loss
```

The target encoder is updated through an EMA of the context encoder.

The important research principle is that the model learns semantic
representations in latent space rather than being forced to reconstruct
low-level pixels.

------------------------------------------------------------------------

# 4. What the Current Teammate Notebook Does

The uploaded `Crop_JEPA_4.ipynb` is the experimental starting point.

It compares configurations including:

-   Random initialization;
-   Supervised ViT;
-   Original I-JEPA;
-   Multi-scale I-JEPA;
-   Vegetation-aware + Multi-scale I-JEPA.

The current Crop-JEPA modifications are primarily:

1.  **vegetation-aware target placement** using an ExG-type vegetation
    score;
2.  **multi-scale target blocks**.

The vegetation score is:

\[ ExG = 2G - R - B \]

The preliminary run showed a promising low-label result.

### Preliminary notebook result

  Model                               10% Label Accuracy   100% Label Accuracy
  --------------------------------- -------------------- ---------------------
  Random-init                                     19.44%                27.96%
  Supervised ViT                                  13.33%                23.52%
  Original I-JEPA                                 24.81%                32.59%
  Multi-scale I-JEPA                              23.33%                31.85%
  Vegetation + Multi-scale I-JEPA             **28.52%**            **33.52%**

The preliminary 10%-label improvement of Vegetation + Multi-scale over
Original I-JEPA is:

\[ 28.52 - 24.81 = `\mathbf{+3.71pp}`{=tex} \]

The corresponding preliminary Macro-F1 improvement was approximately:

\[ 28.02 - 23.73 = `\mathbf{+4.29pp}`{=tex} \]

### Critical interpretation

These are **not final research results**.

The notebook uses a short/debug-style training setup, so the gain must
be revalidated with:

-   longer training;
-   multiple seeds;
-   controlled preprocessing;
-   systematic label fractions;
-   stronger baselines.

The useful research clue is:

> **Agricultural structure-aware target selection may help especially
> when labels are scarce.**

------------------------------------------------------------------------

# 5. What Should NOT Be Claimed as Novel by Itself

A simple:

> I-JEPA + crop classification

is not enough.

Likewise, these ideas are already sufficiently explored that they should
not be the sole novelty claim:

-   basic latent prediction;
-   ordinary multi-block masking;
-   merely changing target size;
-   generic multi-scale representation;
-   attention-guided masking;
-   generic difficulty-guided masking;
-   simply adding intermediate/deep supervision;
-   simply increasing predictor depth;
-   simply applying JEPA to remote sensing/agriculture.

The novelty must come from a **specific research hypothesis and
experimentally demonstrated mechanism**.

------------------------------------------------------------------------

# 6. TOP 5 RESEARCH DIRECTIONS

## Ranking

  ---------------------------------------------------------------------------------------------------------------
  Rank       Direction                            Novelty   Feasibility        Impact Recommendation
                                                Potential                             
  ---------- ------------------------------ ------------- ------------- ------------- ---------------------------
  **1**      Site-Consistent Multi-View              9/10          6/10         10/10 **Primary direction**
             JEPA                                                                     

  **2**      Scale--Depth Aligned                    8/10          8/10          9/10 **Best architecture
             Hierarchical JEPA                                                        direction**

  **3**      Crop-Prior +                            7/10         10/10          8/10 **Run immediately**
             Prediction-Difficulty Adaptive                                           
             Masking                                                                  

  **4**      Scale/Difficulty-Conditioned            7/10          7/10          9/10 **Best efficiency
             Adaptive Predictor                                                       direction**

  **5**      Ground-Level ↔ Satellite                9/10          4/10         10/10 **High-risk/high-reward**
             Cross-Modal JEPA                                                         
  ---------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 7. Direction 1 --- Site-Consistent Multi-View JEPA

## Priority: #1

### Core hypothesis

Agricultural datasets can contain multiple photographs of the **same
agricultural site** from different viewpoints.

Instead of treating these images as unrelated examples, use them as
naturally corresponding views.

Instead of only:

``` text
View A
  |
  v
JEPA
```

learn:

``` text
                 SAME AGRICULTURAL SITE
                         |
          +--------------+--------------+
          |              |              |
        North          South           East
          |              |              |
          +--------------+--------------+
                         |
                shared crop semantics
```

The model learns that crop identity should remain stable despite
viewpoint changes.

------------------------------------------------------------------------

## 7.1 Proposed objective

Standard within-view JEPA:

\[ L\_{intra} = D(P(E_c(x_A)), E_t(x_A)) \]

Cross-view JEPA:

\[ L\_{cross} = D(P(E_c(x_A)), E_t(x_B)) \]

where (x_A) and (x_B) are different views of the same site.

Combined objective:

\[ L = `\lambda`{=tex}*{intra}L*{intra} +
`\lambda`{=tex}*{cross}L*{cross} \]

------------------------------------------------------------------------

## 7.2 Architecture

``` text
View A
  |
  v
Context Encoder
  |
  v
Predictor
  |
  v
Predicted latent
      |
      | compare
      v
EMA Target Encoder <----- View B
```

The important distinction is that View B is a **naturally corresponding
observation**, not merely a random augmentation.

------------------------------------------------------------------------

## 7.3 Why it could work

Same-site views contain:

-   different camera directions;
-   different local backgrounds;
-   different framing;
-   different visible crop regions;
-   illumination variation.

But they share the underlying agricultural site/crop semantics.

Therefore the model may learn a representation that is more invariant to
viewpoint and nuisance appearance.

### Research question

> Does natural same-site cross-view prediction produce better crop
> representations than standard within-image JEPA training?

------------------------------------------------------------------------

## 7.4 Dataset implication

The LUCAS dataset is particularly interesting because a site can contain
multiple perspectives, including North, South, East, West, point and
cover photographs.

The implementation must preserve:

-   site ID;
-   image ID;
-   viewpoint;
-   crop label.

Do **not** flatten the data into class folders and lose the site
relationship.

------------------------------------------------------------------------

## 7.5 Critical split rule

Splits must be **site-disjoint**.

Bad:

``` text
Site X North → train
Site X South → test
```

Correct:

``` text
Site X → train
Site Y → validation
Site Z → test
```

Otherwise the experiment risks site-level leakage.

------------------------------------------------------------------------

## 7.6 Required experiments

Compare:

``` text
A. Original I-JEPA
B. Same-view I-JEPA
C. Cross-view JEPA
D. Same-view + Cross-view JEPA
```

Then evaluate:

``` text
1%
5%
10%
25%
50%
100%
```

Also test missing-view robustness:

``` text
6 views
4 views
2 views
1 view
```

------------------------------------------------------------------------

## 7.7 Expected contribution

Possible contribution statement:

> A site-consistent multi-view JEPA objective that exploits naturally
> corresponding agricultural observations to learn viewpoint-robust crop
> representations.

------------------------------------------------------------------------

## 7.8 Rating

  Attribute                  Rating
  ------------------------ --------
  Novelty                      9/10
  Agricultural relevance      10/10
  Scientific story            10/10
  Feasibility                  6/10
  Publication potential        9/10

------------------------------------------------------------------------

# 8. Direction 2 --- Scale--Depth Aligned Hierarchical JEPA

## Priority: #2

### Core hypothesis

Different spatial scales of crop imagery may correspond to different
semantic levels.

``` text
Fine
 ↓
leaf texture / local morphology

Medium
 ↓
plant structure / row structure

Coarse
 ↓
field organization / broad crop context
```

Therefore explicitly align:

\[ `\text{spatial target scale}`{=tex} `\leftrightarrow`{=tex}
`\text{encoder depth}`{=tex} \]

------------------------------------------------------------------------

## 8.1 Proposed architecture

Use one shared ViT:

``` text
Input
 |
Patch Embedding
 |
Layer 1
 |
Layer 2
 |
Layer 3
 |
Layer 4  ------> Fine representation
 |
Layer 5
 |
Layer 6
 |
Layer 7
 |
Layer 8  ------> Medium representation
 |
Layer 9
 |
Layer 10
 |
Layer 11
 |
Layer 12 ------> Coarse representation
```

Define:

\[ z_S=E\^{(4)}(x) \]

\[ z_M=E\^{(8)}(x) \]

\[ z_L=E\^{(12)}(x) \]

------------------------------------------------------------------------

## 8.2 Multi-scale objective

\[ L = `\lambda`{=tex}\_S D(`\hat `{=tex}z_S,z_S) + `\lambda`{=tex}\_M
D(`\hat `{=tex}z_M,z_M) + `\lambda`{=tex}\_L D(`\hat `{=tex}z_L,z_L) \]

This explicitly couples target scale with semantic depth.

------------------------------------------------------------------------

# 8.3 Nested/coarse-to-fine predictor

This is the stronger version of the original "nested encoder/predictor"
idea.

``` text
Context
   |
   v
Coarse Predictor
   |
   v
Coarse prediction
   |
   v
Medium Predictor
conditioned on coarse prediction
   |
   v
Medium prediction
   |
   v
Fine Predictor
conditioned on medium representation
   |
   v
Fine prediction
```

Mathematically:

\[ `\hat `{=tex}z_L=P_L(c,m_L) \]

\[ `\hat `{=tex}z_M=P_M(c,`\hat `{=tex}z_L,m_M) \]

\[ `\hat `{=tex}z_S=P_S(c,`\hat `{=tex}z_M,m_S) \]

The model therefore predicts:

\[ `\text{coarse}`{=tex} `\rightarrow `{=tex}`\text{medium}`{=tex}
`\rightarrow `{=tex}`\text{fine}`{=tex} \]

rather than treating each target scale as independent.

------------------------------------------------------------------------

## 8.4 Required ablation

``` text
1. Original I-JEPA
2. Multi-scale I-JEPA
3. Multi-depth supervision
4. Scale-depth aligned JEPA
5. Scale-depth aligned + nested predictor
```

This isolates the contribution of each architectural change.

------------------------------------------------------------------------

## 8.5 Expected contribution

> A scale-depth aligned JEPA framework that couples agricultural
> target-region scale with representation depth, optionally using a
> coarse-to-fine latent predictor.

------------------------------------------------------------------------

## 8.6 Rating

  Attribute                     Rating
  --------------------------- --------
  Novelty                         8/10
  Architecture contribution      10/10
  Agricultural relevance          9/10
  Feasibility                     8/10
  Publication potential           9/10

------------------------------------------------------------------------

# 9. Direction 3 --- Crop-Prior + Prediction-Difficulty Adaptive Masking

## Priority: #3

This direction directly extends the existing teammate implementation.

### Core hypothesis

The current ExG rule assumes:

> vegetation-rich regions are useful targets.

But the model itself can estimate which regions are difficult to
predict.

Combine:

1.  vegetation prior;
2.  prediction difficulty;
3.  random exploration.

------------------------------------------------------------------------

## 9.1 Difficulty score

For patch (i):

\[ d_i\^{(t)} = `\beta `{=tex}d_i\^{(t-1)} + (1-`\beta`{=tex})
\|`\hat `{=tex}z_i-z_i\| \]

This tracks how difficult the patch is for JEPA.

------------------------------------------------------------------------

## 9.2 Combined target score

\[ S_i = `\alpha `{=tex}V_i + `\beta `{=tex}D_i + `\gamma `{=tex}R_i \]

where:

-   (V_i) = vegetation/crop prior;
-   (D_i) = prediction difficulty;
-   (R_i) = exploration component.

Target sampling then uses (S_i).

------------------------------------------------------------------------

## 9.3 Why this is useful

A patch can be highly vegetated but easy to predict.

Another patch can have a moderate vegetation score but contain
difficult, discriminative crop structure.

The research question becomes:

> Can JEPA learn which agricultural regions are informative while
> retaining an agriculture-specific prior?

------------------------------------------------------------------------

## 9.4 Curriculum version

Early:

``` text
80% random
20% guided
```

Later:

``` text
30% random
70% guided
```

This prevents the model from immediately becoming dependent on the
handcrafted vegetation prior.

------------------------------------------------------------------------

## 9.5 Important novelty qualification

Dynamic/difficulty-guided masking and attention-guided masking already
exist in related SSL/remote-sensing research.

Therefore the contribution should be specifically framed around:

> **crop-domain adaptive target sampling combining an agricultural
> prior, online JEPA prediction difficulty, and multi-scale target
> selection.**

------------------------------------------------------------------------

## 9.6 Ablations

``` text
A. Random masking
B. Vegetation-only
C. Difficulty-only
D. Vegetation + difficulty
E. Vegetation + difficulty + random exploration
F. Static vs curriculum sampling
```

------------------------------------------------------------------------

## 9.7 Rating

  Attribute                  Rating
  ------------------------ --------
  Novelty                      7/10
  Implementation ease         10/10
  Agricultural relevance      10/10
  Compute efficiency           9/10
  Publication potential        8/10

This is the **best immediate experiment** because it builds directly on
the existing notebook.

------------------------------------------------------------------------

# 10. Direction 4 --- Scale/Difficulty-Conditioned Adaptive Predictor

## Priority: #4

### Core hypothesis

A fixed predictor spends the same amount of computation on every target.

Instead, allocate predictor capacity according to:

-   target scale;
-   prediction difficulty;
-   or both.

------------------------------------------------------------------------

## 10.1 Example

``` text
Easy/small target
        |
        v
2 predictor blocks

Medium target
        |
        v
4 predictor blocks

Hard/large target
        |
        v
6 predictor blocks
```

Formally:

\[ D_p =
```{=tex}
\begin{cases}
2 & \text{easy}\\
4 & \text{medium}\\
6 & \text{hard}
\end{cases}
```
\]

------------------------------------------------------------------------

## 10.2 Research question

> Can JEPA maintain or improve representation quality while reducing
> average predictor computation through conditional computation?

------------------------------------------------------------------------

## 10.3 Required metrics

Accuracy metrics:

-   Accuracy;
-   Macro-F1;
-   per-class F1;
-   linear probe;
-   k-NN.

Efficiency metrics:

-   FLOPs;
-   GPU memory;
-   training time;
-   GPU-hours;
-   samples/sec;
-   tokens/sec;
-   parameter count;
-   average predictor depth.

The final result should include an **accuracy-vs-compute curve**.

------------------------------------------------------------------------

## 10.4 Novelty qualification

Simply changing predictor depth is not enough.

The contribution must be:

> **adaptive/conditional predictor computation based on target scale or
> learned prediction difficulty.**

------------------------------------------------------------------------

## 10.5 Rating

  Attribute                 Rating
  ----------------------- --------
  Novelty                     7/10
  Efficiency relevance       10/10
  Accuracy potential          7/10
  Difficulty                  7/10
  Publication potential       8/10

------------------------------------------------------------------------

# 11. Direction 5 --- Ground-Level ↔ Satellite Cross-Modal JEPA

## Priority: #5

### Core hypothesis

Learn correspondence between:

``` text
Ground-level crop image
          ↕
Satellite observation of the same site
```

The two modalities observe agriculture at dramatically different spatial
scales.

------------------------------------------------------------------------

## 11.1 Proposed architecture

``` text
Ground Image                         Satellite Image
     |                                      |
     v                                      v
Ground Encoder                       Satellite Encoder
     |                                      |
     +----------------+---------------------+
                      |
                      v
               Latent predictors
                      |
                      v
              Cross-modal JEPA
```

------------------------------------------------------------------------

## 11.2 Objective

\[ L = `\lambda`{=tex}*g L*{ground} + `\lambda`{=tex}*s L*{satellite} +
`\lambda`{=tex}*c L*{cross} \]

The cross-modal term forces compatible latent representations between
the ground and satellite observations.

------------------------------------------------------------------------

## 11.3 Expected information

Ground imagery:

-   plant morphology;
-   local crop structure;
-   visible characteristics.

Satellite imagery:

-   field-scale organization;
-   spectral information;
-   broader spatial context;
-   potentially temporal information.

The research question is:

> Can cross-resolution latent prediction create crop representations
> that connect plant-level and field-level information?

------------------------------------------------------------------------

## 11.4 Data requirements

This direction requires:

-   geographic coordinates;
-   reliable site matching;
-   temporal matching;
-   satellite imagery;
-   cloud filtering;
-   spatial alignment;
-   geographic train/test separation;
-   potentially Sentinel-2 or similar multispectral data.

------------------------------------------------------------------------

## 11.5 Why it is high-risk

Agricultural remote sensing already has multimodal and multitemporal
foundation-model research.

Therefore the novelty must specifically be:

> **ground-to-satellite cross-resolution latent prediction using a
> JEPA-style objective**

rather than simply "multimodal agricultural learning."

------------------------------------------------------------------------

## 11.6 Rating

  Attribute                     Rating
  --------------------------- --------
  Novelty                         9/10
  Scientific impact              10/10
  Agricultural relevance         10/10
  Implementation difficulty      10/10
  Data difficulty                10/10
  Publication potential          10/10

This is the **highest-risk/highest-reward** option.

------------------------------------------------------------------------

# 12. Recommended Experimental Strategy

Do not immediately build the largest architecture.

## Phase 1 --- Validate the existing observation

Run:

``` text
Original I-JEPA
        vs
Vegetation-only
        vs
Multi-scale
        vs
Vegetation + Multi-scale
```

Use:

-   3 random seeds;
-   5%, 10%, 25%, 100% labels;
-   identical preprocessing;
-   identical optimizer;
-   identical token budgets;
-   full training schedule.

If the preliminary vegetation-aware improvement survives, it becomes a
legitimate foundation for further work.

------------------------------------------------------------------------

# 13. Phase 2 --- Establish a Strong Baseline

The publication baseline should include:

-   Original I-JEPA;
-   fixed train/validation/test split;
-   multiple seeds;
-   full training;
-   linear probe;
-   k-NN;
-   Macro-F1;
-   class-wise F1;
-   confusion matrix;
-   representation statistics.

The current short/debug run is not enough for the final paper.

------------------------------------------------------------------------

# 14. Phase 3 --- Test Each Direction Independently

Do not combine everything at once.

Test:

``` text
A. Cross-view JEPA

B. Scale-depth JEPA

C. Adaptive crop-aware target sampling

D. Adaptive predictor computation

E. Ground-satellite JEPA
```

The purpose is to discover **which scientific hypothesis actually
works**.

------------------------------------------------------------------------

# 15. Phase 4 --- Combine Only Proven Components

If two components independently demonstrate strong value, combine them.

Examples:

``` text
Cross-view JEPA
+
Scale-depth JEPA
```

or:

``` text
Vegetation prior
+
Prediction difficulty
+
Multi-view learning
```

Do not combine every idea simply to maximize accuracy.

The final architecture should remain interpretable.

------------------------------------------------------------------------

# 16. Final Ablation Matrix

A possible final paper can include:

  ---------------------------------------------------------------------------------------
  Model           Multi-scale        ExG   Difficulty   Cross-view       Depth     Nested
                                                                     alignment 
  ------------- ------------- ---------- ------------ ------------ ----------- ----------
  Original                 No         No           No           No          No         No
  I-JEPA                                                                       

  Crop-JEPA               Yes        Yes           No           No          No         No

  Adaptive                Yes        Yes          Yes           No          No         No
  Crop-JEPA                                                                    

  Scale-Depth             Yes         No           No           No         Yes         No
  JEPA                                                                         

  Nested                  Yes         No           No           No         Yes        Yes
  Scale-Depth                                                                  
  JEPA                                                                         

  Cross-View               No         No           No          Yes          No         No
  JEPA                                                                         

  Final              Selected   Selected     Selected     Selected    Selected   Selected
  proposed                                                                     
  method                                                                       
  ---------------------------------------------------------------------------------------

The final row should contain only components that independently
demonstrate value.

------------------------------------------------------------------------

# 17. Evaluation Framework

## Downstream classification

Report:

-   Top-1 accuracy;
-   Macro-F1;
-   precision;
-   recall;
-   per-class F1;
-   confusion matrix.

## Label efficiency

Use:

``` text
1%
5%
10%
25%
50%
100%
```

The key question is:

> How much labeled data does the proposed representation require to
> reach a given performance level?

A strong result could look like:

``` text
5% labels + proposed method
        ≈
25% labels + baseline
```

## Representation quality

Use:

-   linear probing;
-   k-NN;
-   UMAP;
-   feature variance;
-   covariance;
-   singular-value spectrum;
-   effective rank;
-   representation mean norm;
-   predictor-target similarity.

## Efficiency

Where relevant report:

-   parameters;
-   FLOPs;
-   GPU memory;
-   training time;
-   GPU-hours;
-   samples/sec;
-   tokens/sec;
-   inference time;
-   predictor depth.

------------------------------------------------------------------------

# 18. Collapse Monitoring

The current notebook uses representation standard deviation as a basic
collapse signal.

For the research version, also track:

### Per-dimension variance

\[ Var(z_j) \]

### Covariance

\[ Cov(z) \]

### Singular-value spectrum

\[ `\sigma`{=tex}\_1,`\sigma`{=tex}\_2,`\ldots`{=tex},`\sigma`{=tex}\_d
\]

### Effective rank

\[ R\_{eff} \]

### Feature mean norm

\[ \|`\mu`{=tex}\_z\| \]

### Predictor-target similarity

These are particularly important when experimenting with:

-   nested predictors;
-   intermediate losses;
-   adaptive masking;
-   conditional computation.

------------------------------------------------------------------------

# 19. What Would Constitute a Successful Research Contribution?

The target should not simply be:

> "Our model gets higher accuracy."

A strong contribution should establish:

``` text
Research hypothesis
        ↓
Specific modification
        ↓
Controlled experiment
        ↓
Ablation
        ↓
Improvement
        ↓
Explanation of mechanism
```

For example:

> Natural same-site views contain complementary information about crop
> identity. Cross-view latent prediction forces the encoder to preserve
> site-level crop semantics while becoming less dependent on
> viewpoint-specific appearance, resulting in stronger low-label crop
> representations.

That is a scientific contribution.

------------------------------------------------------------------------

# 20. Preferred Paper Directions

## Option A --- Primary recommendation

### Natural Multi-View JEPA for Crop Representation Learning

Core contribution:

\[ `\boxed{
\text{within-view JEPA}
+
\text{same-site cross-view latent prediction}
}`{=tex} \]

Potential extension:

\[ + `\text{scale-aware target selection}`{=tex} \]

This has the cleanest agricultural motivation.

------------------------------------------------------------------------

## Option B --- Architecture-focused recommendation

### Scale--Depth Aligned Hierarchical JEPA for Crop Representation Learning

Core contribution:

\[ `\boxed{
\text{spatial target scale}
\leftrightarrow
\text{representation depth}
}`{=tex} \]

Potential extension:

\[ `\boxed{
\text{coarse}
\rightarrow
\text{medium}
\rightarrow
\text{fine latent prediction}
}`{=tex} \]

This is the strongest option if the paper is expected to contribute
primarily a new architecture.

------------------------------------------------------------------------

# 21. Recommended Overall Research Path

``` text
Existing Crop-JEPA notebook
            |
            v
Validate vegetation-aware result
            |
            v
Establish strong I-JEPA baseline
            |
            +---------------------+
            |                     |
            v                     v
   Multi-view JEPA       Scale-depth JEPA
            |                     |
            +----------+----------+
                       |
                       v
                Best components
                       |
                       v
          Adaptive target sampling
                       |
                       v
             Adaptive predictor
                       |
                       v
              Final architecture
                       |
                       v
             Full ablation study
                       |
                       v
             Statistical analysis
                       |
                       v
                 Research paper
```

The core principle is:

> **Do not engineer a complicated architecture first and then search for
> a justification. Start with a scientific hypothesis, isolate the
> mechanism, test it, and only then build the final architecture.**

------------------------------------------------------------------------

# 22. Immediate Next Experiment

The first controlled experiment should be:

``` text
Original I-JEPA
        vs
Vegetation-only I-JEPA
        vs
Multi-scale I-JEPA
        vs
Vegetation + Multi-scale I-JEPA
```

Use:

-   3 seeds;
-   5%, 10%, 25%, 100% labels;
-   identical training conditions;
-   full training schedule;
-   identical preprocessing.

If the vegetation-aware gain survives, use it as the foundation for the
next experiment.

The next major experiment should then be:

> **Same-site Cross-View JEPA on a dataset that preserves natural
> agricultural viewpoints.**

This currently offers the strongest combination of:

-   scientific motivation;
-   agricultural relevance;
-   novelty potential;
-   clean ablation;
-   label-efficiency relevance;
-   compatibility with JEPA's latent-prediction philosophy.

------------------------------------------------------------------------

# 23. One-Sentence Research Mission

> **To investigate and develop a novel JEPA-based self-supervised
> representation learning framework that exploits the intrinsic spatial,
> multi-view, and semantic structure of agricultural imagery to achieve
> more accurate, label-efficient, robust, and potentially more
> computationally efficient crop classification, with the final
> objective of producing a rigorously validated research-paper
> contribution.**

------------------------------------------------------------------------

# 24. Important Novelty Disclaimer

The five directions above are **candidate research directions**, not
guaranteed novelty claims.

Before submitting a paper, the final literature search must be repeated
against the latest papers, preprints, conference proceedings, and
closely related agricultural/remote-sensing foundation models.

The paper should claim novelty only after establishing that the exact
proposed mechanism has not already been published.

------------------------------------------------------------------------

## Key research areas to monitor

-   I-JEPA
-   V-JEPA / V-JEPA 2.x
-   hierarchical/deep JEPA
-   masked-image self-supervised learning
-   attention-guided masking
-   difficulty-guided masking
-   agricultural remote-sensing foundation models
-   LUCAS crop classification
-   CropSTS and agricultural JEPA approaches
-   multimodal agricultural representation learning
-   ground-to-satellite representation learning
