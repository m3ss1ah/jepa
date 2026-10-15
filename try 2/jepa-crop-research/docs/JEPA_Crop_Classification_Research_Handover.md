# Research Handover Report: JEPA for Crop Classification

**Document version:** 1.0  
**Prepared:** 9 October 2026  
**Project status:** Research planning and baseline-design stage  
**Primary objective:** Build a reproducible, compute-aware research pipeline to study and modify Joint-Embedding Predictive Architectures (JEPAs) for agricultural crop classification, identify a defensible research contribution, and develop the evidence needed for a publishable paper.

---

## 1. Executive summary

The project is to investigate whether a carefully designed JEPA-based self-supervised learning method can produce better or more efficient representations for crop classification from satellite observations. The project is **not** to build a farmer-facing application. The main deliverables should be a research implementation, experimental evidence, reproducible code and configurations, and ultimately a paper.

The main strategy is:

1. Understand the canonical I-JEPA learning mechanism and reproduce a small, trustworthy version.
2. Select a crop dataset and evaluation task whose labels, spatial structure and temporal observations match the research question.
3. Establish fair baselines before changing the architecture.
4. Diagnose a measurable weakness—such as label efficiency, missing-observation robustness, class confusion or compute cost.
5. Test **one clearly described architectural or objective modification at a time**.
6. Run ablations, multiple seeds and held-out evaluation; report both predictive quality and compute cost.
7. Verify novelty against the most relevant existing JEPA and Earth-observation papers before making a contribution claim.

### Recommended dataset strategy

- **Main spatial/temporal benchmark:** PASTIS, or PASTIS-R if radar–optical fusion later becomes important. PASTIS provides multitemporal Sentinel-2 data with pixel-level semantic and parcel annotations. It is therefore a dense crop-classification/segmentation problem, not merely a table of crop labels.
- **Fast temporal-classification prototype:** EuroCropsML. It provides parcel-level annual Sentinel-2 time series and crop labels, so a compact temporal encoder and JEPA-style temporal prediction can be prototyped without first building a spatial patch-masking pipeline.
- **Potential India-focused external test:** SICKLE, with careful attention to its sample definitions, annotations and task compatibility.

These datasets are not interchangeable. EuroCropsML is good for a quick temporal-classification prototype; PASTIS is more suitable for testing whether spatial and temporal representations help dense crop mapping. The final task must be explicitly fixed before the main benchmark runs.

### Recommended first research hypothesis

A candidate worth testing is **phenology-aware target selection for JEPA pretraining under limited crop labels and missing temporal observations**. The model would select useful target observations based on unsupervised temporal change signals derived from the imagery, rather than choosing every target uniformly at random. This is only a hypothesis, **not a proven novelty claim**. It must be compared against standard JEPA target sampling and related temporal/spatiotemporal methods, especially AnySat, CropSTS and T-JEPA.

### Hardware decision

- Use the local RTX 3050 Laptop GPU (4 GB VRAM).
- Do not attempt to reproduce original large-scale I-JEPA training. Implement a compact research baseline that preserves the essential JEPA learning mechanism and clearly disclose its scale.

All runtime estimates in this report are planning estimates only. Measure actual step time on the chosen implementation before scheduling larger experiments.

---

## 2. Research purpose and scope

### 2.1 Purpose statement

> To investigate, implement and evaluate a modified Joint-Embedding Predictive Architecture for crop classification from satellite observations, with an emphasis on discovering a specific failure mode of current approaches and testing whether one principled modification improves label efficiency, robustness or computational efficiency. The final result should be supported by fair baselines, ablation studies, reproducible evaluation and a literature-grounded novelty argument.

### 2.2 In scope

- Understanding JEPA fundamentals and the I-JEPA reference implementation.
- Self-supervised pretraining on unlabeled observations and downstream crop classification/segmentation.
- Temporal and optionally spatial/spectral information from satellite data.
- Modifying target selection, masking, prediction conditioning or the latent-space objective.
- Studying low-label performance, missing-date robustness, class-wise quality, geographic transfer and compute cost.
- Reproducible PyTorch notebooks/scripts, experiment tracking, ablations and paper preparation.

### 2.3 Out of scope for the first phase

- Building a production farmer application or insurance-payout workflow.
- Training internet-scale or multi-GPU foundation models.
- Adding multiple architectural components before the contribution of each component is measured.
- Claiming novelty solely because JEPA is used on a crop dataset.
- Reporting only training loss or overall accuracy without baselines, class-aware metrics and reproducible evaluation.

### 2.4 What counts as success?

A successful early phase produces a working pipeline, baseline metrics and a clear diagnosis. A successful research study must go further: it must show that the proposed change addresses a meaningful limitation, outperforms or improves the efficiency of relevant baselines under fair conditions, and does so consistently enough to support the stated claims.

**Publication is not guaranteed by any particular accuracy threshold.** Novelty, sound comparisons, the strength of the empirical evidence, reproducibility and fit with the target venue all matter.

---

## 3. JEPA fundamentals

### 3.1 What JEPA means

JEPA stands for **Joint-Embedding Predictive Architecture**. It is a family of representation-learning methods, not one immutable network architecture. The central principle is to predict a representation of a target observation from a context observation, rather than requiring exact reconstruction of missing pixels.

In a masked-image setting, the model sees a context region and predicts latent representations for target regions that were not visible to the context encoder.

### 3.2 Intuition: pixel reconstruction versus latent prediction

A masked autoencoder (MAE) typically learns to reconstruct missing pixels or input values. This can encourage the model to represent details that are difficult to predict and may not help the downstream task.

A JEPA instead learns to predict a target embedding. The target embedding is a learned representation of the target region. The goal is not to reproduce every pixel; it is to learn useful structure shared between context and target.

```text
MAE-style objective
Image -> hide patches -> encoder/decoder -> reconstruct missing pixels

JEPA-style objective
Image -> context/target split
  context -> context encoder -> predictor -> predicted target embedding
  target  -> target encoder ----------------> target embedding
  compare predicted target embedding with target embedding
```

This does **not** mean JEPA automatically ignores all nuisance information or always beats MAE. Those are empirical questions that must be evaluated on the intended downstream task.

### 3.3 Canonical I-JEPA components

```text
                        ONE IMAGE
                            |
                +-----------+-----------+
                |                       |
                v                       v
          CONTEXT REGION            TARGET REGION(S)
           visible patches          hidden target patches
                |                       |
                v                       v
        +----------------+      +----------------+
        | Context /      |      | Target         |
        | online encoder |      | encoder        |
        | f_theta        |      | f_theta_bar    |
        +----------------+      +----------------+
                |                       |
                v                       v
          context tokens           target tokens
                |                       |
                v                       |
        +----------------+              |
        | Predictor      |              |
        | g_phi          |              |
        +----------------+              |
                |                       |
                v                       v
          predicted target  <---->  target embedding
                \\                    /
                 \---- latent loss ---/
```

The target encoder in original I-JEPA is an exponential-moving-average (EMA) version of the online/context encoder, and it is not updated by ordinary gradient descent. The online encoder and predictor are optimized by gradient descent.

A typical update is:

`target_parameters = momentum * target_parameters + (1 - momentum) * online_parameters`

A conceptual prediction objective is:

`L_predict = distance(predictor(context_embedding, target_position), target_embedding)`

The exact normalization, loss, masking strategy and predictor implementation must be taken from the chosen reference code/paper rather than inferred from this simplified expression.

### 3.4 Why two encoders?

Without constraints, a representation-learning model can collapse to an uninformative constant representation. In I-JEPA, the slowly updated target encoder, asymmetry between context and target paths, predictor and carefully designed training setup help avoid trivial solutions and stabilize learning. Collapse prevention is a core engineering and evaluation concern when modifying the objective.

Later work, notably LeJEPA, studies a different route using SIGReg regularization. Do not assume every JEPA variant uses EMA or that every variant called JEPA has exactly the I-JEPA training objective.

### 3.5 Why masking is a research choice

The original I-JEPA work emphasizes that the scale of target blocks and the informativeness/spatial distribution of context strongly affect learned representations. In a crop dataset, a target policy might also be defined over:

- spatial patches or field areas;
- time points or intervals between acquisitions;
- selected spectral channels or derived spectral signals;
- target observations selected by changes in the vegetation signal.

That flexibility is promising, but target-selection ideas must be compared against existing masking/prediction methods before novelty is claimed.

---

## 4. Research landscape and relevant prior work

This is a **high-priority reading list**, not a claim to be an exhaustive bibliography of every paper using the JEPA name. The field is moving quickly, and the related-work search must be repeated before submitting a paper. Papers with “JEPA” in the title may use different objectives or be related by philosophy rather than reproducing canonical I-JEPA.

### 4.1 Foundations and general JEPA variants

| Work | Main contribution | Why it matters here | Link |
|---|---|---|---|
| LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) | Conceptual proposal for world models and abstract predictive representations | Background philosophy; not a crop-classification method | [Project page](https://jepa.si5.pl/methods/path-autonomous-mi.html) |
| *Joint Embedding Predictive Architectures Focus on Slow Features* (2022) | Studies the slow/predictable-feature perspective | Useful intuition for predictable structure over time | [arXiv:2211.10831](https://arxiv.org/abs/2211.10831) |
| Assran et al., *I-JEPA* (CVPR 2023) | Predicts target-region representations from visible context; highlights the importance of mask design | Canonical image-JEPA baseline and first paper to reproduce conceptually | [Paper](https://arxiv.org/abs/2301.08243) · [Official code](https://github.com/facebookresearch/ijepa) |
| Bardes et al., *MC-JEPA* (2023) | Jointly learns content and motion/optical-flow-related information | Shows that auxiliary predictive objectives can alter the representation | [arXiv:2307.12698](https://arxiv.org/abs/2307.12698) |
| Fei et al., *A-JEPA: Joint-Embedding Predictive Architecture Can Listen* (2023) | Extends JEPA ideas to audio spectrograms and adapts masking to time-frequency structure | Example of designing the target/mask around the data modality | [arXiv:2311.15830](https://arxiv.org/abs/2311.15830) |
| Bardes et al., *V-JEPA* (2024) | Predicts latent representations of masked video content | A temporal visual-learning reference; do not equate video time with irregular satellite acquisition automatically | [arXiv:2405.09987](https://arxiv.org/abs/2405.09987) |
| Littwin et al., *How JEPA Avoids Noisy Features* (NeurIPS 2024) | Theoretical analysis of the implicit bias of latent prediction in a simplified model | Helps explain why predictable, high-influence features may be learned preferentially; theory has assumptions and is not a universal empirical guarantee | [arXiv:2407.03475](https://arxiv.org/abs/2407.03475) |
| Mo & Yun, *DMT-JEPA: Discriminative Masked Targets for JEPA* (2024) | Aggregates semantically related neighboring patches into target representations | Directly relevant to local semantics and dense spatial tasks; target construction is already an active research area | [arXiv:2405.17995](https://arxiv.org/abs/2405.17995) · [Code](https://github.com/DMTJEPA/DMTJEPA) |
| Mo & Tong, *Connecting JEPA with Contrastive Self-Supervised Learning* / C-JEPA (2024) | Combines JEPA-style learning with VICReg-style regularization | Useful comparison for embedding collapse, variance and representation quality | [arXiv:2410.19560](https://arxiv.org/abs/2410.19560) |
| *Enhancing JEPAs with Spatial Conditioning* (2024) | Adds explicit spatial conditioning to JEPA-style prediction | Spatial conditioning is not itself a new idea; any new variant must be compared with this line | [arXiv:2410.10773](https://arxiv.org/abs/2410.10773) |
| Velarde & Parra, *Recurrent JEPA with Recurrent Forward Propagation Learning* (2024) | Combines predictive representation learning with recurrent processing | Potentially relevant to streaming or sequential observations, but task-specific downstream results must be checked | [arXiv:2411.16695](https://arxiv.org/abs/2411.16695) |
| Assran et al., *V-JEPA 2* (2025) | Scales latent predictive learning to over one million hours of internet video and investigates prediction/planning | Important temporal/world-model extension, but not a direct crop baseline | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) |
| Balestriero & LeCun, *LeJEPA* (2025) | Uses Sketched Isotropic Gaussian Regularization (SIGReg) with a JEPA-style objective; argues that teacher/student EMA and several heuristics can be avoided in its formulation | Relevant alternative for the training objective and collapse prevention; it is not the same recipe as original I-JEPA | [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) · [Code](https://github.com/galilai-group/lejepa) |
| Perera et al., *CrossJEPA* (2025) | Uses JEPA-style cross-modal prediction from 3D point-cloud input to 2D image embeddings | Demonstrates that predictive targets need not be masked regions from the same input modality | [arXiv:2511.18424](https://arxiv.org/abs/2511.18424) |

### 4.2 Earth-observation and agriculture work — mandatory comparisons

| Work | Main contribution / scope | Implication for this project | Link |
|---|---|---|---|
| Li et al., *Predicting Gradient is Better: SAR-JEPA* (ISPRS Journal of Photogrammetry and Remote Sensing, 2024) | Uses multi-scale SAR gradient representations as targets to cope with SAR speckle noise and small-target problems in SAR automatic target recognition | Strong example of changing target features to reflect sensor physics; it is SAR target recognition, not crop classification | [Journal DOI](https://doi.org/10.1016/j.isprsjprs.2024.09.013) · [arXiv](https://arxiv.org/abs/2311.15153) · [Code](https://github.com/waterdisappear/SAR-JEPA) |
| Astruc et al., *AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities* (CVPR 2025) | A JEPA-based Earth-observation model with scale-adaptive spatial encoders, trained across heterogeneous EO data; evaluated across tasks including crop-type classification | **Critical baseline/related work.** “Apply JEPA to satellite crop data” is not sufficient novelty. Check its official code, model/data compatibility, training setup and reported crop benchmarks | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html) · [Code](https://github.com/gastruc/AnySat) |
| Yan et al., *CropSTS: A Remote Sensing Foundation Model for Cropland Classification with Decoupled Spatiotemporal Attention* (Remote Sensing, 2025) | Uses temporal-first decoupled spatial/temporal attention and a hybrid self-supervised strategy described as JEPA plus distillation from a DINO vision model; evaluates on PASTIS-R | **Critical crop-specific baseline.** Its temporal/spatial efficiency and boundary/label-efficiency claims overlap with broad ideas one might otherwise propose. Read the method and loss definitions carefully; do not automatically treat it as a canonical I-JEPA reproduction | [Journal paper](https://doi.org/10.3390/rs17142481) |
| Peng et al., *T-JEPA: A Temporal JEPA for Remote-Sensing Representations* (arXiv preprint, submitted 5 October 2026) | Conditions latent transitions on elapsed time for irregular and sparse EO observations; also reports auxiliary reconstruction-related supervision in its method | **Very recent and highly relevant.** A generic “temporal JEPA for remote sensing” claim overlaps this preprint. Compare the exact target-selection, time-conditioning, objective and evaluation choices before proposing a temporal modification | [arXiv:2610.05731](https://arxiv.org/abs/2610.05731) |
| Junghare & Banerjee, *UA-JEPA: Uncertainty-aware joint-embedding predictive learning for remote sensing image retrieval* (2026) | Models latent prediction probabilistically and estimates uncertainty; targets remote-sensing image retrieval | The uncertainty direction is already explored in remote sensing, but this is not automatically a crop-classification baseline. Check the exact publication details and code availability before relying on implementation comparisons | [Journal DOI](https://doi.org/10.1016/j.patrec.2026.08.007) |

### 4.3 Other JEPA-family directions to search before submission

The literature search should also investigate, without assuming that every item transfers directly to crops:

- sparse-latent and structured-representation variants;
- target ordering/saliency-driven prediction, including DSeq-JEPA-style work;
- hyperspectral/spectral JEPA variants;
- JEPA plus diffusion or generative objectives;
- cross-modal and multimodal JEPA;
- domain-specific temporal prediction and time-series JEPA methods;
- crop mapping/foundation models not called JEPA, because they may solve the same problem with a different method.

Some of these names appear in fast-moving preprints or have multiple papers with similar acronyms. Verify the exact title, author list, venue/status, method and code before citing them. This document intentionally prioritizes sources for which a concrete paper or official project page is linked above rather than treating every previously encountered acronym as verified evidence.

### 4.4 The key novelty warning

The following ideas are **not sufficient on their own** as a new paper contribution:

- running I-JEPA on an agricultural dataset;
- using a ViT with JEPA;
- introducing temporal inputs to a crop classifier;
- saying that the model uses “phenology” without a precise operational definition and comparison;
- stacking several established modules together and reporting only one aggregate metric;
- reporting an improvement against a weak baseline while ignoring relevant published methods.

The contribution must be narrower and empirically justified.

---

## 5. Dataset decision and data policy

### 5.1 Dataset comparison

| Dataset | What it contains | Strengths for this research | Limitations / cautions | Recommended use |
|---|---|---|---|---|
| **PASTIS** | 2,433 multitemporal Sentinel-2 tiles; 128 × 128 pixels; 10 spectral bands; variable sequences reported as 38–61 acquisitions; 18 crop classes; pixel-level semantics and parcel-instance labels; official five-fold split | Provides spatial structure, temporal variation and dense labels; supports crop-type segmentation and boundary-sensitive analysis | Data volume and preprocessing can be non-trivial; class imbalance and scene/region generalization matter; dense segmentation is more compute-intensive than parcel-level classification | **Main benchmark** if the goal includes spatial crop mapping and spatial/temporal JEPA |
| **PASTIS-R** | PASTIS extended with aligned Sentinel-1 radar observations as well as Sentinel-2 optical observations | Adds multimodal sensor information and supports radar–optical comparisons | Increases input complexity and compute; do not add radar until the optical-only experiment is sound | Second phase for a justified multimodal extension |
| **EuroCropsML** | 706,683 labeled parcel-level Sentinel-2 data points, 176 crop classes, annual 2021 time series, aggregated across Estonia, Latvia and Portugal | Convenient for fast temporal classification and few-shot/transfer studies; compact per-parcel temporal inputs can reduce implementation complexity | It is not a dense image/field-boundary benchmark; class distributions differ across countries; parcel samples are not equivalent to PASTIS tiles | **Fast temporal prototype**, or primary dataset if the final research question is parcel-level time-series classification |
| **SICKLE** | 2,370 season-wise samples from 388 plots, 21 crop types in four districts in the Cauvery Delta; includes Sentinel-1, Sentinel-2 and Landsat-8 data and annotated crop-season parameters for a subset | Indian agricultural setting, crop phenology information and multi-sensor observations | Modest plot count; the unit of analysis and train/test grouping must be handled carefully; annotations/tasks differ from PASTIS | External Indian-agriculture check once the primary method works |

Authoritative starting points:

- [PASTIS official benchmark repository](https://github.com/VSainteuf/pastis-benchmark)
- [AnySat official repository](https://github.com/gastruc/AnySat)
- [EuroCropsML documentation](https://eurocropsml.readthedocs.io/en/stable/index.html)
- [SICKLE paper DOI](https://doi.org/10.1109/WACV57701.2024.00589)

### 5.2 Recommended decision

For research depth, use **PASTIS as the eventual main benchmark** and keep the first experiments optical-only. For quick iteration, optionally prototype the temporal target-selection idea on a small EuroCropsML subset first. If the prototype is promising, port it to the PASTIS setting and test whether it helps dense crop mapping.

This two-stage plan is useful because it separates two questions:

1. Does the temporal JEPA objective have a measurable benefit at all?
2. Does the proposed idea also work when spatial context, mixed pixels and field boundaries matter?

If the final paper only evaluates EuroCropsML, then its claims should remain about parcel-level time-series classification and should not imply spatial segmentation improvements.

### 5.3 How much data should be used?

Use a progressive schedule. The percentages below are **engineering recommendations for experimentation**, not requirements of the official datasets.

| Stage | Data amount | Model/input setting | Goal |
|---|---|---|---|
| Data sanity check | 20–50 PASTIS tiles or a few hundred temporal samples | 4–8 dates, low resolution | Verify loading, labels, shapes, loss and one backward pass |
| Pilot | About 10% of the official training fold; roughly 200 tiles is a reasonable order-of-magnitude starting point for PASTIS | 8–12 selected dates, compact model, 5–10 epochs | Ensure the loss decreases and the model can be trained/resumed |
| Early baseline | 25–50% of training fold | 12–16 dates, same small architecture | Obtain initial supervised/JEPA comparisons and identify bottlenecks |
| Main pretraining | All imagery permitted by the declared training protocol | Start at 12–24 selected dates; retain full date metadata | Test the main hypothesis with a credible amount of unlabeled data |
| Final evaluation | Complete held-out fold(s), no ad-hoc shrinking of the final test set | Fixed protocol and metrics | Estimate generalization honestly |

For EuroCropsML, start with a modest subset (for example, 20,000–50,000 records or a class-aware subset for debugging), then expand after the pipeline works. Do not use a deliberately tiny or cherry-picked class subset for headline results.

### 5.4 Unlabeled pretraining versus labeled fine-tuning

The two data budgets must be tracked separately:

- **Self-supervised pretraining budget:** number of permitted training-fold sequences/tiles available without using crop labels in the loss.
- **Labeled downstream budget:** number or fraction of training labels exposed to a classifier/segmentation head or fine-tuning procedure.

To test label efficiency, pretrain according to a declared protocol, then train/evaluate the downstream model with label fractions such as 1%, 5%, 10%, 25% and 100%. The labels in validation/test data must not be used for model selection or training.

### 5.5 Split and leakage rules

1. Use the official PASTIS five-fold metadata and report which folds are used.
2. For the strict initial protocol, do not use held-out test-fold imagery during SSL pretraining. This avoids ambiguity about transductive access.
3. If unlabeled test-region imagery is later permitted, label that experiment as **transductive** and report it separately.
4. Use training-fold statistics for normalization; do not estimate normalization on held-out data unless the benchmark protocol explicitly requires it.
5. For parcel datasets, keep the same parcel or plot from leaking across training and test. Use group-aware splits; geographic holdout should be considered if class coverage allows.
6. Preserve acquisition dates and missing-observation masks. Do not silently interpolate or drop dates in a way that leaks future information into an earlier-time input.
7. Record dataset version, download URL, licence/terms, preprocessing code and checksums. Verify current archive size before downloading; package sizes may vary by release and mirror.

---

## 6. Candidate research hypothesis

### 6.1 Candidate: phenology-aware JEPA target selection

**Question:** Can target observations selected using unsupervised temporal changes in the vegetation signal produce more discriminative crop representations than standard random/regular target selection, particularly when few crop labels are available or observations are missing?

Possible outline:

```text
Multitemporal Sentinel-2 sequence + acquisition dates
                        |
                        v
           derive input-only seasonal-change cues
              (e.g. vegetation-index trajectory)
                        |
                        v
           select context and target observations
                        |
              +---------+----------+
              |                    |
              v                    v
        context encoder       EMA target encoder
              |                    |
              v                    |
           predictor               |
              |                    |
              +---- latent loss ---+
                        |
                        v
             pretrained representation
                        |
                        v
       crop classifier / segmentation head
```

The precise method still needs to be defined. Potential approaches include selecting targets near large trajectory changes, or selecting observations that cover distinct seasonal phases. Any vegetation index must be calculated from the correctly mapped bands after checking the dataset's band order and preprocessing.

### 6.2 Why it is worth a pilot

- Crop types often differ in their seasonal trajectories, not only in an isolated observation.
- Target selection is central to I-JEPA; the original work establishes that mask/target design can materially affect representation learning.
- The hypothesis can be tested with a small, matched experiment before building a more complicated network.
- It can be framed around a measurable setting: fewer labels and/or missing observations, rather than only a small improvement in one aggregate metric.

### 6.3 Novelty threats and required comparisons

This is **not yet a novelty claim**. At minimum, investigate:

- T-JEPA's time-gap-conditioned latent transitions for irregular EO acquisitions;
- CropSTS's decoupled temporal/spatial attention and hybrid JEPA/distillation setup;
- AnySat's JEPA-based heterogeneous EO foundation model and crop-type results;
- DMT-JEPA and other discriminative/local target-construction methods;
- relevant phenology-aware crop mapping and temporal self-supervised learning papers even when they do not use the name JEPA.

A target policy based on seasonal change may be a small adaptation unless it yields a clear, principled distinction and a measurable benefit under fair comparisons. An explicit related-work table should contrast **input, target definition, predictor conditioning, loss, data, downstream task and compute**.

### 6.4 Keep the experiment falsifiable

Use the same encoder, predictor size, training data, seed schedule and approximate optimizer budget for:

- standard/random target selection;
- a simple date-stratified or regular target schedule;
- the proposed phenology-aware target selection;
- a relevant published baseline or pretrained EO model where feasible.

If the proposed strategy does not improve downstream metrics, label efficiency or robustness consistently, do not keep it merely because it sounds biologically plausible. Revise the hypothesis based on the error analysis.

---

## 7. Experiment plan: when results reveal novelty

A useful experiment must answer a research question. It is not enough to add models to a notebook.

| Experiment | What to measure | What a result could mean | Decision after experiment |
|---|---|---|---|
| E0: data and training sanity | Input shapes, label maps, loss finiteness, gradient norms, overfit-a-tiny-batch test | Confirms code and data pipeline are not broken | Fix implementation problems before research comparisons |
| E1: supervised compact baseline | mIoU, macro-F1, per-class F1, accuracy, runtime | Establishes what a small labeled model can do | Keep as a fair baseline throughout |
| E2: frozen AnySat features, if hardware/code permit | Same downstream metrics under same label budget | Tests whether an available EO foundation representation already solves the task well | Helps determine whether training a new JEPA is justified |
| E3: compact I-JEPA baseline | SSL loss, collapse diagnostics, downstream linear probe/fine-tuning | Measures whether the JEPA-style learning setup is useful on this data | If it fails, debug/modify the pretraining setup before claiming a new method |
| E4: proposed target policy | Downstream metrics, training steps, FLOPs, memory and elapsed time | A consistent gain could support the target-policy hypothesis | Keep only if comparisons are controlled and reproducible |
| E5: label-efficiency curve | Performance at 1%, 5%, 10%, 25%, 100% labeled data | Identifies whether the method helps under scarce labels | This may become the central contribution if gains are strongest here |
| E6: missing-date robustness | Performance after controlled date removal, with the same held-out examples | Tests robustness to sparse/incomplete observations | Support only robustness levels actually tested |
| E7: ablations | Remove/replace one component at a time | Determines whether each claimed component contributes | Remove components that do not justify their cost |
| E8: repeated runs and broader folds | Mean, standard deviation and fold-level results | Indicates whether improvements exceed random variation or split effects | Only then consider stronger claims and paper submission |
| E9: external/generalization check | SICKLE or another compatible dataset, with aligned task definition | Tests whether the approach transfers outside the original setting | Report data/task differences; never imply direct comparability if labels differ |

### 7.1 What would be a meaningful finding?

Examples of potentially strong findings are:

- A consistent improvement over relevant baselines at low label fractions, with little or no loss at full supervision.
- Similar or better performance with materially fewer input dates, lower FLOPs, lower GPU memory or shorter training time.
- Better performance when dates are missing, demonstrated through a predeclared temporal-dropout protocol.
- A well-supported explanation of when the new target-selection policy helps and when it does not, supported by per-class analysis and ablations.

A small gain on one seed or one random subset is not compelling by itself. A method can still be valuable without increasing the highest full-data accuracy if it has a meaningful, well-supported efficiency or label-efficiency advantage.

---

## 8. Baselines, metrics and publication evidence

### 8.1 Baselines to prioritize

Fit them to the same task and data split, not merely to what is easiest to run.

1. **Compact supervised baseline:** a small temporal model or U-Net/ViT-style segmentation baseline appropriate to the chosen task.
2. **Existing pretrained Earth-observation features:** AnySat frozen features with a simple downstream head, if the official checkpoint, code and PASTIS variant are compatible.
3. **Compact JEPA baseline:** a small I-JEPA-style implementation. Describe the model scale and any departures from the official implementation.
4. **Relevant reconstruction/SSL baseline:** MAE or another appropriate temporal SSL method if implementation and compute budget allow.
5. **Recent related methods:** compare directly to published numbers only with extreme care. Ideally reproduce a relevant baseline; otherwise clearly label published results and document dataset/split/task differences. T-JEPA is very recent, so verify public code/checkpoint availability before assuming a fair direct reproduction is possible.

### 8.2 Metrics

For PASTIS dense segmentation/crop mapping:

- mean Intersection over Union (mIoU);
- macro-F1 and class-wise F1;
- overall accuracy;
- class-wise precision/recall and confusion matrix;
- optionally boundary-sensitive metrics if boundary quality is a stated claim and the metric is implemented correctly.

For EuroCropsML parcel-level classification:

- macro-F1;
- balanced accuracy;
- per-class precision/recall/F1;
- overall accuracy as a secondary metric;
- class-wise error analysis and region/country transfer where feasible.

For self-supervised representations and efficiency:

- frozen linear probe and/or lightweight fine-tuning performance;
- representation-collapse diagnostics (e.g. feature variance and effective rank, appropriate to the loss);
- number of parameters;
- FLOPs or a documented computational proxy;
- peak GPU memory;
- training time and inference latency;
- performance against number of labeled examples and number of observed dates.

Always report the metric that matches the claim. A classification accuracy result does not establish better segmentation boundaries; a faster inference result does not establish faster pretraining.

### 8.3 Recommended evidence standard

For a serious paper draft, aim for:

- a clear comparison with the best relevant baselines available within the compute budget;
- multiple independent random seeds (three is a reasonable minimum target for key compact experiments, more is better if resources allow);
- multiple official PASTIS folds in final evaluation, or a strong explanation if compute limits require fewer folds;
- label-efficiency tests if low-label learning is a claimed contribution;
- missing-observation tests if robustness is claimed;
- ablations that isolate the proposed mechanism;
- confidence intervals or other appropriately described uncertainty estimates when feasible;
- complete training and evaluation configurations;
- limitations, failure cases and disclosure of any use of external pretrained data.

There is no universal “required percentage improvement” or accuracy score for publication. The result must be novel relative to the literature, technically sound and useful for its stated purpose.

---

## 9. Hardware and runtime planning

### 9.1 Local PC specifications supplied for this project

| Component | Specification |
|---|---|
| Discrete GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Dedicated VRAM | 4 GB |
| Integrated GPU | Intel(R) UHD Graphics, reported 128 MB (typically backed partly by shared system memory) |
| System RAM | 16 GB |
| CPU | Exact model not supplied; record it in the first notebook cell |
| OS / storage | Not supplied; record OS, free disk space and storage type before downloading the dataset |

The RTX 3050's **4 GB VRAM is the key constraint**. The local machine is useful for development and small runs but will likely need small batch sizes, mixed precision where stable, smaller spatial crops, fewer time points and gradient accumulation. Integrated graphics memory is not a substitute for dedicated CUDA VRAM for typical PyTorch training.

### 9.2 Candidate platforms

| Platform | Intended role | Strengths | Limits / operational precautions |
|---|---|---|---|
| Google Colab free T4, when available | Main pilot training and compact model experiments | Generally more GPU memory than the local 4 GB device; straightforward notebook workflow | GPU access, available accelerator, session length and usage limits are dynamic; check the actual assigned GPU in every run |
| Local RTX 3050 Laptop GPU | Debugging, visualization, small baselines, profiling, lightweight training | Always available; easy to edit code and inspect data | 4 GB VRAM; 16 GB system RAM; long runs and larger temporal/spatial inputs may be impractical |
| Kaggle free GPU, when available | Alternative training runs and resumable experiments | Notebook environment and persistent output/artifact workflow can be convenient | GPU type, quota and session restrictions can change; inspect the current quota and GPU model rather than assuming a fixed allotment |

### 9.3 Illustrative wall-clock estimates

These are **rough planning ranges, not measured benchmark results**. They assume a compact custom encoder, Sentinel-2 only, mixed precision where supported, a small number of selected dates, and implementation-level optimization. They do not include potentially lengthy downloads, environment debugging or large-scale paper-quality evaluations. Exact times depend strongly on the model, patch size, batch size, data-loader speed and training steps.

| Experiment | Colab T4 | RTX 3050 Laptop, 4 GB | Kaggle free GPU |
|---|---:|---:|---:|
| Environment and initial data inspection | 0.5–2 hours, excluding slow downloads | 0.5–2 hours | 0.5–2 hours |
| Small pilot: roughly 10–20% of training tiles, 5–10 epochs | 1–3 hours | 2–6 hours | 1–4 hours |
| One larger compact-JEPA run, around 20 epochs | 4–12 hours | 10–30 hours | 4–12 hours |
| Multiple baselines, ablations and repeated seeds | 30–100+ aggregate GPU hours | Often 80–200+ hours; not recommended as the only training device | 30–100+ aggregate GPU hours, subject to quota/session limits |

These figures should not be treated as promises. A larger architecture or a slow input pipeline may take much longer; a very small model may take less time. The original I-JEPA ViT-H/14 ImageNet experiment was trained using 16 A100 GPUs, so reproducing that scale on these resources is not realistic. A smaller, purpose-built JEPA baseline is a different experiment and must be described as such.

### 9.4 How to measure the real runtime early

After the first few warm-up iterations, time at least 100 training steps. Record:

- average seconds per step;
- samples/tiles per batch and steps per epoch;
- time spent loading data versus GPU computation;
- peak allocated GPU memory;
- GPU name and VRAM;
- precision, batch size, spatial resolution and number of dates.

Estimate:

`training_seconds ≈ mean_step_seconds × steps_per_epoch × number_of_epochs`

Then separately add validation, checkpoints, data preprocessing and file transfer time. Recompute the estimate after changing the input resolution, number of dates, model or batch size.

### 9.5 Memory-saving defaults for the first implementation

- Start with a compact encoder rather than ViT-H.
- Start with batch size 1–2 locally; tune upward only after a memory check.
- Use mixed precision only if supported and stable on the allocated GPU.
- Use gradient accumulation when an effective larger batch is needed.
- Start with 64 × 64 or similarly small spatial crops during debugging; restore the benchmark resolution for main comparisons where the method permits it.
- Select a modest number of dates for the pilot, while preserving actual acquisition-time metadata.
- Cache preprocessed samples to disk rather than loading the entire dataset into system RAM or VRAM.
- Use AMP, gradient clipping and checkpointing only where appropriate; report any training change that materially affects the baseline.
- Save checkpoints by epoch/step and implement resume logic from day one.

---

## 10. Master Jupyter notebook design

Recommended file: `jepa_crop_research.ipynb`.

The notebook should orchestrate the full research pipeline but should keep core components small and testable. For longer runs, place reusable model/data code in `.py` modules and import them into the notebook. This makes experiments easier to test, reuse and resume than placing hundreds of lines of training logic in notebook cells.

### Part 0 — Configuration and environment audit

- Print Python, PyTorch, CUDA and library versions.
- Print available GPU name, VRAM and free system/disk space.
- Set all random seeds and deterministic settings where feasible.
- Define paths, model configuration, number of dates, spatial resolution, batch size, learning rate, optimizer and epoch count.
- Save the configuration as JSON/YAML alongside each run.

### Part 1 — Dataset download and audit

- Download the chosen dataset from its official source and record the exact release/version.
- Validate file integrity and licence/terms.
- Print sample counts, tensor shapes, date counts and band count.
- Visualize RGB/composite images, labels and selected temporal curves.
- Inspect missing observations, data ranges, NaNs and class imbalance.
- Create a data manifest and check hashes for processed caches where practical.

### Part 2 — Split and preprocessing

- Load official train/validation/test fold metadata.
- Implement group-aware splits for parcel-level data where needed.
- Map bands to documented names; never assume the band order.
- Normalize from training data only.
- Preserve temporal order, elapsed times and missing-data masks.
- Implement reproducible date selection and augmentation.
- Add an explicit test proving that held-out IDs do not appear in training/pretraining manifests under the strict protocol.

### Part 3 — Tiny-batch correctness tests

Before a long run:

- Make the model overfit a tiny labeled batch as a sanity check for the downstream head.
- Run one complete SSL step and verify that the loss is finite and gradients flow through the intended components.
- Verify the target/teacher update rule.
- Check the target encoder's gradients are disabled when using EMA teacher training.
- Check output dimensions and target positions match.
- Log embedding variance, feature norms and relevant collapse diagnostics.

### Part 4 — Supervised baseline

- Train a small task-appropriate baseline.
- Save metrics and predictions.
- Establish data-loader throughput and memory use.
- Generate a confusion matrix and per-class results.

### Part 5 — JEPA baseline

Implement or adapt an I-JEPA-style objective with:

- context encoder;
- target encoder/teacher if following I-JEPA;
- predictor;
- context/target mask generator;
- positional or temporal conditioning appropriate to the task;
- latent prediction loss;
- EMA teacher update;
- collapse and optimization diagnostics;
- downstream evaluation using frozen features/linear probe and/or fine-tuning.

Maintain clear separation between the faithful baseline and any proposed modifications.

### Part 6 — Existing EO representation baseline

Where compatible, extract pretrained AnySat features and fit a simple downstream head. Document whether the features/checkpoint are frozen, whether any fine-tuning occurs, and which data the released checkpoint was pretrained on. Do not claim a perfectly controlled comparison if training data or capacity differ materially.

### Part 7 — Candidate modification

- Implement the phenology-aware target-selection policy as a separate configurable component.
- Keep a baseline switch to use ordinary targets with all other settings unchanged.
- Document exactly how targets are selected and how the method handles irregular intervals or missing dates.
- Do not add several other architectural components in the same experiment.

### Part 8 — Experiment runner

Every run should have a unique run ID and record:

- model version and configuration;
- dataset version and split;
- random seed;
- label fraction and pretraining sample count;
- number of temporal observations;
- epochs/steps;
- parameter count, FLOPs where measurable, peak VRAM and duration;
- checkpoint and metrics file paths.

Write per-epoch logs to CSV/JSON and save the best checkpoint based only on validation criteria declared in advance.

### Part 9 — Evaluation and ablations

Run label fractions, missing-date robustness, ablations and repeated seeds. Freeze the evaluation configuration before final testing. Create tables and plots directly from the saved metrics rather than manually entering result values.

### Part 10 — Research notebook outputs

The notebook should produce:

- data audit report;
- reproducible configurations;
- training/loss curves;
- per-class metrics and confusion matrix;
- label-efficiency plots;
- missing-observation robustness plots;
- ablation tables;
- model size, GPU memory and runtime table;
- CSV/JSON metrics;
- checkpoint inventory;
- short experiment decision log stating what each run taught us and which run should come next.

---

## 11. Recommended folder and artifact layout

```text
jepa-crop-research/
├── README.md
├── requirements.txt              # or environment.yml / pyproject.toml
├── configs/
│   ├── supervised_baseline.yaml
│   ├── ijepa_baseline.yaml
│   ├── phenology_targets.yaml
│   └── evaluation.yaml
├── notebooks/
│   └── jepa_crop_research.ipynb
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── masks.py
│   ├── objectives.py
│   ├── train_ssl.py
│   ├── train_downstream.py
│   ├── evaluate.py
│   └── utils.py
├── manifests/
│   ├── data_manifest.csv
│   └── fold_ids.json
├── outputs/
│   └── <run_id>/
│       ├── config.yaml
│       ├── train_log.csv
│       ├── metrics.json
│       ├── predictions/
│       ├── checkpoints/
│       └── notes.md
├── figures/
├── tables/
└── paper/
    ├── literature_matrix.csv
    ├── experiment_log.md
    └── draft.md
```

Keep raw data out of version control. Keep the exact commands/configurations needed to recreate results in the repository. Do not silently overwrite the only best checkpoint.

---

## 12. Research best practices and common failure modes

### Experimental fairness

- Match data, folds, label budget, preprocessing, downstream head, seed schedule and training budget as closely as practical.
- Separate reproduced baseline scores from published scores taken from papers.
- State any mismatch in the input modality, image resolution, crop classes, segmentation protocol or pretrained data.
- If using a public pretrained checkpoint, disclose its training data and potential overlap with the benchmark.

### Leakage and evaluation

- Do not use test labels for normalization, model selection, threshold setting or target-policy design.
- For strict inductive evaluation, keep held-out test imagery out of pretraining.
- Use spatial/parcel grouping to reduce near-duplicate leakage.
- Define the missing-date protocol once, then apply it consistently to each model.

### Statistical reliability

- Use multiple seeds for key runs and report mean and variation.
- Avoid stopping one model early while giving another model a larger optimization budget without disclosure.
- Save all run results, not just the best seed.
- Predefine the primary metric and secondary metrics.

### Interpretation

- Lower SSL loss does not necessarily mean a better representation.
- A single overall accuracy can hide failures on minority crop classes.
- Improvements on random test tiles do not automatically demonstrate geographic transfer.
- Parameter count is not the same as inference efficiency, and FLOPs do not capture input-pipeline bottlenecks; report measured latency/memory as well.
- A method that works only on one small pilot split is a lead to investigate, not a publishable general claim.

### Efficient research workflow

Use **decision gates**:

1. If data sanity checks fail, stop and repair the dataset loader.
2. If a tiny-batch test fails, stop before training for hours.
3. If the supervised baseline is implausibly weak, validate the labels, metrics and preprocessing.
4. If JEPA collapses or the downstream result is no better than a random representation, audit target/teacher logic before adding novel modules.
5. If the proposed modification has no measurable gain, analyze where it fails and revise or abandon it.
6. Only scale runs that answer a defined research question.

---

## 13. Step-by-step roadmap

### Stage 1 — Learn and specify the baseline

**Tasks**
- Read I-JEPA and inspect the official repository.
- Draw the context/target/predictor flow and explain EMA, masking, positional conditioning and loss.
- Define the exact downstream task: PASTIS dense crop-type segmentation or EuroCropsML parcel-level classification.
- Record implementation differences from original I-JEPA.

**Exit criteria**
- A one-page architecture note;
- a minimal forward-pass diagram;
- a tested explanation of tensor shapes and target updates;
- a selected dataset/task statement.

### Stage 2 — Dataset audit and fast pilot

**Tasks**
- Download/audit a small subset;
- establish official/group-aware splits;
- implement visualizations and input validation;
- benchmark a compact supervised baseline;
- measure 100-step throughput and GPU memory.

**Exit criteria**
- no leakage across folds;
- a finite training run;
- saved evaluation metrics;
- a reliable runtime estimate for the actual implementation.

### Stage 3 — Compact JEPA baseline

**Tasks**
- Implement/adapt context encoder, target encoder, predictor and target masking;
- verify teacher updates and collapse diagnostics;
- pretrain on a small, fixed budget;
- evaluate frozen features/linear probe and a fine-tuned downstream head.

**Exit criteria**
- reproducible baseline checkpoint;
- downstream metrics;
- loss/embedding diagnostics;
- documented configuration and runtime.

### Stage 4 — Baseline comparison and diagnosis

**Tasks**
- Compare supervised baseline, compact JEPA, and AnySat features when feasible;
- inspect per-class confusions;
- test reduced label fractions;
- test a small, controlled missing-date protocol;
- summarize the clearest weakness.

**Exit criteria**
- a written diagnosis with evidence;
- a shortlist of one or two specific hypotheses;
- a literature matrix addressing the closest related work.

### Stage 5 — One architectural/objective modification

**Tasks**
- define a precise mathematical/algorithmic change;
- explain why it targets the observed failure;
- implement a switchable baseline/new-method comparison;
- keep the training data and optimization budget matched;
- run the smallest informative test first.

**Exit criteria**
- result is reproducible;
- benefit or failure is understood;
- related-work comparison supports a potential gap;
- no unsupported novelty claim.

### Stage 6 — Ablation and label-efficiency study

**Tasks**
- compare multiple label fractions;
- ablate the new component;
- report per-class results and compute cost;
- repeat key runs across seeds.

**Exit criteria**
- evidence shows whether the modification itself contributes;
- statistical variation is reported;
- efficiency/accuracy trade-offs are clear.

### Stage 7 — Broader evaluation

**Tasks**
- evaluate multiple official PASTIS folds;
- test the method under missing observations if this is a claimed capability;
- consider a compatible external dataset such as SICKLE;
- align classes, task definitions and input modalities carefully.

**Exit criteria**
- findings are not tied solely to one random subset;
- limitations and transfer scope are explicitly stated.

### Stage 8 — Paper preparation

**Paper structure**
1. Abstract and problem statement;
2. related work on JEPA, EO foundation models and crop time series;
3. problem formulation and data protocol;
4. architecture and objective;
5. experimental setup, baselines and evaluation metrics;
6. main results;
7. label-efficiency/robustness results;
8. ablations and compute analysis;
9. error analysis, limitations and ethics/data considerations;
10. conclusion and reproducibility statement.

**Before submission**
- repeat the literature search, especially work released after October 2026;
- verify every citation, publication status and reported metric against the source paper;
- make clear what is novel relative to AnySat, CropSTS, T-JEPA, DMT-JEPA and related crop time-series methods;
- release code/configs and describe data access/licensing constraints;
- avoid claiming superiority unless the comparison is genuinely compatible.

---

## 14. Literature matrix template

Use this matrix as a living document. One row per paper is recommended.

| Field | What to record |
|---|---|
| Citation / year / venue | Exact paper metadata and publication status |
| Modality | RGB, multispectral optical, SAR, video, multimodal, time series |
| Input unit | Image, patch, pixel sequence, parcel sequence, multiple sensors |
| Context definition | What the online/context path sees |
| Target definition | Which target observations/patches are predicted and how they are selected |
| Encoder and predictor | Architecture, parameter count, temporal/spatial structure |
| Objective | Prediction loss, regularizers, teacher/EMA, auxiliary losses |
| Pretraining data | Dataset, number of samples, labels used or not used, overlap risk |
| Downstream task | Classification, segmentation, retrieval, forecasting, etc. |
| Dataset/split | Official split, spatial grouping, label fractions |
| Metrics and headline results | Record exact metrics and cite table/figure/page |
| Compute | GPU, duration, model size, batch size, input size |
| Limitations | What the authors state and what the experiment does not establish |
| Relevance to this project | Exact overlap with the proposed method |
| Possible gap | Only after reading method and supplemental material |

Do not write “no prior work exists” based on search-engine snippets. A defensible novelty statement should name the closest existing methods and explain the precise technical and experimental difference.

---

## 15. First-session checklist

Complete these tasks in order before launching a long training run:

- [ ] Create a clean repository and `jepa_crop_research.ipynb`.
- [ ] Print the local CPU model, operating system, storage availability, Python version, PyTorch version, CUDA status, GPU name and VRAM.
- [ ] Choose a first task: recommended final target is PASTIS dense crop classification/segmentation; optional quick temporal prototype is EuroCropsML.
- [ ] Download only what is needed for a data audit; verify the current dataset version and official splits.
- [ ] Visualize one sample, its time series, label map and per-class distribution.
- [ ] Run a tiny supervised baseline and overfit a tiny batch as a correctness test.
- [ ] Benchmark 100 training steps on the actual assigned GPU and update runtime estimates.
- [ ] Run a compact JEPA sanity test and check target/EMA updates and representation diagnostics.
- [ ] Save the first experiment configuration, metrics and checkpoint.
- [ ] Only then decide whether to scale the dataset or begin the proposed modification.

---

## 16. References and official resources

### Core JEPA

1. LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. [Project page](https://jepa.si5.pl/methods/path-autonomous-mi.html)
2. *Joint Embedding Predictive Architectures Focus on Slow Features* (2022). [arXiv](https://arxiv.org/abs/2211.10831)
3. Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR 2023. [arXiv](https://arxiv.org/abs/2301.08243) · [Official repository](https://github.com/facebookresearch/ijepa)
4. Bardes, Ponce & LeCun (2023). *MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features*. [arXiv](https://arxiv.org/abs/2307.12698)
5. Fei, Fan & Huang (2023). *A-JEPA: Joint-Embedding Predictive Architecture Can Listen*. [arXiv](https://arxiv.org/abs/2311.15830)
6. *V-JEPA* (2024). [arXiv](https://arxiv.org/abs/2405.09987)
7. Littwin et al. (2024). *How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks*. NeurIPS 2024. [arXiv](https://arxiv.org/abs/2407.03475)
8. Mo & Yun (2024). *DMT-JEPA: Discriminative Masked Targets for Joint-Embedding Predictive Architecture*. [arXiv](https://arxiv.org/abs/2405.17995) · [Code](https://github.com/DMTJEPA/DMTJEPA)
9. Mo & Tong (2024). *Connecting JEPA with Contrastive Self-Supervised Learning*. [arXiv](https://arxiv.org/abs/2410.19560)
10. *Enhancing JEPAs with Spatial Conditioning* (2024). [arXiv](https://arxiv.org/abs/2410.10773)
11. Velarde & Parra (2024). *Recurrent Joint Embedding Predictive Architecture with Recurrent Forward Propagation Learning*. [arXiv](https://arxiv.org/abs/2411.16695)
12. Assran et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. [arXiv](https://arxiv.org/abs/2506.09985)
13. Balestriero & LeCun (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*. [arXiv](https://arxiv.org/abs/2511.08544) · [Code](https://github.com/galilai-group/lejepa)
14. Perera et al. (2025). *CrossJEPA: Cross-Modal Joint-Embedding Predictive Architecture for Efficient 3D Representation Learning from 2D Images*. [arXiv](https://arxiv.org/abs/2511.18424)

### Earth observation and crops

15. Li et al. (2024). *Predicting Gradient is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture*. ISPRS Journal of Photogrammetry and Remote Sensing. [DOI](https://doi.org/10.1016/j.isprsjprs.2024.09.013) · [arXiv](https://arxiv.org/abs/2311.15153) · [Code](https://github.com/waterdisappear/SAR-JEPA)
16. Astruc et al. (2025). *AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities*. CVPR 2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html) · [Code](https://github.com/gastruc/AnySat)
17. Yan, Gu & Chen (2025). *CropSTS: A Remote Sensing Foundation Model for Cropland Classification with Decoupled Spatiotemporal Attention*. Remote Sensing, 17(14), 2481. [DOI](https://doi.org/10.3390/rs17142481)
18. Peng et al. (2026). *T-JEPA: A Temporal Joint-Embedding Predictive Architecture for Learning Better Remote Sensing Representations*. arXiv preprint submitted 5 October 2026. [arXiv](https://arxiv.org/abs/2610.05731)
19. Junghare & Banerjee (2026). *UA-JEPA: Uncertainty-aware joint-embedding predictive learning for remote sensing image retrieval*. Pattern Recognition Letters. [DOI](https://doi.org/10.1016/j.patrec.2026.08.007)

### Datasets

20. PASTIS benchmark and official splits. [Repository](https://github.com/VSainteuf/pastis-benchmark)
21. AnySat code/checkpoints and supported datasets. [Repository](https://github.com/gastruc/AnySat)
22. EuroCropsML documentation and data description. [Documentation](https://eurocropsml.readthedocs.io/en/stable/index.html)
23. SICKLE multi-sensor crop dataset. [DOI](https://doi.org/10.1109/WACV57701.2024.00589)

**Bibliography note:** The references above are a starting set, not a final paper bibliography. Before writing the related-work section, verify each paper's exact publication status, version, results and methods from the full paper/supplementary material, then search for additional papers published up to the actual submission date.

---

## 17. Final working principle

The core job is to move from an appealing architecture idea to evidence. Start small, record everything, diagnose a real limitation, change one thing, and test that change fairly. The strongest possible result is not necessarily the most elaborate network. It is a precise method that demonstrably improves crop-relevant representations or makes them cheaper/more robust, compared against the closest established approaches and supported by reproducible experiments.
