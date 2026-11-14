# Research Direction

The task is unsupervised multi-class visual anomaly detection. On each of two benchmarks, one model is trained on normal images only. It must score every test image and produce a pixel heatmap:

- MVTec-AD: 15 categories.
- VisA: 12 categories of complex, multi-instance and circuit-board objects.

The reference compares each input with the normal images of its category in a frozen pretrained feature space. It then refines that comparison with a component learned from normal images plus synthetic anomalies.

Valid improvements make the evidence from that comparison more reliable at the same operating point: the same frozen feature extractor at 224 px, and normal data only. Examples:

- how the input-to-reference matching is represented and refined;
- which normal references are used and how;
- how learned and raw evidence are combined into maps and image scores;
- how the learned part is trained on synthetic anomalies (synthesis, objectives, schedules).

The measure of success is detecting and localising defect types the method has never seen. The ranked slices hold defect types absent from the visible ones. Image-level detection and pixel-level localisation are both scored, on both benchmarks.

Out of bounds (see `scoring/CONSTRAINTS.md`):

- training on real anomalies, including the visible development answers;
- any use of the hidden answers or of public MVTec-AD or VisA test annotations;
- a larger or fine-tuned feature extractor, or a higher resolution;
- scoring a test input with information from other test inputs;
- external anomaly datasets.
