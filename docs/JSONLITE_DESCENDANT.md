## Native JSON descendant update

The current [OpenSML-JSONLite-175M-Instruct](https://huggingface.co/OpenSML/OpenSML-JSONLite-175M)
continues this pretrained base through grounded QA, requested-field and focused extraction,
a 15,000-presentation native JSON foundation, and an additional focused extraction SFT.
The added stage combines original JSON TRAIN replay, matched zero/one/two/three-value
examples, and training-only frozen-parent distillation. The main model generates the
entire JSON object itself from source text and a supplied schema, with no assembly or repair.

On the same previously observed 1,444-request native JSON evaluation, complete-request
correctness is **1,199/1,444 (83.03%)**, compared with 1,191/1,444 for the prior native release.
Field accuracy is 91.36%, value F1 90.40%, missing-field accuracy 92.79%, and schema validity
99.79%. The canonical subset is 1,163/1,388 (83.79%); observed playground correctness is
44/54. These are descendant results, not scores of this pretrained base. Some domains and
value precision/recall regress slightly, and strict automatic retention gates were not met.
See the [JSONLite technical report](https://huggingface.co/OpenSML/OpenSML-JSONLite-175M/blob/main/TECHNICAL_REPORT.md)
for full training provenance, matched comparison and limitations.

The base weights, tokenizer, inference runner and source bundle are unchanged by this documentation update.
