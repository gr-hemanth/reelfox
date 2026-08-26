# Instagram Content Analyzer

An experimental CLI prototype exploring whether a public Instagram URL can
eventually be turned into accurate, structured, multimodal information.

**This repository is at Phase 4 of that experiment. It does not analyze
content yet — extraction is proven viable but reliability has not been
benchmarked.**

## Purpose

The long-term feasibility question is whether this pipeline can be made to
produce trustworthy structured output:

```
Instagram URL
  -> URL validation
  -> Instagram content extraction
  -> Extraction validation
  -> Audio / speech understanding
  -> Vision understanding
  -> OCR / on-screen text
  -> Multimodal synthesis
  -> Structured JSON
  -> Cleanup
  -> Evaluation
```
