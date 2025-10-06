# Windows CPU-Only Offline AI App Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## Offline-first architecture (.NET + Python microservice)

```mermaid
flowchart TD
    N1["Step 1\nMapped field workflows and device limits; prioritised image capture, cropping, OCR"]
    N2["Step 2\nDeveloped Windows tablet app in .NET with Python micro-services; packaged custom C"]
    N1 --> N2
    N3["Step 3\nOptimised models via quantisation, operator fusion, minimal batching to meet targe"]
    N2 --> N3
    N4["Step 4\nImplemented preprocessing (crop, contrast, denoise) and local OCR; persisted struc"]
    N3 --> N4
    N5["Step 5\nBuilt resilient sync: detect connectivity, queue events, deduplicate, encrypt payl"]
    N4 --> N5
```

## ONNX quantization impact chart

```mermaid
flowchart LR
    N1["Inputs\nImages or camera frames entering the inference workflow"]
    N2["Decision Layer\nONNX quantization impact chart"]
    N1 --> N2
    N3["User Surface\nOperator-facing UI or dashboard surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nInference or response latency"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
