# Windows CPU-Only Offline AI Application

> **Domain:** Logistics

## Overview

Field teams needed reliable AI on standard Windows tablets where connectivity is patchy and specialist hardware is impractical. Cloud-only inference caused delays, missed scans, and rising data costs. GPU-class devices were expensive and hard to maintain at remote sites. Paper-heavy processes and inconsistent capture made downstream reporting slow and error-prone. Without a lightweight, offline option, operations risked slower depot turns, higher total cost of ownership, poor audit readiness. The objective involved running custom models locally on CPU, keeping work moving without internet, synchronising structured results securely when connectivity returned, improving accuracy, speed, and control across dispersed locations.

## Approach

- Mapped field workflows and device limits; prioritised image capture, cropping, OCR, offline storage, background synchronisation
- Developed Windows tablet app in .NET with Python micro-services; packaged custom CPU-tuned models for on-device inference
- Optimised models via quantisation, operator fusion, minimal batching to meet target accuracy/latency on mid-range CPUs
- Implemented preprocessing (crop, contrast, denoise) and local OCR; persisted structured results in SQLite for downstream analytics
- Built resilient sync: detect connectivity, queue events, deduplicate, encrypt payloads, push automatically when online
- Validated through test harnesses and field pilots; tracked throughput, accuracy, power draw; delivered in two-week increments

## Skills & Technologies

- .NET
- Python
- ONNX Runtime
- Tesseract OCR
- SQLite
- Image Processing
- Model Quantization
- Edge AI Deployment
- Background Sync Design
- Data Security
