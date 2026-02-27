# Hyper Efficient Eye Cropping/Tracking

A high-performance eye tracking and cropping tool that uses **pure heuristics and mathematical filters** instead of machine learning models for production execution. 

Designed for maximum speed and efficiency in resource-constrained environments.

# Features:
- **No AI in Production**: Uses pure image processing (OpenCV + NumPy) for detection. Zero model latency, zero weights to download.
- **AI-Verified Quality**: Includes a benchmark suite that use `dlib` 68-point landmarks as "ground truth" to verify and tune the heuristic performance.
- **Advanced Math Filters**:
    - **Kalman Stabilization**: Uses position + velocity modeling to predict eye movement, eliminating jitter.
    - **CLAHE Preprocessing**: Adaptive contrast normalization for robust performance across skin tones and lighting.
    - **Spatial Grouping**: Weights candidates by center-proximity and eye-pair geometry.
    - **Radial Gradient Validation**: Filters dark spots by analyzing light-gradient transitions (dark center → bright sclera).
- **Extreme Area Reduction**: Real-time tuning enables ~80% reduction in screen area while maintaining >90% capture rate in tests
> [!NOTE]
> This preformance is not stable and dependant on circumstances, fast movement and bad positioning can muck it up

# New Version:
- A new, better version that can be used by running main.py

# Usage:

Production script:
lean production version
```bash
uv run eye_crop.py
```

Benchmarking and Tuning:
Run next to an AI ground-truth model to see capture rates and area reduction in real-time.
```bash
uv run benchmark.py
```
- **Arrow Keys**: Adjust crop tightness.
- **[ ]**: Adjust dark percentile sensitivity.
- **q**: Quit and print optimal config.
  
# Dependencies:

- Python 3.12+
- `opencv-python`
- `numpy`
- `dlib` (Benchmark tool only)
- `uv` (Recommended package manager)



