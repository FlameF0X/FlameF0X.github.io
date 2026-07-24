
> *Affiliation / Link*  
> *Date: YYYY-MM-DD*

---

## Abstract

One paragraph summarizing the work: the problem, the approach, the key result, and why it matters.

---

## 1. Introduction

Context and motivation. What gap does this work fill? What is the core idea?

---

## 2. Method / Architecture

Describe the design. Use math where helpful:

$$S_t = S_{t-1} \odot W + k_t \odot v_t$$

And tables for configs:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $d_\text{model}$ | 512 | Hidden dimension |
| $n_\text{layers}$ | 6 | Number of blocks |

ASCII-art diagrams inside fenced code blocks work well:

```
┌──────────┐    ┌──────────────┐
│  Input   │───▶│   Block × N  │───▶ Output
└──────────┘    └──────────────┘
```

---

## 3. Experiments

Training setup, datasets, hyperparameters.

### 3.1 Results

| Metric | Value |
|--------|-------|
| Validation loss | X.XXX |
| Perplexity | XX.XX |

### 3.2 Inference Speed

| Hardware | Throughput |
|----------|-----------|
| CPU (2 vCPU) | XX.X tok/s |
| GPU (T4) | XX.X tok/s |

---

## 4. Discussion

What worked, what didn't, limitations, future directions.

---

## 5. Conclusion

Summary of contributions and final remarks.

---

## References

1. Author, A., & Author, B. (Year). *Title.* Venue.

---

*© Year Author Name. License.*
