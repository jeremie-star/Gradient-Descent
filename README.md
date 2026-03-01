# Gradient-Descent Group-10 Formative 3

A collection of Jupyter notebooks exploring core machine learning and statistics concepts from scratch — Bayesian probability, gradient descent optimization, and probability distributions.

## Notebooks

### 1. Bayesian Probability — Sentiment Analysis

**File:** `Bayesian_Probablity.ipynb`
**Dataset:** IMDB Movie Reviews (50,000 reviews)

Applies Bayes' theorem to predict whether a movie review is positive or negative based on keyword presence.

**What it does:**
- Loads and preprocesses IMDB reviews (lowercasing, punctuation removal, stopword filtering)
- Identifies the top positive keywords (`great`, `best`, `love`) and negative keywords (`bad`, `worst`)
- Computes Bayesian probabilities for each keyword:
  - **Prior** — P(Positive) = 0.50
  - **Likelihood** — P(keyword | Positive)
  - **Marginal** — P(keyword)
  - **Posterior** — P(Positive | keyword)

**Key result:**

| Keyword | Posterior P(Positive\|keyword) |
|---------|-------------------------------|
| great   | 0.6741                        |
| best    | 0.6453                        |
| love    | 0.6349                        |
| bad     | 0.2512                        |
| worst   | 0.0927                        |

---

### 2. Gradient Descent — Linear Regression

**File:** `Gradient Descent.ipynb`

Implements gradient descent from scratch to fit a line (`y = mx + b`) to a small dataset.

**What it does:**
- Starts with initial parameters `m = -1`, `b = 1`
- Uses SciPy's `approx_fprime` to compute numerical gradients of the MSE cost function
- Updates `m` and `b` over 4 iterations with learning rate `α = 0.1`
- Visualizes parameter convergence and error reduction

**Key result:**

```
Iteration 1: m = 1.7000, b = 2.1000
Iteration 2: m = 1.2600, b = 1.9000
Iteration 3: m = 1.3400, b = 1.9160
Iteration 4: m = 1.3336, b = 1.8968
```

Plots included: `m` and `b` over iterations, MSE error curve.

---

### 3. Probability Distributions — Bivariate Normal

**File:** `Probability Distributions.ipynb`
**Dataset:** Old Faithful Geyser (272 observations)

Builds a bivariate normal PDF from scratch and visualizes the joint distribution of eruption duration vs. waiting time.

**What it does:**
- Fetches the Old Faithful dataset and computes sample statistics (mean, std dev, correlation)
- Implements the bivariate normal PDF formula manually using only `math` and standard Python
- Computes probability density for each data point
- Produces three visualizations:
  - **Scatter plot** — data points colored by PDF value
  - **2D contour plot** — topographical density map with data overlay
  - **3D surface plot** — the full bivariate normal bell shape

**Key statistics:**

| Parameter           | Value  |
|---------------------|--------|
| Duration (X) Mean   | 3.49   |
| Waiting (Y) Mean    | 70.90  |
| Correlation (ρ)     | 0.9008 |

---

## Tech Stack

- **Python 3**
- **pandas** — data loading and manipulation (Bayesian notebook)
- **matplotlib** — all visualizations
- **NumPy** — grid computation for contour/surface plots
- **SciPy** — numerical gradient approximation (Gradient Descent notebook)
- **Google Colab** — execution environment

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/jeremie-star/Gradient-Descent.git
   ```

2. Open any `.ipynb` file in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.

3. For the Bayesian notebook, upload the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) to `/content/sample_data/`.

4. Run all cells.

## Contributors

- [Iyamurinze](https://github.com/Iyamurinze)
- [Tapiwanashe6](https://github.com/Tapiwanashe6)
- [ayioka](https://github.com/ayioka)
- [Hasbiyallah](https://github.com/hasby-umutoniwabo)

## License

This project is for educational purposes.
