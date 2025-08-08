# BRCA Multi-omics GNN Notebook

## Overview

This Jupyter Notebook implements a graph neural network (GNN) pipeline for binary survival classification (alive vs. deceased) using multi-omics breast cancer data. All components,from data loading through visualization, training, evaluation, and interpretability, are structured as executable cells, ideal for interactive exploration.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Notebook Structure & Sections](#notebook-structure--sections)
4. [How to Run](#how-to-run)
5. [Key Notebook Cells](#key-notebook-cells)
6. [Customization Tips](#customization-tips)
7. [Saving Outputs](#saving-outputs)
8. [Troubleshooting & FAQs](#troubleshooting--faqs)
9. [Next Steps & Enhancements](#next-steps--enhancements)
10. [Contact & License](#contact--license)

---

## 1. Quick Start

1. Open the notebook (e.g., `BRCA_TCGA.ipynb`) in Jupyter (Notebook or Lab).
2. Ensure `data.csv` is placed in the same directory.
3. Execute cells **sequentially** from top to bottom.
4. Visualizations and progress will appear inline as you go.

---

## 2. Prerequisites

Make sure the following packages are installed (via `pip` or your preferred manager):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow networkx scipy
```

---

## 3. Notebook Structure & Sections

The notebook is organized into these modular sections:

* **Imports & Setup**: loads libraries, configures warning filters.
* **Data Loading & Preprocessing**: includes `load_and_preprocess_data()` cell with detailed printouts.
* **Adjacency (Graph) Construction**: `create_adjacency_matrix()` with optional threshold/KNN logic and memory-reducing feature selection.
* **Visualization**: inline plots for class distribution, feature types, adjacency heatmap, PCA, etc.
* **Model Definition**: code cell defining `GraphNeuralLayer` and `build_gnn_model()`, followed by `.summary()`.
* **Train & Evaluate**: cells to train the model with callbacks and to evaluate with accuracy, ROC, classification report.
* **Results Plots**: training curves, confusion matrix, ROC curves displayed inline.
* **Feature Importance**: gradient-based attribution visualization for top features.
* **Results Summary**: prints top-10 features with importance at the bottom.

---

## 4. How to Run

* Open the notebook in Jupyter and **run each cell** in order.
* For long-running cells (like training), you’ll see progress bars from TensorFlow inline.
* If interrupted, you can skip back to the last completed step (e.g., reload model or re-run training cell).

---

## 5. Key Notebook Cells (Highlights)

| Section                        | Description                                                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **Data Preprocessing**         | Handles missing values, label cleaning, upsampling, and prints distribution dynamics.                                     |
| **Adjacency Construction**     | Efficiently builds a graph structure from correlations or neighbor graphs; includes feature reduction.                    |
| **Model & Training**           | Defines custom GNN layer, composes the model, shows architecture summary, and fits with early stopping and checkpointing. |
| **Evaluation & Visualization** | Generates performance metrics and decision plots directly in notebook for easy analysis.                                  |
| **Feature Interpretation**     | Uses gradient-based attribution to highlight influential features, visualized as a bar chart.                             |

---

## 6. Customization Tips

* **Adjust graph method**: Toggle between correlation vs. k-NN, or change the threshold parameter inline.
* **Tweak model architecture**: Edit layer sizes, add/drop layers, or change dropout rates in the model definition cell.
* **Balance strategies**: Replace random upsampling with SMOTE by adding a cell (via `imbalanced-learn`).
* **Use sparse adjacency**: You can modify adjacency construction cells to use sparse matrices if needed for scalability.
* **Try interpretability alternatives**: Add a cell to incorporate **Integrated Gradients** (TensorFlow example) for more robust attributions.

---

## 7. Saving Outputs

* Visualizations show inline automatically.
* To **save figures**, add:

  ```python
  fig.savefig('outputs/figure_name.png', dpi=300)
  ```
* The model checkpoint (`bestmodel.h5`) is saved automatically in training cells.
* For later predictions, persist the `scaler` object using `pickle`.

---

## 8. Troubleshooting & FAQs

* **Out-of-memory crashes**: Reduce `max_features` in adjacency cell or lower the batch size.
* **Missing `data.csv` error**: Ensure the file is in the same folder or update path in the first cell.
* **Graphs or plots not appearing**: Confirm matplotlib inline mode is enabled (`%matplotlib inline`) at top.
* **Random seed consistency**: If you need reproducible training, add seed setting cells:

  ```python
  import numpy as np, tensorflow as tf, random
  np.random.seed(42)
  tf.random.set_seed(42)
  random.seed(42)
  ```

---

## 9. Next Steps & Enhancements

* **Convert to script**: Extract logic into `.py` modules with an entry-point like `run.py` or use `nbconvert` for scripting.
* **Advanced imbalanced handling**: Try SMOTE, class weighting, or focal loss alternatives.
* **Incorporate biological networks**: Use STRING, BioGRID, or curated gene-gene interaction priors to replace or augment the correlation graph.
* **Explore model variants**: Edge-conditioned GNNs, GAT (attention), or multi-layer/multi-omics heterogeneous graphs.

---

## 10. Contact & License

* **License**: GPL 3.
* **Questions / improvements**: Feel free to file a GitHub issue or PR.
