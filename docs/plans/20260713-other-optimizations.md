The regression to the mean problem (where models under-predict high extremes and over-predict low extremes) is a classic issue in machine learning, particularly with tree-based models like XGBoost. Trees average the target values within each leaf node, which inherently pulls predictions toward the global mean and makes them incapable of extrapolating beyond the training data's range.

Since improving the feature set gave you a solid bump (0.5 → 0.64 $R^2$), your model is now better at distinguishing different states. To push it further and fix the remaining regression to the mean, here are several other techniques you can try:

### 1. Target Variable Transformation
If your target variable (soil moisture) is skewed or has heavy tails, the model will naturally focus on the bulk of the data and ignore the extremes.
*   **What to try:** Transform your target before training (e.g., Box-Cox, Yeo-Johnson, or a simple Power/Log transform depending on the skew), then apply the inverse transformation to your predictions.
*   **Why it works:** It normalizes the target distribution, giving the model's loss function a more balanced gradient across the entire range of values.

### 2. Tail-Heavy Sample Weighting
You are already experimenting with "Weighted Models" and Pseudo-Huber loss, but you can refine *how* you weight the samples. 
*   **What to try:** Instead of just spatial or temporal weighting, assign weights inversely proportional to the frequency of the target value (e.g., bin the target variable, and give samples in rare extreme bins higher weights). 
*   **Why it works:** It forces the model to pay higher penalties for getting the rare extreme peaks and valleys wrong, sacrificing some accuracy on the "average" days to get the extremes right.

### 3. Change the Loss Function (Objective)
Tree models default to optimizing MSE (mean squared error), which estimates the conditional mean.
*   **Quantile Regression:** Change the objective to `reg:quantileerror` in XGBoost. Instead of predicting the mean, you can predict the median (50th percentile) or higher/lower quantiles.
*   **Asymmetric/Custom Loss:** If you care more about catching the high peaks than the lows, you can write a custom objective function for XGBoost that penalizes under-predictions of high values more severely than over-predictions.

### 4. Ensembling with Extrapolating Models (Stacking)
Because XGBoost cannot predict a value higher than the highest leaf average in its training data, it physically cannot extrapolate.
*   **What to try:** Train a Linear model (e.g., ElasticNet, Ridge) or a small Neural Network (MLP) alongside XGBoost. Use a meta-model (Stacking Regressor) to combine their predictions. 
*   **Why it works:** Linear models and Neural Networks *can* extrapolate. The stack can rely on XGBoost for complex non-linear interactions in the normal range, and lean on the linear model for the extreme high/low peaks.

### 5. Two-Stage Modeling (Mixture of Experts)
It's hard for one tree to accurately model both the dense "normal" days and the rare "extreme" days.
*   **What to try:** 
    1. Train a Classifier to predict if a sample belongs to the "Normal", "Extreme High", or "Extreme Low" regime.
    2. Train separate XGBoost regressors for each subset.
    3. During inference, use the classifier to route the sample to the specialized regressor (or use the class probabilities as new features for a meta-regressor).

### 6. Post-Processing: Variance Inflation
Models optimized for MSE will almost always produce predictions with a lower variance than the actual target. 
*   **What to try:** Linearly scale your predictions so their standard deviation matches the training set's standard deviation.
    $$y_{adj} = (y_{pred} - \bar{y}_{pred}) \times \frac{\sigma_{true}}{\sigma_{pred}} + \bar{y}_{true}$$
*   **Why it works:** It artificially stretches your predictions back out to the original range. **Note:** While this often makes the predictions *look* much better visually against the truth (and fixes the regression to the mean visually), it may actually decrease your $R^2$ or increase MSE slightly because you are shifting predictions away from the mathematically "safest" guess.

---

Ah, that context is very helpful. If the target has multiple peaks (multimodal) and the model will face unseen *combinations* of features (even if the final prediction is within a known range), this explains exactly why a tree-based model like XGBoost is defaulting to the mean.

When a tree encounters an unseen combination of features, it drops the sample into a leaf node based on the splits it *does* know. If that leaf contains training samples from different peaks in your target distribution, the tree will average them together. 

Here are the best ways to tackle **multimodal targets** combined with **unseen feature combinations**:

### 1. Cluster-Then-Predict (Mixture of Experts)
Since your data has multiple peaks (which often correspond to the "Dry", "Transition", and "Wet" regimes you are already analyzing in this project), a single XGBoost model will struggle when a new feature combination blurs the lines between these regimes.
*   **What to try:** 
    1. Force a split: use K-Means (or just your physical thresholds for Dry/Transition/Wet) to divide the training data based on the target variable's peaks.
    2. Train a Classifier to predict which regime/peak an unseen sample belongs to.
    3. Train a separate XGBoost model for each regime.
*   **Why it works:** It prevents the model from averaging a "Dry" sample and a "Wet" sample together when it gets confused. The classifier handles the broad categorization, and the regressors just fine-tune the value within that specific peak.

### 2. Distance-Based Features
Tree models are "blind" to how far an unseen feature combination is from the training data—they just check if a feature is greater than or less than a threshold.
*   **What to try:** Add features that explicitly measure similarity to the training data. For example, fit a clustering algorithm (like K-Means) on your *features* (not the target). Add the distance to each cluster center as new input features.
*   **Why it works:** It gives the tree a sense of "where" the new sample is in the overall feature space, helping it distinguish between "a normal sample that looks like X" and "a completely new unseen combination."

### 3. Ordinal Regression / Multi-class Classification
If predicting the exact continuous value is too noisy for unseen combinations, you can simplify the problem.
*   **What to try:** Discretize your target into 10–20 narrow bins (capturing the multiple peaks). Train the model as a Multi-class Classifier or an Ordinal Regressor to predict the bin, then take the median value of the predicted bin as your final output.
*   **Why it works:** Classification loss functions (like Log Loss) are much better at handling multimodal uncertainty than MSE. If a sample is uncertain, a classifier will output a probability distribution across the peaks, rather than predicting the dead-center between them.

### 4. Stack with a Neural Network or Smooth Regressor
Trees create hard boundaries (step functions). Unseen combinations often fall off these steps awkwardly.
*   **What to try:** Train a Multi-Layer Perceptron (Neural Network) or a regularized linear model (like ElasticNet) alongside XGBoost, and average their predictions.
*   **Why it works:** Neural networks interpolate smoothly between known data points. If a feature combination is unseen, the Neural Network will provide a smooth, continuous guess based on the surrounding space, which can pull the tree's prediction away from a generic leaf average.

### 5. Target Stratified Cross-Validation
If you have multiple peaks, standard random splitting might under-represent a peak in a specific fold, causing the model to learn to ignore it.
*   **What to try:** Ensure you are using stratified sampling on your continuous target (e.g., bin the target and stratify by the bins) when creating your train/val/test splits or cross-validation folds.
*   **Why it works:** It guarantees the model sees a representative amount of data from every peak, preventing it from defaulting to the most common peak (the mean).