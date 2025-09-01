# Conclusions

## Performance Summary

The project began at position 1,427 on the competition leaderboard with a baseline of Regressor Trees tuned using Optuna. Introducing duration-based feature engineering provided the biggest leap, raising the rank to 1,170 with a score of 0.969. Subsequent enhancements—categorical encodings, numerical transformations, and feature interactions—nudged the model further to 1,152 with a final score of 0.96916.

While the initial improvement was significant, progress slowed once performance reached the 0.969 range. This plateau reflects a common reality in well-performing models: each incremental gain requires big effort and deeper experimentation.

## Feature Engineering Journey

The feature engineering process unfolded in several stages:

- **Duration Features**: Binning engagement levels, applying log transformations, and introducing high-engagement flags proved most influential.

- **Categorical Variables**: Target encoding for job and education captured group-level differences effectively.

- **Numerical Variables**: Binning age, campaign frequency, and economic indicators revealed non-linear effects.

- **Feature Interactions**: Combinations such as previous campaign success × call engagement added complementary predictive power.

## Key Takeaways

The move from 1,427 to 1170 shows feature engineering can make a real difference in competition performance. However, the smaller gains afterward highlight how challenging optimization becomes at higher performance levels. The model’s performance remained consistent across different validation methods, including k-fold cross-validation, which suggests that the engineered features captured predictive signals rather than overfitting to noise.