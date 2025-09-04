# Conclusions

## Performance Summary

The project started at position 1,427 with a score of 0.96715 on the competition leaderboard. **Optuna optimization of the baseline Regressor Trees provided the largest boost**, raising the rank to 1,193 with a score of 0.96878. Subsequent duration-based feature engineering further improved the rank slightly to 1,170 with a score of 0.969, but the gains were comparatively small. Further enhancements, such as categorical encodings, numerical transformations, and feature interactions, only yielded marginal improvements. In fact, **categorical encodings were the only ones that slightly improved the ranking**, while the rest had negligible impact. This illustrates how, at high metric values, it becomes increasingly difficult to achieve meaningful improvements, and not all optimizations translate into better results.


## Feature Engineering Journey

The feature engineering process unfolded in several stages:

- **Duration Features**: Binning engagement levels, applying log transformations, and introducing high-engagement flags proved most influential.

- **Categorical Variables**: Target encoding for job and education captured group-level differences effectively.

- **Numerical Variables**: Binning age, campaign frequency, and economic indicators revealed non-linear effects.

- **Feature Interactions**: Combinations such as previous campaign success × call engagement added complementary predictive power.

!!! warning "Binning Numerical Features"
    Splitting numerical features into bins may not improve performance for tree-based models. However, this app is implemented following a factory pattern, so new models based on regression or other approaches can easily take advantage of it. The implementation is already in place and ready to be used for such models.

## Key Takeaways

The move from 1,427 to 1170 shows feature engineering can make a difference in competition performance. However, the **smaller gains** afterward **highlight how challenging optimization becomes** at higher performance levels. The **model’s performance remained consistent across different validation methods**, including k-fold cross-validation, which suggests that the engineered features captured predictive signals rather than overfitting to noise.