"""
This is to join pre-processing (PCA, UMAP (LDA?)) and gradient boosting classifier in a non leaky way

I want to add PCA, UMAP (and may be LDA?) to the feature set, but retain original features; next, preprocessing is
to be combined with classifier using sklearn pipeline, so that it does not leak in CV.
FeatureUnion is, likely, way to go. Since I don't know how to retain automatically the original features,
SelectKBest(k=len(features_list)) can be a workaround.

pipeline example
https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py
"""


