# Matrix Factorizer using TensorFlow

This is some proof-of-concept code for doing matrix factorization using TensorFlow for the purposes of making content recommendations. It was inspired by the following papers on matrix factorization:

- [Matrix Factorization techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)
- [Predicting movie ratings and recommender systems](http://arek-paterek.com/book/)

## Example usage
Ratings triplets (COO sparse matrix format) need to be available as a feather file with columns user_id, item_id and rating, where user_id is the zero-based user index, item_id is the zero-based item index, and rating is the rating for the specified (user, item) pair.

The path to the feather file is passed in as the first argument.

Usage:

`python factorizer.py [path/to/triplets.feather] [maximum_iterations] [what_to_learn] [[regularization_parameter] [rank]]`

For example, to learn 5 latent features:

`python factorizer.py ~/Desktop/ratings_triplets.feather 100 features-only 10.0 5`