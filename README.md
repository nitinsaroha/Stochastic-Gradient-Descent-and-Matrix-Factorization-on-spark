# Stochastic-Gradient-Descent-for-Matrix-Factorization-on-spark
Implementation of Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent (DSGD-MF) in Apache Spark.

Dataset used is a subsample of Netflix Data in user,movie,rating format

#### How to Run
  **$** **spark-submit matrix_factorization_dsgd.py 30 20 10 0.6 0.1**
  
- First argument is number of Iterations
- Second argument is number of workers
- Third argument is number of factors
- Fourth argument is beta value
- Fifth argument is lambda value

*Note:- spark-submit should be in path*

For Scipy Svd:-

  **$** **spark-submit scipy_svd.py**
