# MPI-based parallel implementation of the Expectation-Maximization algorithm for Gaussian points clustering. 

## Steps
<ul>
  <li><strong>Initialization step</strong>: the master process reads the dataset, divides and distributes the rows, and initializes mean, covariance and weights. The distribution of data is performed using MPI Scatterv for the input matrix and with MPI Bcast for mean, covariance and weights</li>
  <li><strong>Log-likelihood step</strong>: each process computes the log-likelihood on its sub-matrix, then we use MPI Allreduce to estimate the total log-likelihood and distribute the result to each process.</li>
  <li><strong>E-step</strong>: each process computes the soft cluster assignment of its sub-matrix as depicted in the sequential implementation.</li>
  <li><strong>M-step</strong>
    <ul>
      <li>Each process computes the sum of probabilities of assignments for each cluster on its sub-matrix. MPI_Reduce is used to calculate the total sum and the result is sent to the master process.</li>
      <li>Each process computes the numerator of the mean on its sub-matrix. MPI Reduce is called to calculate the total sum gathered from all the processes and the result is sent to the master process. The master process updates the mean values using the result gathered from the processes and the sum of probabilities estimated before</li>
      <li>Each process computes the numerator of the covariance on its sub-matrix. MPI Reduce is called to calculate the total sum gathered from all the processes and the result is sent to the master process. The master process updates the covariance values using the result gathered from the processes and the sum of probabilities estimated before.</li>
      <li>Master process updates the values of the weights using the sum of probabilities estimated before.</li>
      <li>Master process broadcasts updated values of mean, covariance and weights to all the processes with MPI_Bcast</li>
    </ul>
  </li>
  <li><strong>Log-likelihood step</strong>: each process estimates the new log-likelihood of the data on its sub-matrix after each M-step. MPI_Allreduce is used to estimate the total log-likelihood. If the log-likelihood does not change for more than 5 consecutive iterations or the maximum number of iterations is reached the algorithm stops, otherwise, it keeps iterating going back to the E-step.
</li>
</ul>

## Convergence
![loglikelihood](https://github.com/OlyaKhomyn/parallel-expectation-maximization/assets/41692593/27c6f842-fa2d-441e-b15f-6a562677c78a)

## Speedup
![Parallel_1_Speedup](https://github.com/OlyaKhomyn/parallel-expectation-maximization/assets/41692593/dfb16a86-fd18-4ea9-9a36-ff2dc4656edd)
