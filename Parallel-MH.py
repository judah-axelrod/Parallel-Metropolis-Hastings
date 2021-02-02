# Import required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import scipy.stats as st
from multiprocessing import Pool
from functools import partial


# Define function to calculate log-posterior density, from which we wish to sample from
def log_posterior(data, mu, sigma):
    # Assumed priors are mu~N(0,100) and sigma~IGamma(0.01, 0.01)
    log_prior = np.log(st.norm(0, 10).pdf(mu)) + np.log(st.invgamma(a=0.01, scale=1 / 0.01).pdf(sigma))

    # Know (or assume if not known) that the data comes from a normal distribution
    log_lik = np.sum(np.log(st.norm(mu, sigma).pdf(data)))

    # Log-posterior density is the sum of the log-prior and log-likelihood
    log_post_density = log_prior + log_lik
    return (log_post_density)


def MH_sampler(samples, data, mu_init, sigma_init, proposal_sd=[0.05, 0.05]):
    # Initialisation
    acceptances = 0
    mu_current, sigma_current = mu_init, sigma_init
    chain = [[mu_current], [sigma_current]]  # First value in the sample
    rejected = [[np.nan], [np.nan]]

    for _ in range(1, samples):
        # Draw Proposals
        mu_proposal, sigma_proposal = np.random.normal([mu_current, sigma_current], proposal_sd, (2,))

        # Calculate Acceptance Probability
        proposal_log_posterior = log_posterior(data, mu_proposal, sigma_proposal)
        current_log_posterior = log_posterior(data, mu_current, sigma_current)
        p_accept = np.min([1, np.exp(proposal_log_posterior - current_log_posterior)])

        # Accept/Reject Step
        if np.random.uniform(0, 1) < p_accept:
            acceptances += 1
            rejected[0].append(np.nan), rejected[1].append(np.nan)
            mu_current, sigma_current = mu_proposal, sigma_proposal
        else:
            rejected[0].append(mu_proposal), rejected[1].append(sigma_proposal)

        chain[0].append(mu_current), chain[1].append(sigma_current)

    return (chain, rejected, acceptances)


if __name__ == "__main__":

    # Generate data
    np.random.seed(444)
    data = st.norm(5, 2).rvs(1000)
    mu_obs = np.mean(data)
    sigma_obs = np.std(data)
    plt.hist(data)
    plt.title('Histogram of N(5,2)')
    plt.show()
    print('Observed sample mean = ' + str(mu_obs) + '\nObserved sample std = ' + str(sigma_obs))

    # Set up parallel/non-parallel runs
    loop = True
    while loop == True:
        toggle = input("Enter 'P' for parallel MH or enter 'X' for non-parallel MH: ")
        if toggle == 'P':
            parallel, loop = True, False
            print('Running parallel MH...')
        elif toggle == 'X':
            parallel, loop = False, False
            print('Running unparallel MH...')
        else:
            print("ERROR: Only enter 'P' or 'X'.")

    # Designated number of samples to be drawn
    samples = 10000

    # Designated # of Burn-in steps
    burnin = 200

    # Start the timer (assume time to check value of 'parallel' is negligible)
    start = timeit.default_timer()

    # Generate MH samples (non-parallel)
    if parallel == False:
        nchains = 1
        np.random.seed(444)
        results = [MH_sampler(samples, data, mu_obs, sigma_obs)]
        chain, rejected, n_accept = results[0][0], results[0][1], results[0][2]
        chain[0] = chain[0][burnin:]
        chain[1] = chain[1][burnin:]
        rejected[0] = rejected[0][burnin:]
        rejected[1] = rejected[1][burnin:]
        print("Acceptance Rate): ", 100 * n_accept / samples, "%")

    # Generate MH samples (non-parallel)
    elif parallel == True:
        nchains = 4
        samples_per_chain = int(samples / nchains)
        p = Pool(processes=4)
        target = partial(MH_sampler, samples_per_chain, data, mu_obs, sigma_obs)
        results = p.map(target, [[0.05, 0.05] for _ in range(nchains)])
        chain = [[], []]
        rejected = [[], []]
        for i in range(nchains):
            chain[0].extend(results[i][0][0][burnin:]), chain[1].extend(results[i][0][1][burnin:])
            rejected[0].extend(results[i][1][0][burnin:]), rejected[1].extend(results[i][1][1][burnin:])
            n_accept = results[i][2]
            print("Acceptance Rate (chain ", i + 1, "): ", 100 * n_accept / samples_per_chain, "%")

    # End the time and analyse run time
    end = timeit.default_timer()
    print("Run Time:", str(end - start), "secs")

    for i in range(nchains):
        # Trace Plots of mu and sigma for each chain
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        for j in range(2):
            axes[j].plot(results[i][0][j])
            axes[j].set_xlabel('Iteration')
        axes[0].set_ylabel('$\mu$')
        axes[1].set_ylabel('$\sigma$')
        if parallel:
            plt.suptitle("Trace Plots for $\mu$ and $\sigma$ \nChain " + str(i + 1))
        else:
            plt.suptitle("Trace Plots for $\mu$ and $\sigma$")
        plt.show()

    # Joint Accept-Reject plot of mu-sigma
    mh_data = pd.DataFrame({'mu_samples': chain[0], 'mu_rejected': rejected[0], \
                            'sigma_samples': chain[1], 'sigma_rejected': rejected[1]}).reset_index()

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='mu_rejected', y='sigma_rejected', data=mh_data, color='red', marker='x')
    sns.scatterplot(x='mu_samples', y='sigma_samples', data=mh_data, color='blue', linewidth=0.1, s=50)
    axes.set_xlabel('$\mu$')
    axes.set_ylabel('$\sigma$')
    axes.set_title('Metropolis Hastings samples for the posterior distributions of $\mu$ and $\sigma$');
    axes.legend(labels=['Rejected Proposals', 'Samples'])
    plt.show()

    # Traceplots and Histograms of mu and sigma individually
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    for j in range(2):
        sns.histplot(chain[j], stat='density', color='red', ax=axes[j])
        axes[j].set_ylabel('Density')

    axes[0].set_ylabel('$\mu$')
    axes[1].set_ylabel('$\sigma$')
    plt.suptitle("Histograms for Posterior Density Functions of $\mu$ and $\sigma$")
    plt.show()

    # Summary output  
    np.random.seed(444)
    results = MH_sampler(samples, data, mu_obs, sigma_obs)
    print("{} samples".format(samples))
    print("Observed mu = {:.3f}, observed sigma = {:.3f}".format(mu_obs, sigma_obs))
    print("Burn-in = {}".format(burnin))
    print("Posterior averages: mu = {:.3f}, sigma = {:.3f}".format(np.mean(results[0][0][burnin:]),
                                                               np.mean(results[0][1][burnin:])))