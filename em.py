#! /usr/bin/env python
import sys
import logging
import math, random, copy
import numpy as np

"""
A practical implementation of EM algorithm with a comparasion with k-means and GMM in sklearn

More references:
    1. http://code.activestate.com/recipes/577735-expectation-maximization/
"""

def normpdf(x, mu, sigma):
    """
    Compute the multivariate normal distribution

    Parameters:
    x : numpy array, shape=[1, n_features]
    mu : numpy array, shape=[1, n_features]
    sigma : numpy array, shape=[n_features, n_features]

    Return:
    res : the probability of x, a scalar value
    """
    xmt = np.matrix(x - mu).transpose()
    for i in xrange(len(sigma)):
        if sigma[i,i] <= sys.float_info[3]:
            sigma[i,i] = sys.float_info[3]
    sinv = np.linalg.inv(sigma)
    xm = np.matrix(x - mu)

    return ((2.0*math.pi) ** (-len(x) / 2.0) * (1.0 / math.sqrt(np.linalg.det(sigma)))
            * math.exp(-0.5 * (xm * sinv * xmt)))

def init_params(X, nclusters, from_dataset=True):
    nsamples, nfeatures = X.shape
    params = []
    for i in xrange(nclusters):
        if from_dataset:
            mu = np.array(1.0 * X[random.uniform(0, nsamples),:], np.float64)
        else:
            mu = np.array([random.uniform(0, 1) for f in xrange(nfeatures)], np.float64)

        params.append(dict(mu=mu, sigma=np.matrix(np.diag([random.uniform(0, 1.0) * 0.5 for f in
                range(nfeatures)])), prob=1.0 / nclusters))
    
    return params

def norm_features(X):
    """
    normalize the feature vector

    Parameters:
    X : numpy array, shape = (nsamples, nfetures)
    """
    nsamples, nfeatures = X.shape
    for f in xrange(nfeatures):
        min_val, max_val = X[:,f].min(), X[:,f].max()
        X[:,f] = (X[:, f] - min_val) / (max_val - min_val)

    return X
    
def EM_cluster(X, nclusters=2, niters=10, normalize=False, epsilon=0.001, datasetinit=True):
    """
    Use EM algorithm to cluster data

    Parameters
    ----------
    X : numpy array, shape = [n_samples, n_features]
    ncluster : number of clusters
    nbiter : number of iterations
    epsilon : the convergence bound/criterium
    """
    nsamples, nfeatures = X.shape
    X = norm_features(X) 
    result = {}
    quality = 0.0
    random.seed()
    cluster_prob_given_obs = np.ndarray([nsamples, nclusters], np.float64)
    obs_prob_given_cluster = np.ndarray([nsamples, nclusters], np.float64)
    
    for niter in xrange(niters):
        # step1: init nclusters parameters
        ncluster_params = init_params(X, nclusters)
        old_neg_log_prob = 0
        neg_log_prob = 2 * epsilon
        estimation_round = 0
        while (abs(neg_log_prob - old_neg_log_prob) > epsilon and (neg_log_prob > old_neg_log_prob)):
            restart = False
            old_neg_log_prob = neg_log_prob
            # step 2: compute p(cluster | obs) for each observation
            for i in xrange(nsamples):
                for c in xrange(nclusters):
                    # compute p(obs | cluster)
                    obs_prob_given_cluster[i, c] = normpdf(X[i,:],
                            ncluster_params[c]['mu'], ncluster_params[c]['sigma'])
            
            for i in xrange(nsamples):
                for c in xrange(nclusters):
                    # compute p(cluster, obs)
                    cluster_prob_given_obs[i, c] = (obs_prob_given_cluster[i, c]
                                        * ncluster_params[c]['prob'])
                # compute p(cluster | obs)
                cluster_prob_given_obs[i, :] /= sum(cluster_prob_given_obs[i,:])
                
            # step 3: update the cluster parameters, (mu, sigma, proba)
            logging.info('iter=%s, estimation#=%s, params=%s', niter,
                    estimation_round, ','.join('(' + ','.join(['%s=%s'% (k, v) for k, v
                            in param.iteritems()]) + ')' for param in ncluster_params))
            
            new_ncluster_params = copy.deepcopy(ncluster_params)
            for c in xrange(nclusters):
                in_cluster_sum = math.fsum(cluster_prob_given_obs[:, c])
                new_ncluster_params[c]['prob'] = in_cluster_sum / nsamples
                if new_ncluster_params[c]['prob'] <= 1.0 / nsamples:
                    restart = True
                    logging.warning('restart, cluster=%s, prob=%s', c, new_ncluster_params[c]['prob'])
                    break
                mu = np.zeros(nfeatures, np.float64)
                sigma = np.matrix(np.diag(np.zeros(nfeatures, np.float64)))
                for i in xrange(nsamples):
                    mu += X[i,:] * cluster_prob_given_obs[i, c] / in_cluster_sum
                new_ncluster_params[c]['mu'] = mu

                for i in xrange(nsamples):
                    sigma += (np.matrix((X[i,:] - mu)).T * np.matrix((X[i,:] -
                            mu)) * cluster_prob_given_obs[i, c] / in_cluster_sum)
                new_ncluster_params[c]['sigma'] = sigma

            if not restart:
                restart = True
                for c in xrange(1, nclusters):
                    if (not np.allclose(new_ncluster_params[c]['mu'], ncluster_params[c]['mu']) 
                        or np.allclose(new_ncluster_params[c]['sigma'], ncluster_params[c]['sigma'])):
                        restart = False
                        break
                    logging.warning('restart, iter=%s, convergence', niter)
            if restart:
                old_neg_log_prob = 0
                neg_log_prob = 2 * epsilon
                ncluster_params = init_params(X, nclusters)
                continue
            
            #import pdb;pdb.set_trace()
            ncluster_params = new_ncluster_params
            # step 4: compute the log estimate
            neg_log_prob = (-1.0 * math.fsum([math.log(math.fsum([obs_prob_given_cluster[i, c] * ncluster_params[c]['prob'] for c in xrange(nclusters)])) for i in xrange(nsamples)]))
            
            logging.info('(EM) old_log_loss=%s, log_loss=%s', old_neg_log_prob, neg_log_prob)

            estimation_round += 1
        # Pick/save the best clustering as the final result
        quality = neg_log_prob
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['params'] = copy.deepcopy(ncluster_params)
            result['clusters'] = [[i for i in xrange(nsamples) if obs_prob_given_cluster[i, c] == max(obs_prob_given_cluster[i,:])] for c in xrange(nclusters)]

    params_str = ','.join('(' + ','.join(['%s=%s'% (k, v) for k, v
                                        in param.iteritems()]) + ')' for
                                        param in result['params'])
    logging.info('quality=%s, \nparams = %s, \nclusters=%s', quality,
            params_str, ','.join('[' + ','.join([str(t)
            for t in c]) + ']' for c in result['clusters']))
    return result

def KMean_cluster(X, nclusters):
    from sklearn.cluster import KMeans
    nsamples, nfeatures = X.shape
    kmean = KMeans(n_clusters=nclusters)
    kmean.fit(X)
    clusters = [[i for i in xrange(nsamples) if kmean.labels_[i] == c] for c in [0, 1]]
    logging.info('k-mean: clusters = %s', ','.join('['+','.join([str(t) for t in
            c])+']' for c in clusters))

def GMM_cluster(X, nclusters):
    from sklearn.mixture import GMM
    nsamples, nfeatures = X.shape
    gmm = GMM(n_components=nclusters)
    gmm.fit(X)
    predicts = gmm.predict(X)
    clusters = [[i for i in xrange(nsamples) if predicts[i] == c] for c in [0, 1]]
    logging.info('gmm: clusters = %s', ','.join('['+','.join([str(t) for t in
            c])+']' for c in clusters))

def logging_config(logger=None, format='%(asctime)s, %(levelname)s %(message)s',
        level=logging.INFO, console_log_level=logging.DEBUG):
    if logger is None:
        logger = logging.getLogger()
    formatter = logging.Formatter(format)
    logger.setLevel(level)
    if console_log_level is not None:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(format)
        ch.setFormatter(logging.Formatter(format))
        ch.setLevel(console_log_level)
        logger.addHandler(ch)

if __name__ == '__main__':
    from numpy.random import multivariate_normal as normal
    from numpy import concatenate, random
    logging_config()
    N = 200
    a = 0.3
    X1 =  normal([0, 0], [[1, 0], [0, 1]], size=(N*a, 1))
    X2 = normal([2, 3], [[1, 0], [0, 1]], size=(N*(1-a), 1))
    X = concatenate([X1,X2])
    X = np.array([t[0] for t in X])
    KMean_cluster(X, nclusters=2)
    EM_cluster(X)
    GMM_cluster(X, nclusters=2)
