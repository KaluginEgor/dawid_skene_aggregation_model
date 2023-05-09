import pandas as pd
import numpy as np

_EPS = np.float_power(10, -10)

class DawidSkene:

    def __init__(self, n_iter = 100, tolerance = 0.01):
        self.n_iter = n_iter
        self.tolerance = tolerance
    
    def _e_step(self, data, priors, errors):
        joined = data.join(np.log2(errors), on=['worker', 'label'])
        joined.drop(columns=['worker', 'label'], inplace=True)
        log_likelihoods = np.log2(priors) + joined.groupby('task', sort=False).sum()
        log_likelihoods.rename_axis('label', axis=1, inplace=True)
        scaled_likelihoods = np.exp2(log_likelihoods.sub(log_likelihoods.max(axis=1), axis=0))
        return scaled_likelihoods.div(scaled_likelihoods.sum(axis=1), axis=0)
    
    def _m_step(self, data, probas):
        joined = data.join(probas, on='task')
        joined.drop(columns=['task'], inplace=True)

        errors = joined.groupby(['worker', 'label'], sort=False).sum()
        errors.clip(lower=_EPS, inplace=True)
        errors /= errors.groupby('worker', sort=False).sum()

        return errors
    
    def _evidence_lower_bound(self, data, probas, priors, errors):
        joined = data.join(np.log(errors), on=['worker', 'label'])

        joined = joined.rename(columns={True: 'True', False: 'False'}, copy=False)
        priors = priors.rename(index={True: 'True', False: 'False'}, copy=False)

        joined.loc[:, priors.index] = joined.loc[:, priors.index].add(np.log(priors))

        joined.set_index(['task', 'worker'], inplace=True)
        joint_expectation = (probas.rename(columns={True: 'True', False: 'False'}) * joined).sum().sum()

        entropy = -(np.log(probas) * probas).sum().sum()
        return float(joint_expectation + entropy)

    def fit(self, data):
        data = data[['task', 'worker', 'label']]

        scores = data[['task', 'label']].value_counts().unstack('label', fill_value=0)
        probas = scores.div(scores.sum(axis=1), axis=0)
        priors = probas.mean()
        errors = self._m_step(data, probas)
        loss = -np.inf
        self.loss_history_ = []

        for _ in range(self.n_iter):
            probas = self._e_step(data, priors, errors)
            priors = probas.mean()
            errors = self._m_step(data, probas)
            new_loss = self._evidence_lower_bound(data, probas, priors, errors) / len(data)
            self.loss_history_.append(new_loss)

            if new_loss - loss < self.tolerance:
                break
            loss = new_loss

        probas.columns = pd.Index(probas.columns, name='label', dtype=probas.columns.dtype)

        self.probas_ = probas
        self.priors_ = priors
        self.errors_ = errors
        self.labels_ = probas.idxmax(axis='columns')

        return self
    
    def fit_predict_proba(self, data):
        return self.fit(data).probas_

    def fit_predict(self, data):
        return self.fit(data).labels_
