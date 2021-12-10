# CyOpT


## Description


## Install CytOpt


```
pip install -r requirements.txt
pip install CytOpt # pip3 install CytOpt
```

## Example

### Packages

```
from CytOpt.Cytopt import CytOpt
```


### Preparing data

```
# Source Data
Stanford1A_values = pd.read_csv('../tests/data/W2_1_values.csv',
                                    usecols=np.arange(1, 8))
                                    
Stanford1A_clust = pd.read_csv('../tests/data/W2_1_clust.csv',
                                   usecols=[1])

    # Target Data
Stanford3A_values = pd.read_csv('../tests/data/W2_7_values.csv',
                                    usecols=np.arange(1, 8))

Stanford3A_clust = pd.read_csv('../tests/data/W2_7_clust.csv',
                                   usecols=[1])

X_source = np.asarray(Stanford1A_values)

X_target = np.asarray(Stanford3A_values)

Lab_source = np.asarray(Stanford1A_clust['x'])

Lab_target = np.asarray(Stanford3A_clust['x'])

theta_true = np.zeros(10)

for k in range(10):
    theta_true[k] = np.sum(Lab_target == k + 1) / len(Lab_target)
```
### Comparison of methods
#### Steps
###### Classification using optimal transport with reweighted proportions. 
###### The target measure  𝛽  is reweighted in order to match the weight vector  ℎ̂   estimated with  𝙲𝚢𝚝𝙾𝚙𝚝.
###### Approximation of the optimal dual vector u. In order to compute an approximation of the optimal transportation plan, we need to approximate  𝑃𝜀 .
###### Class proportions estimation with  𝙲𝚢𝚝𝙾𝚙𝚝 Descent-Ascent procedure Setting of the parameters
###### Minmax swapping procedure. Setting of the parameters
###### Plot all Bland-Altman


```
CytOpt(X_source, X_target, Lab_source, Lab_target=None, cell_type=None,
                              method="comparison_opt", theta_true=theta_true, eps=1e-04, n_iter=4000, power=0.99,
                              step_grad=50, step=5, lbd=1e-04, n_out=1000, n_stoc=10, n_0=10,
                              n_stop=1000, monitoring=False, minMaxScaler=True)
```

### CytOpT Minmax or Desasc 
```
from CytOpt import cytopt_minmax, cytopt_desasc

cytopt_desasc(X_source, X_target, Lab_source, eps=0.0001, n_out=4000, n_stoc=10, step_grad=50, const=0.1, theta_true=0)

cytopt_minmax(X_source, X_target, Lab_source,eps=0.0001, lbd=0.0001, n_iter=4000,
                  step=5, power=0.99, theta_true=0, monitoring=False)
                  
/// Or use CytOpt function with specified method parameter
```

## Documents

## Urls
https://arxiv.org/abs/2006.09003