# Cost-sensitive Feature Selection for Support Vector Machines

## Introduction
This project aims to find a classifier with minimum feature costs without damaging the original performance too much. 
The reference paper is ["Cost-sensitive Feature Selection for Support Vector Machines"](<https://www.sciencedirect.com/science/article/pii/S0305054818300741> "Title")

- Why cost-sensitive feature selection?
    - Feature acquisition can be costly (in terms of money, time, etc.)
    - E.g., in medical applications, each feature may correspond to one test result
- Why SVM?
    - Mathematical-optimization-based: flexible, additional constraints(e.g., performance guarantee) allowed and objective(e.g., minimizing total cost) to be incorporated
## Mathematical Models
### Phase-I: Feature Selection
- Decision variables
  - $w$: Vector with weights of features
  - $\beta$: Offset
  - $z_k$: Binary variable. Has the value 1 if the $k$-th feature is selected and 0 otherwise
  - $\zeta_i$: Binary variable. Has the value 1 if the $i$-th instance is correctly classified
- Parameters
  - $c_k$: The cost of the $k$-th feature
  - $M_1$, $M_2$: Large numbers
  - $x_i$: Features vector of the $i$-th instance
  - $y_i$: Label of the $i$-th instance, and for amy instance $i$, we have $y_i\in\\{1, -1\\}$
    - The multiclass datasets are transformed into 2-class datasets. The details will be explained in the later section
  - $\lambda_1^\star$: Threshold of **True Positive Rate(TPR)**
  - $\lambda_{-1}^\star$: Threshold of **True Negative Rate(TNR)**
- Formulation <br/>
  $$\text{minimize}_{\textbf{w}, \beta, z, \zeta}\quad \sum_k c_k z_k$$
    
  subject to <br/>
  
  $$y_i(\textbf{w}^T x_i + \beta) \ge 1 - M_1(1-\zeta_i), \quad\forall i \in I$$
  
  $$\sum_{i\in I}\zeta_i(1-y_i) \ge \lambda_{-1}^\star\sum_{i\in I}(1-y_i)$$

  $$\sum_{i\in I}\zeta_i(1+y_i) \ge \lambda_1^\star\sum_{i\in I}(1+y_i)$$

  $$|w_k| \le M_2 z_k, \quad \forall k$$

  $$\zeta_i \in \\{0, 1\\},\quad \forall i$$

  $$z_k \in \\{0, 1 \\},\quad \forall k$$

For both phase-II and phase-III, the result of $z$ in phase-I is embedded, and there are 
- Additional decision variables
  - $\xi_i$: Penalty associated to misclassifying instance $i$
- Additional parameters
  - $M_3$: Large number
  - $C$: Cost of misclassifying

### Phase-II: Linear SVM kernel
- Formulation <br/>
  $$\text{minimize}_{\textbf{w},\beta,\xi}\sum_j w_j^2z_j + C \sum_i\xi_i$$

  subject to <br/>
  $$y_i\left(\sum_j w_jz_jx_{ij} + \beta\right) \ge 1 - \xi_i, \quad \forall i\in I$$

  $$\sum_{i\in I}\zeta_i(1-y_i)\ge \lambda_{-1}^\star\sum_{i\in I}(1-y_i)$$

  $$\sum_{i\in I}\zeta_i(1+y_i)\ge \lambda_1^\star\sum_{i\in I}(1+y_i)$$
  
  $$0\le \xi_i\le M_3(1-\zeta_i),\quad \forall i\in I$$

  $$\zeta_i\in\\{0, 1\\},\quad\forall i\in I$$



### Phase-III: Radial kernel
- Radial kernel
    $$K_z(x, x')=\exp\left(-\gamma\left(\sum_k z_k(x^{(k)}-x'^{(k)})^2\right)\right),$$
    where the value of $z$ is the result of phase_I and $\gamma$ should be tuned
- Formulation(Dual)<br/>
    
    $$\text{minimize}_ {\alpha,\xi,\beta,\zeta} \sum_{i, j}\alpha_i y_i\alpha_j y_j K_z(x_i, x_j)+C\sum_i\xi_i$$
      
    subject to

    $$y_i\left(\sum_j\alpha_j y_j K_z(x_i, x_j)+\beta\right)\ge 1 -\xi_i,\quad\forall i\in I$$

    $$\sum_i\zeta_i(1-y_i) \ge \lambda_{-1}^\star\sum_i(1-y_i)$$

    $$\sum_i\zeta_i(1+y_i) \ge \lambda_1^\star\sum_i(1+y_i)$$ 

    $$\sum_i \alpha_i y_i = 0$$

    $$0\le \alpha_i\le C/2, \quad\forall i\in I$$

    $$0\le \xi_i \le M_3(1-\xi_i), \quad\forall i\in I$$

    $$\zeta_i \in \\{0, 1\\},\quad\forall i\in I$$
## Datasets 
### Data Description
We use the datasets listed in the paper, the following table shows
- Name of the dataset
- Number of samples
- Number of features
- Number of positive samples
 
| Name        | $\|\Omega\|$ | $V$     | $\|\Omega_+\|$ |
| :---        | :----:       | :---:   | :---:          |
| wisconsin   | 569          | 30      |  357 (66.7%)    |
| votes       | 435          | 32      |  267 (61.4%)    |
| nursery     | 12,960       | 19      |  4,320 (33.3%)  |
| australian  | 690          | 34      |  383 (55.5%)    |
| careval     | 1,728        | 15      |  1,210 (70.0%)  |
 
### Pre-processing
- Multiclass datasets are transformed into 2-class datasets, and the positive label is assigned to the majority class
- Categorical variables are transformed into dummy variables
- Missing values are imputed by median(numerical) or by mode(categorical)
- Real-valued data are preprocessed using **RobustScalar** from the module **sklearn** to avoid outliers

## Experiments
### Pseudocode

### Experiment 1: Feature Selection
Run phase-I under $\lambda_{-1}^\star = 0.5$ and $\lambda_1^\star=0.85$ <br/>

| Name       | # Origin features | # Selected features |
| :---       | :---: | :---: |
| wisconsin  | 30    | 1     |
| votes      | 32    | 1     |
| nursery    | 19    | 1     |
| australian | 34    | 1     |
| careval    | 15    | 2     |

As the table shown above, the number of selected features is drastically decreased

### Experiment 2: Performance Analysis under Different $\lambda^\star$
Run the entire flow under different $(\lambda_ {-1}^\star , \lambda_ 1^\star)$ with the dataset *australian* 

| $\lambda_{-1}^\star$ | $\lambda_1^\star$ | Acc.  | TPR   | TNR   | avg. # Feature selected (std.)|
| :---                 | :---:             | :---: | :---: | :---: | :---:            |
| 0.9                  | 0.5               | 0.76  | 0.53  | 0.94  | 1.2 (0.40)       |
| 0.85                 | 0.5               | 0.73  | 0.56  | 0.87  | 1 (0.00)         |
| 0.85                 | 0.575             | 0.81  | 0.65  | 0.93  | 1.9 (0.30)       | 
| 0.85                 | 0.6               | 0.82  | 0.67  | 0.94  | 2 (0.00)         |
| 0.75                 | 0.85              | 0.86  | 0.92  | 0.79  | 1 (0.00)         |
| 0.8                  | 0.85              | 0.84  | 0.88  | 0.80  | 1.7 (0.46)       |
| 0.75                 | 0.9               | 0.86  | 0.92  | 0.79  | 1 (0.00)         |
| 0.8                  | 0.9               | 0.86  | 0.92  | 0.80  | 1.7 (0.46)       |
| 0.85                 | 0.9               | -     | -     | -     | -                |
| 0.65                 | 0.95              | 0.80  | 0.93  | 0.68  | 2.8 (0.40)       |

Note: mathematical model could be infeasible with some $(\lambda_{-1}^\star, \lambda_1^\star)$, e.g., $(\lambda_{-1}^\star = 0.85, \lambda_1^\star = 0.9)$
## Discussion
### Theoretical contribution

### Experimental contribution 
- The number of features can be substantially reduced using the proposed method with the datasets from the paper
### Problems & Limitation
  - All datasets required at most 10% of original features only. Should try more datasets to support the robustness of the model
  - Due to computation resources limitation, we tuned $C$ and $\gamma$ (if the radial kernel is used) only for the first fold for $k$-fold
### Other extension
  - Effects of varying feature costs(currently all features have unit cost in our experiment, i.e., $c_k = 1$ for all $k$)
  - Move TPR and TNR constraints into  objective function <br/>
    $$\text{minimize}_ {\textbf{w}, \beta, z, \zeta, \lambda_ {-1}^\star , \lambda_1^\star} \quad \sum_k c_k z_k - \sum_i(\zeta_i - \lambda_{-1}^\star)(1-y_ i) - \sum_ i (\zeta_ i-\lambda_ 1^\star)(1 + y_ i),$$
     
    which minimizes total cost with consideration of maximizing $\lambda_1^\star$ and $\lambda_{-1}^\star$
