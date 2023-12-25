# Cost-sensitive Feature Selection for Support Vector Machines

## Introduction
This project is aim to find a classifier with minimum feature costs without damaging the original performance too much. 
The reference paper is ["Cost-sensitive Feature Selection for Support Vector Machines"](<https://www.sciencedirect.com/science/article/pii/S0305054818300741> "Title")

- Why cost-sensitive feature selection?
    - Feature acquisition can be costly (in terms of money, time, etc.)
    - E.g., in medical application, each feature may corresponds to one test result
- Why SVM?
    - Mathematical-optimization-based: flexible, additional constraints(e.g., performance guarantee) allowed and objective(e.g., minimizing total cost) to be incoporated
## Mathematical Models
### Phase-I: Feature Selection
- Decision variables
  - $w$ : Vector with weights of features
  - $\beta$ : Offset
  - $z_k$ : Binary variable. Has the value 1 if the $k$-th feature is selected and 0 otherwise
  - $\zeta_i$ : Binary variable. Has the value 1 if the $i$-th instance is correctly classified
- Parameters
  - $c_k$ : The cost of the $k$-th feature
  - $M_1$, $M_2$ : Large numbers
  - $x_i$ : Features vector of the $i$-th instance
  - $y_i$ : Label of the $i$-th instance, and for amy instance $i$, we have $y_i\in\\{1, -1\\}$
    - The multiclass datasets are transformed into 2-class dataset. The details will be explained in the later section
  - $\lambda_1^*$ : Threshold of **True Positve Rate(TPR)**
  - $\lambda_{-1}^*$ : Threshold of **True Negative Rate(TNR)**
- Formulation <br/>
  $$\text{minimize}_{\textbf{w}, \beta, z, \zeta}\quad \sum_k c_k z_k$$
    
  subject to <br/>
  
  $$y_i(\textbf{w}^T x_i + \beta) \ge 1 - M_1(1-\zeta_i), \quad\forall i \in I$$
  
  $$\sum_{i\in I}\zeta_i(1-y_i) \ge \lambda_{-1}^*\sum_{i\in I}(1-y_i)$$

  $$\sum_{i\in I}\zeta_i(1+y_i) \ge \lambda_1^*\sum_{i\in I}(1+y_i)$$

  $$|w_k| \le M_2 z_k, \quad \forall k$$

  $$\zeta_i \in \\{0, 1\\},\quad \forall i$$

  $$z_k \in \\{0, 1 \\},\quad \forall k$$

For both phase-II and phase-III, the result of $z$ in phase-I is embedded, and there are 
- Additional decision variables
  - $\xi_i$ : Penalty associated to misclassifying instance $i$
- Additional parameters
  - $M_3$ : Large number
  - $C$ : Cost of misclassifying

### Phase-II: Linear SVM kernel
- Formulation <br/>
  $$\text{minimize}_{\textbf{w},\beta,\xi}\sum_j w_j^2z_j + C \sum_i\xi_i$$

  subject to <br/>
  $$y_i\left(\sum_j w_jz_jx_{ij} + \beta\right) \ge 1 - \xi_i, \quad \forall i\in I$$

  $$\sum_{i\in I}\zeta_i(1-y_i)\ge \lambda_{-1}^*\sum_{i\in I}(1-y_i)$$

  $$\sum_{i\in I}\zeta_i(1+y_i)\ge \lambda_1\sum_{i\in I}(1+y_i)$$
  
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

    $$\sum_i\zeta_i(1-y_i) \ge \lambda_{-1}^*\sum_i(1-y_i)$$

    $$\sum_i\zeta_i(1+y_i) \ge \lambda_1^*\sum_i(1+y_i)$$ 

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
| wisconsin   | 569          | 30      |  357(66.7%)    |
| votes       | 435          | 32      |  267(61.4%)    |
| nursery     | 12,960       | 19      |  4,320(33.3%)  |
| australian  | 690          | 34      |  383(55.5%)    |
| careval     | 1,728        | 15      |  1,210(70.0%)  |
 
### Pre-processing
- Multiclass datasets are transformed into 2-class datasets, and the positive label is assigned to the majority class
- Categorical variables are transformed into dummy variables
- Missing values are imputed by median(numerical) or by mode(categorical)
- Real-valued data are preprocessed using **RobustScalar** from the module **sklearn** to avoid outliers

## Experiments
### Pseudocode

### Experiment 1: Feature Selection
Run phase-I under $\lambda_{-1}^\*=0.5$ and $\lambda_1^\*=0.85$ <br/>

| Name       | # Origin features | # Selected features |
| :---       | :---: | :---: |
| wisconsin  | 30 | 1 |
| votes      | 32 | 1 |
| nursery    | 19 | 1 |
| australian | 34 | 1 |
| careval    | 15 | 2 |

As the table shown above, the number of selected features is drastically decreased

### Experiment 2: Performance Analysis under Different $\lambda$
- Run the entire flow under different $(\lambda_{-1}^\*, \lambda_1^\*)$ with the dataset *australian* 

| $\lambda_1^*$ | $\lambda_{-1}^*$ | Acc.  | TPR   | TNR   | avg. # Feature selected |
| :---          | :---:            | :---: | :---: | :---: | :---:                   |
| 0.5           | 0.85             | 0.88  | 0.77  | 0.94  | 1                       |
| 0.55          | 0.85             | 0.73(0.04)  | 0.56(0.06)  | 0.87(0.06)  | 1(0)                       |
| 0.575         | 0.85             | 0.81(0.06)  | 0.65(0.11)  | 0.93(0.11)  | 1.9(0.3)                     | 
| 0.6           | 0.85             | 0.82(0.04)  | 0.67(0.08)  | 0.94(0.08)  | 2(0)                       |
| 0.5           | 0.9              | 0.76(0.05)  | 0.53(0.11)  | 0.94(0.11)  | 1.2(0.4)                     |

## Discussion
- The number of features can be sustantially reduced using the proposed method with the datasets from the paper
- Problem: All datasets required at most 2 features only
- Other extension:
  - Effects of varying feature costs(currently all features have unit cost in our experiment)
  - Move TPR and TNR constraints into  objective function 
    $$\text{minimize}_{\textbf{w}, \beta, z, \zeta, \lambda_{-1}^\*, \lambda_1^\*}\quad \sum_k c_k z_k - \sum_i (\zeta_i - \lambda_{-1}^\*)(1-y_i) - \sum_i (\zeta_i - \lambda_1^\*)(1+y_i),$$
    
    which minimizing total cost with consideration of maximizing $\lambda_1^\*$ and $\lambda_{-1}^\*$
