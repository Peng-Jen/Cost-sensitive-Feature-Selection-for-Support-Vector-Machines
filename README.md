## Cost-sensitive Feature Selection for Support Vector Machines

### Introduction
This project is aim to find a classifier with minimum feature costs without damaging the original performance too much. 
The reference paper is ["Cost-sensitive Feature Selection for Support Vector Machines"](<https://www.sciencedirect.com/science/article/pii/S0305054818300741> "Title")
- Why cost-sensitive feature selection?
    - Feature acquisition can be costly (in terms of money, time, etc.)
    - E.g., in medical application, each feature may corresponds to one test result
- Why SVM?
    - Mathematical-optimization-based: flexible, additional constraints(e.g., performance guarantee) allowed and objective(e.g., minimizing total cost) to be incoporated
### Mathematical Models
#### Phase-I: Feature Selection
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

#### Phase-II: Linear SVM kernel
- Formulation <br/>
  $$\text{minimize}_{\textbf{w},\beta,\xi}\sum_j w_j^2z_j + C \sum_i\xi_i$$

  subject to <br/>
  $$y_i\left(\sum_j w_jz_jx_{ij} + \beta\right) \ge 1 - \xi_i, \quad \forall i\in I$$

  $$\sum_{i\in I}\xi_i(1-y_i)\ge \lambda_{-1}^*\sum_{i\in I}(1-y_i)$$

  $$\sum_{i\in I}\xi_i(1+y_i)\ge \lambda_1\sum_{i\in I}(1+y_i)$$
  
  $$0\le \xi_i\le M_3(1-\xi_i),\quad \forall i\in I$$

  $$\xi_i\in\\{0, 1\\},\quad\forall i\in I$$



#### Phase-III: Radial kernel
- Radial kernel:
    $$K_z(x, x')=\exp\left(-\gamma\left(\sum_k z_k(x^{(k)}-x'^{(k)})^2\right)\right),$$
    where the value of $z$ is the result of phase_I and $\gamma$ should be tuned
- Formulation(Dual)
    $$\text{minimize}_{\alpha,\xi,\beta,\zeta}\sum_{i,j}\alpha_i y_i \alpha_j y_j K_z(x_i, x_j) + C\sum_{i\in I}\xi_i$$

    subject to
    $$y_i\left(\sum_j \alpha_j y_j K_z(x_i, x_j) + \beta\right) \ge 1 - \xi_i,\quad\forall i\in I$$

    $$\sum_i\zeta_i(1-y_i) \ge \lambda_{-1}^*\sum_i(1-y_i)$$

    $$\sum_i\zeta_i(1+y_i) \ge \lambda_1^*\sum_i(1+y_i)$$ 

    $$\sum_i \alpha_i y_i = 0$$

    $$0\le \alpha_i\le C/2, \quad\forall i\in I$$

    $$0\le \xi_i \le M_3(1-\xi_i), \quad\forall i\in I$$

    $$\zeta_i \in \\{0, 1\\},\quad\forall i\in I$$
### Experiments
#### Datasets 
#### Pseudocode
### Discussion
