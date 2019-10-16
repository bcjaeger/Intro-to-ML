---
  layout: false
background-image: url(Intro_to_Predictive_Analytics_files/cat_snow_gif.gif)
background-position: 50% 60%
  background-size: 50%

# In summary...

---
  layout: true
background-image: url(hex_stickers/PNG/dplyr.png)
background-position: 95% 2.5%
  background-size: 13%

# A tangent for programmers

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    nsubs=10
    # start with nsubs ...
    nsubs %>% #<<
      # then generate...
      runif(0, 10) %>% 
      # then frame it
      enframe() %>%
      # rename it
      rename(x=value) %>% 
      # create an outcome
      mutate(
        y =
          # signal
          x*(3+sin(pi*x/3)) + 
          # noise
          rnorm(nsubs,sd=x)
      ) ->
      # and call it...
      simulated_data
    
    ```
    ]

.pull-right[
  ```{r, echo=FALSE}
  nsubs
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    nsubs=10
    # start with nsubs ...
    nsubs %>% 
      # then generate...
      runif(0, 10) %>% #<<
      # then frame it
      enframe() %>%
      # rename it
      rename(x=value) %>% 
      # create an outcome
      mutate(
        y =
          # signal
          x*(3+sin(pi*x/3)) + 
          # noise
          rnorm(nsubs,sd=x)
      ) ->
      # and call it...
      simulated_data
    
    ```
    ]

.pull-right[
  ```{r, echo=FALSE}
  rvals=simulated_data$x
  suppressWarnings(
    rvals %>% matrix(data=., ncol=1)
  )
  ```
  ]

---
  
  
  .pull-left[
    ```{r, echo=TRUE}
    
    nsubs=10
    # start with nsubs ...
    nsubs %>% 
      # then generate...
      runif(0, 10) %>% 
      # then frame it
      enframe() %>% #<<
      # rename it
      rename(x=value) %>% 
      # create an outcome
      mutate(
        y =
          # signal
          x*(3+sin(pi*x/3)) + 
          # noise
          rnorm(nsubs,sd=x)
      ) ->
      # and call it...
      simulated_data
    
    ```
    ]

.pull-right[
  ```{r, echo=FALSE}
  rvals %>% enframe() %>% as_tibble() %>% print()
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    nsubs=10
    # start with nsubs ...
    nsubs %>% 
      # then generate...
      runif(0, 10) %>% 
      # then frame it
      enframe() %>% 
      # rename it
      rename(x=value) %>% #<<
      # create an outcome
      mutate(
        y =
          # signal
          x*(3+sin(pi*x/3)) + 
          # noise
          rnorm(nsubs,sd=x)
      ) ->
      # and call it...
      simulated_data
    
    ```
    ]

.pull-right[
  ```{r, echo=FALSE}
  rvals %>% enframe() %>% 
    as_tibble() %>% rename(x=value)
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    nsubs=10
    # start with nsubs ...
    nsubs %>% 
      # then generate...
      runif(0, 10) %>% 
      # then frame it
      enframe() %>% 
      # rename it
      rename(x=value) %>% 
      # create an outcome
      mutate( #<<
        y = #<<
          # signal #<<
          x*(3+sin(pi*x/3)) + #<<
          # noise #<<
          rnorm(nsubs,sd=x) #<<
      ) ->
      # and call it...
      simulated_data
    
    ```
    ]

.pull-right[
  ```{r, echo=FALSE}
  simulated_data
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    nsubs=10
    # start with nsubs ...
    nsubs %>% 
      # then generate...
      runif(0, 10) %>% 
      # then frame it
      enframe() %>% 
      # rename it
      rename(x=value) %>% 
      # create an outcome
      mutate(
        y = 
          # signal
          x*(3+sin(pi*x/3)) +
          # noise
          rnorm(nsubs,sd=x)
      ) -> #<<
      # and call it... #<<
      simulated_data #<<
    
    ```
    ]

.pull-right[
  ```{r, echo=TRUE}
  print(simulated_data)
  ```
  ]



```{r}
x=simulated_data$x
xgrid=seq(min(x),max(x),length.out=100)
truth=data.frame(x=xgrid)%>%mutate(y=x*(3+sin(pi*x/3)))

```

---
  layout: true
background-image: url(hex_stickers/PNG/ggplot2.png)
background-position: 95% 2.5%
  background-size: 13%

# A tangent for programmers

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    simulated_data %>% #<<
      ggplot( #<<
        aes( #<<
          x = x, y = y  #<<
        ) #<<
      ) + #<<
      labs(
        title='Simulated Data',
        x='X-value',
        y='Y-value'
      ) +
      geom_point()+
      geom_line(
        data=truth,
        aes(x=x,y=y),
        color='red',
        linetype=2
      )
    
    ```
    ]

.pull-right[
  
  ```{r, fig.width=6, fig.height=7.5}
  
  simulated_data %>% 
    ggplot(
      aes(
        x = x, y = y  
      )
    )
  
  ```
  
  ]

---
  
  
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    simulated_data %>% 
      ggplot(
        aes(
          x = x, y = y  
        )
      ) +
      labs( #<<
        title='Simulated Data', #<<
        x='X-value', #<<
        y='Y-value' #<<
      ) #<<
    geom_point()+
      geom_line(
        data=truth,
        aes(x=x,y=y),
        color='red',
        linetype=2
      )
    
    ```
    ]

.pull-right[
  
  ```{r, fig.width=6, fig.height=7.5}
  
  simulated_data %>% 
    ggplot(
      aes(
        x = x, y = y  
      )
    ) + 
    labs(
      title='Simulated Data',
      x='X-value',
      y='Y-value'
    )
  
  ```
  
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    simulated_data %>% 
      ggplot(
        aes(
          x = x, y = y  
        )
      ) +
      labs(
        title='Simulated Data',
        x='X-value', 
        y='Y-value'
      )
    geom_point()+ #<<
      geom_line(
        data=truth,
        aes(x=x,y=y),
        color='red',
        linetype=2
      )
    
    ```
    ]

.pull-right[
  
  ```{r, fig.width=6, fig.height=7.5}
  
  simulated_data %>% 
    ggplot(
      aes(
        x = x, y = y  
      )
    )+
    labs(
      title='Simulated Data',
      x='X-value',
      y='Y-value'
    ) +
    geom_point()
  
  
  ```
  
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    simulated_data %>% 
      ggplot(
        aes(
          x = x, y = y  
        )
      ) +
      labs(
        title='Simulated Data',
        x='X-value', 
        y='Y-value'
      ) +
      geom_point()+ 
      geom_line( #<<
        data=truth, #<<
        aes(x=x,y=y), #<<
        color='red', #<<
        linetype=2 #<<
      ) #<<
    
    ```
    ]

.pull-right[
  
  ```{r, fig.width=6, fig.height=7.5}
  
  simulated_data %>% 
    ggplot(
      aes(
        x = x, y = y  
      )
    )+
    labs(
      title='Simulated Data',
      x='X-value',
      y='Y-value'
    ) +
    geom_point() +
    geom_line( 
      data=truth,
      aes(x=x,y=y),
      color='red', 
      linetype=2 
    )
  
  ```
  
  ]

<!-- --- -->
  
  <!-- **Problem:** Predict a testing point $y$ with predictor values $x$. -->
  
  <!-- - Denote  -->
  <!--  - the derivation data as $\mathcal{D}$ drawn from a population $\mathcal{P}$,  -->
  <!--  - the prediction rule as $\widehat{y} = \hat{f}(x)$,  -->
  <!--  - and the true mapping as $y = f(x) + \varepsilon$. -->
  
  <!-- -- -->
  
  <!-- - The expected prediction error is: $$\text{MSE}(x) =  E_{\mathcal{D}} \left[y - \widehat{y}\right]^2 = E_{\mathcal{D}}\left[ (f(x) + \varepsilon -\widehat{y})^2 \right]$$ where $E_{\mathcal{D}}[\cdot]$ is the average value of $[\cdot]$ taken over all possible derivation sets from $\mathcal{P}$ -->
  
  <!-- -- -->
  
  <!-- - Add a well chosen 0, $E_{\mathcal{D}}[\widehat{y}]-E_{\mathcal{D}}[\widehat{y}]$, expand $(a+b+c)^2$ and simplify  -->
  
  <!-- $$\text{Bias-squared}=\left(y - E_{\mathcal{D}}[\widehat{y}] \right)^2, \hspace{2cm} \text{Variance} = E_{\mathcal{D}}\left[ (\widehat{y} - E_{\mathcal{D}}(\widehat{y}))^2 \right]$$ -->
  
  <!-- $$\text{Irreducible error}= E_{\mathcal{D}}[\varepsilon^2] = \sigma^2, \hspace{2cm} \text{Cross-terms}=0 \text{ (Why?)}$$  -->
  
  ---
  layout: false
class: inverse, middle, center

# Internal Validation of Predictive Algorithms

---
  layout: true
background-image: url(hex_stickers/PNG/tidymodels.png)
background-position: 95% 2.5%
  background-size: 13%

# Internal validation

---
  
  ### Modeling algorithms
  
  **Definition:** A collection of steps to develop a prediction rule $\widehat{f}(\cdot)$.

--
  
  **Examples:** 
  
  - `Data -> impute (EM) -> stepwise regression ->` $\widehat{f}_1(\cdot)$
  
  --
  
  - `Data -> impute (MICE) -> random forest ->` $\widehat{f}_2(\cdot)$
  
  --
  
  - `Data -> impute (mean) -> neural network ->` $\widehat{f}_2(\cdot)$
  
  --
  
  **Problem:** 
  
  - Choose an appropriate modeling algorithm 

- Develop prediction equations without overfitting the training data.

---
  
  ### Resampling
  
  **Definition:** iteratively and randomly drawing samples (with or without replacement) from a derivation dataset.

**Applications:** 
  
  *Statistical inference:* Bootstrap resampling can be used to estimate distributions of parameters with unknown distributions. 

- The difference between correlated $R^2$ statistics.

*Predictive analytics:* Estimate the error of a prediction rule when it is applied to new data.

- Cross-validation
- Bootstrap validation
- Monte-carlo validation (focus for today)

---
  
  ### Measuring prediction error
  
  **Continuous outcomes:** 
  
  - mean squared difference between predicted and observed values
- mean absolute difference between predicted and observed values

**Binary outcomes:**
  
  - Discrimination between cases and non-cases
- Absolute accuracy of predicted probability (calibration)
- Overall performance (Brier score)
- Net reclassification indices (If decisions are involved) 

**Survival outcomes:**
  
  - Time-dependent analogues of discrimination, calibration, Brier scores, and net reclassification.

---
  
  ### Step-by-step example
  
  Randomly select a set of training data (blue rows) from the derivation data.

```{r}

par(mar = c(1, 1, 1, 1)/10, mfrow = c(1, 1))

nsubs=8
trn_col="skyblue"
tst_col="grey70"

trn=sample(nsubs, round(nsubs/2))
ntrn=length(trn)
cols=rep(tst_col,nsubs)
cols[trn]=trn_col

plotmat(matrix(nrow = nsubs, ncol = nsubs, byrow = TRUE, data = 0), 
  pos = rep(1,nsubs), name = paste("Participant", 1:nsubs), 
  lwd = 1, box.lwd = 1, box.cex = 1.5, box.size = 0.3,
  box.type = "rect", box.prop = 0.075, shadow.size=0.002,
  box.col=cols)

```

---
  
  ### Step-by-step example
  
  Use the training data to develop one prediction rule per modeling algorithm. 

```{r}

par(mar = c(1, 1, 1, 1)/10, mfrow = c(1, 1))

mdl_col='green'
prd_col='violet'

yvals=seq(0.20,0.80,length.out=ntrn)


pos=map(yvals,~c(0.20,.))%>%
  reduce(rbind)%>%rbind(c(0.50,0.25),
    c(0.50,0.75),
    c(0.80,0.25),
    c(0.80,0.75))

M <- matrix(nrow = ntrn+4, 
  ncol = ntrn+4, 
  byrow = TRUE, data = 0)

M[ntrn+1:2,1:ntrn]=1
M[8,6]=1
M[7,5]=1

C=M
C[ntrn+1,1:ntrn]=0.07
C[ntrn+2,1:ntrn]=-0.07
C[8,6]=0
C[7,5]=0


plotmat(M, pos = pos, 
  box.col = c(rep(trn_col, ntrn),rep(mdl_col,2),rep(prd_col,2)),
  name = c(paste("Participant", sort(trn, decreasing = T)),
    paste('Modeling \nAlgorithm',2:1), 
    paste('Prediction \nRule',2:1)),
  lwd = 1, box.lwd = 1, box.cex = 1.5, 
  box.size = c(rep(0.1,ntrn),rep(0.1,2),rep(0.1,2)),
  box.type = c(rep("rect",ntrn),rep("ellipse",2),rep("square",2)), 
  box.prop = c(rep(0.3,ntrn),rep(1,2),rep(0.5,2)), 
  shadow.size=0.002,dtext=1000, curve=C)

```

---
  
  ### Step-by-step example
  
  Apply each prediction rule to the testing data, creating a set of predictions for each. 

```{r}

par(mar = c(1, 1, 1, 1), mfrow = c(1, 1))

tst=setdiff(1:nsubs,trn)
ntst=length(tst)

p1=map(yvals,~c(0.20,.))%>%reduce(rbind)
p2=c(0.50,0.50)
p3=map(yvals,~c(0.80,.))%>%reduce(rbind)

pos=rbind(p1,p2,p3)

M <- matrix(nrow = 2*ntst+1, 
  ncol = 2*ntst+1, 
  byrow = TRUE, data = 0)

M[ntst+1,1:ntst]=1; M[(ntst+2):nrow(M),ntst+1]=1; 

C=M; C[ntst+2, ntst+1]=0
C[ntst+1,1:ntst]=seq(-0.06, 0.06, length.out=ntst)
C[(ntst+2):nrow(C),ntst+1]=0

set.seed(329)

prds=round(runif(ntst),2)

outs=rbinom(ntst,prob = 0.5,size=1)

dtbl=data.frame(cbind(prds=rev(prds),outs=outs))%>%
  mutate(diff=prds-outs,sqdf=diff^2)


plotmat(M, pos = pos, 
  box.col = c(rep(tst_col, ntst),prd_col,
    rep(tst_col, ntst)),
  name = c(paste("Participant", sort(tst, decreasing = T)),
    'Prediction \nRule',
    paste("P(ASCVD) =", format(prds,nsmall=2))),
  lwd = 1, box.lwd = 1, box.cex = 1.5, 
  box.size = c(rep(0.11,ntrn),0.1),
  box.type = c(rep("rect",ntst),"square"), 
  box.prop = c(rep(0.3,ntst),1), 
  shadow.size=0.002, curve=C,dtext=1000)

```

---
  
  ### Step-by-step example
  
  Evaluate each set of predictions

```{r}

dtbl%>%
  set_names(
    c("P(ASCVD)","ASCVD",
      "Difference","Squared Difference")
  ) %>% 
  kable(
    align=c('c','c','c','c')
  ) %>%
  kable_styling(
    position = "center", 
    bootstrap_options = c("striped", "hover"),
    full_width = TRUE
  ) 


```

$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (\widehat{f}(x_i)-y_i)^2 = \frac{`r paste(format(round(dtbl$sqdf,2),nsmall=2),collapse=" + ")`}{`r ntst`} = `r format(round(mean(dtbl$sqdf),2),nsmall=2)`$$
  
  ---
  
  ### Scaling the Brier score
  
  - Suppose the prevalence of ASCVD is 10%. 

- What would you guess an individual's risk is if you don't know anything about them?
  
  --
  
  - 10%. 

- The Brier score that results from this general prediction is a 'reference' Brier score. 

- The Brier score of a predictive model can be scaled using the reference Brier score: $$1 - \frac{\text{Model Brier Score}}{\text{Reference Brier Score}}$$ 
  
  ---
  layout: false
class: middle, center

```{r, fig.height=7, fig.width=9, fig.align='center'}

par(mar = c(1, 1, 1, 1)/10, mfrow = c(1, 1))

trn_col="burlywood"
tst_col="antiquewhite"

#xvals=c(0.125,0.325,0.725)
xvals=c(0.12,0.78)
yvals = c(0.75, 0.55, 0.35, 0.10)

step1=paste(
  'Randomly',
  'split the',
  'derivation',
  'data into',
  'a training', 
  'set and a',
  'test set',
  sep=' \n')

step2 = paste( 
  'Apply each candidate modeling',
  'algorithm\u2020 to the training',
  'set, separately, to develop one',
  'predictive equation for each.',
  sep=' \n')

step3 = paste(
  'Apply each predictive equation',
  'from Step 2, separately, to the',
  'test set to create one set of',
  'predictions for each algorithm',
  'that can be evaluated in Step 4.',
  sep=' \n')

step4 = paste( 
  'Compute and record the',
  'calibration, discrimination,',
  'and Brier score for each',
  'set of predictions from',
  'Step 3, separately.',
  sep=' \n')

step5 = paste( 
  'Repeat Steps 1-4',
  '100+ times to',
  'internally validate', 
  'each algorithm.',
  sep=' \n')

stepf = paste( 
  'Based on internally validated calibration, discrimination, and Brier score estimates,',
  'select one algorithm to develop predictive equations using the full set of derivation data.',
  sep=' \n')


rx=list(super=1/3, big=1/6, small=1/15)
ry=list(super=.15, big=.08, small=0.03)

bump_coef <- 6/5
txt_size=1.1

xmid = mean(
  c(
    xvals[1]+rx$big,
    xvals[2]-rx$super
  )
)

openplotmat()


# Arrows ------------------------------------------------------------------

straightarrow(
  from = c(xvals[1]+rx$big,yvals[1]+0.45*ry$super*bump_coef),
  to = c(xvals[2],yvals[1]+0.45*ry$super*bump_coef),
  arr.pos=0.43
)

straightarrow(
  from = c(xvals[2],yvals[1]+2.2*ry$small),
  to = c(xvals[2],yvals[2]+0.13),
  arr.pos = 1
)

straightarrow(
  from = c(xvals[2],yvals[2]),
  to = c(xvals[2],yvals[3]-2.2*ry$small+0.13),
  arr.pos = 1
)

bentarrow(
  from = c(xmid,yvals[1]-0.45*ry$super*bump_coef),
  to = c(xvals[2],yvals[2]),
  path = 'V',
  arr.pos=0.41
)

treearrow(
  from = c(xvals[1]+rx$small,yvals[1]),
  to = c(xmid-rx$small,yvals[1]+0.45*ry$super*bump_coef),
  path = 'V',
  arr.pos=0.97
)

treearrow(
  from = c(xvals[1]+rx$small,yvals[1]),
  to = c(xmid-rx$small,yvals[1]-0.45*ry$super*bump_coef),
  path = 'V',
  arr.pos=0.97
)

straightarrow(
  from = c(xvals[2]-rx$big,yvals[3]-2.2*ry$small),
  to = c(xvals[1]+rx$big*0.65, yvals[3]-2.2*ry$small),
  arr.pos = 1
)

straightarrow(
  from = c(xvals[1],yvals[3]+ry$small),
  to = c(xvals[1],yvals[1]-0.20),
  arr.pos = 1
)

straightarrow(
  from = c(xvals[1],yvals[3]-5*ry$small),
  to = c(xvals[1],yvals[4]+ry$small),
  arr.pos = 1
)


# Step 1 ------------------------------------------------------------------

textempty(
  mid = c(xvals[1],yvals[1]+0.15), 
  lab = 'STEP 1', 
  font = 2, 
  cex = txt_size)

textempty(
  mid = c(xvals[1],yvals[1]-0.15), 
  lab = 'Derivation \ndata', 
  font = 2, 
  cex = txt_size)

textrect(
  mid = c(xvals[1],yvals[1]), 
  radx = rx$small*1.20, 
  rady = 0.80*ry$super, 
  lab = step1, 
  font = 1,
  cex = txt_size,
  shadow.size = 1e-10,
  box.col='white')

textrect(
  mid = c(xmid,yvals[1]+0.45*ry$super*bump_coef), 
  radx = rx$small, 
  rady = 0.90*ry$super/2, 
  lab = "Training \nset", 
  font = 2,
  cex = txt_size,
  shadow.size = 1e-10,
  box.col=trn_col)

textrect(
  mid = c(xmid,yvals[1]-0.45*ry$super*bump_coef), 
  radx = rx$small, 
  rady = 0.90*ry$super/2, 
  lab = "Test \nset", 
  font = 2,
  cex = txt_size,
  shadow.size = 1e-10,
  box.col=tst_col)


# Step 2 ------------------------------------------------------------------

textempty(
  mid = c(xvals[2],yvals[1]+2.2*ry$small+0.10), 
  lab = 'STEP 2', 
  font = 2, 
  cex = txt_size)

textrect(
  mid = c(xvals[2],yvals[1]+2.2*ry$small), 
  radx = rx$big, 
  rady = ry$big, 
  lab = step2, 
  cex = txt_size,
  shadow.size = 1e-10,
  box.col=trn_col)


# Step 3 ------------------------------------------------------------------

textempty(
  mid = c(xvals[2],yvals[2]+0.10), 
  lab = 'STEP 3', 
  font = 2, 
  cex = txt_size)

textrect(
  mid = c(xvals[2],yvals[2]), 
  radx = rx$big, 
  rady = ry$big, 
  lab = step3, 
  cex = txt_size,
  shadow.size = 1e-10,
  box.col=tst_col)

# Step 4 ------------------------------------------------------------------

textempty(
  mid = c(xvals[2],yvals[3]-2.2*ry$small+0.10), 
  lab = 'STEP 4', 
  font = 2, 
  cex = txt_size)

textrect(
  mid = c(xvals[2],yvals[3]-2.2*ry$small), 
  radx = rx$big, 
  rady = ry$big, 
  lab = step4, 
  cex = txt_size,
  shadow.size = 1e-10,
  box.col='mistyrose')

# Step 5 ------------------------------------------------------------------

textempty(
  mid = c(xvals[1],yvals[3]), 
  lab = 'STEP 5', 
  font = 2, 
  cex = txt_size)

textempty(
  mid = c(xvals[1],yvals[3]-0.025*3), 
  lab = step5, 
  cex = txt_size)

# Final step --------------------------------------------------------------

textempty(
  mid = c(xvals[1],yvals[4]), 
  lab = 'STEP 6', 
  font = 2, 
  cex = txt_size)

textempty(
  mid = c(xvals[1]+0.05, yvals[4]+0.025), 
  lab = stepf, 
  adj = c(0,1),
  cex = txt_size)

```

---
  
  class: inverse, center, middle

# Application
## The Jackson Heart Study (JHS)

---
  layout: true
background-image: url(Intro_to_Predictive_Analytics_files/JHS_logo_2018.jpg)
background-position: 92.5% 2.5%
  background-size: 13%

### ASCVD risk prediction in the JHS

---
  
  **Atherosclerotic cardiovascular disease (ASCVD) events in the JHS**
  
  A total of 5,126 JHS participants consented to provide follow up data after their baseline assessment. During a follow up period of up to 14 years,

- ASCVD (stroke or coronary heart disease) events were identified and adjudicated. 

- participants who were lost to follow up or died were censored at the time of last contact or death, respectively. 

--
  
  **Analysis plan**
  
  1. Compare 9 candidate modeling algorithms using internal validation.

2. Select one modeling algorithm to develop a risk prediction equation based on time-varying discrimination and Brier score.  

3. Communicate our findings to a general audience.

---
  
  **1. Candidate modeling algorithms:**
  
  *Decision tree ensembles*
  
  - Oblique random survival forest (ORSF)
- Oblique random survival forest with internal cross-validation (ORSF-CV)
- Random survival forest (RSF)
- Conditional inference forest (CIF)
- Gradient boosted decision trees (GBT)

--
  
  *Statistical models*
  
  - Lasso penalized proportional hazards regression (Lasso)
- Ridge penalized proportional hazards regression (Ridge)
- Gradient boosted proportional hazards regression (GBM)
- *Stepwise proportional hazards regression* (Step PH)

--
  
  .footnote[
    [1] Missing values were imputed in the training dataset and the testing dataset, separately.
    
    [2] The missForest algorithm was applied to impute missing values for each modeling algorithm. 
    ]

```{r, echo=FALSE, include=FALSE}

library(feather)

file.path(
  "Intro_to_Predictive_Analytics_files",
  "jhs_bmark.feather"
) %>% 
  read_feather() ->
  jhs_benchmark_data

xlab='Time since baseline exam, years'
ylab='Concordance index'
main='Discrimination of the ORSF-CV'

jhs_benchmark_plot <- 
  jhs_benchmark_data %>% 
  dplyr::filter(
    mdl!='Reference',
    measure=='cstat'
  ) %>% 
  droplevels() %>% 
  dplyr::mutate(
    mdl=fct_recode(
      mdl,
      'gbt'='xgbst',
      'gbm'='bst'
    ),
    mdl=fct_reorder(
      mdl, .x=1-err
    )
  ) %>% 
  dplyr::rename(Model=mdl) %>% 
  group_by(Model, seed) %>%
  filter(time==max(time)) %>% 
  dplyr::select(-time,-measure) %>% 
  ggplot(
    aes(
      x = 1-err, 
      y = reorder(Model, 1-err), 
      height = ..density..,
      fill=Model
    )
  ) +
  ggridges::geom_density_ridges(
    bandwidth=0.007
  )+
  coord_cartesian(xlim=c(0.75,0.9))+
  labs(
    x='\nConcordance index at 5 years',
    y='',
    fill=''
  )+
  theme(
    legend.position = '',
    panel.grid.major.y = element_line(
      color='black',
      linetype=1
    ),
    text = element_text(
      size=25
    ),
    panel.border = element_blank(),
    axis.ticks = element_blank()
  )+
  scale_fill_viridis_d()


jhs_benchmark_plot_orsf_cv <- 
  jhs_benchmark_data %>%
  dplyr::filter(
    mdl == 'orsf.cv', 
    measure == 'cstat'
  ) %>%
  ggplot(
    aes(x = time, y = 1 - err)
  ) +
  labs(
    x=xlab, y=ylab, title=main
  ) + 
  geom_line(
    aes(group = seed), col = 'grey50', alpha = 0.05
  ) +
  geom_smooth(
    method = 'gam',formula = y~s(x)
  )+
  coord_cartesian(
    ylim = c(0.70, 1.00)
  ) +
  scale_x_continuous(breaks=0:5)

```

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    jhs_benchmark_data %>% #<<
      filter( #<<
        mdl=='orsf.cv', #<<
        measure=='cstat' #<<
      ) %>% #<<
      ggplot( 
        aes(x=time, y=1-err) 
      ) + 
      coord_cartesian(ylim=c(.7,1)) + 
      scale_x_continuous(breaks=0:5) + 
      labs( 
        x=xlab, y=ylab, title=main 
      ) + 
      geom_line( 
        aes(group=seed), 
        col='grey50', alpha=0.05 
      ) + 
      geom_smooth( 
        method='gam', formula=y~s(x) 
      )
    
    ```
    
    ]

.pull-right[
  ```{r, fig.height=8, fig.width=6}
  
  jhs_benchmark_data %>%
    filter(
      mdl=='orsf.cv',
      measure=='cstat'
    ) 
  
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    jhs_benchmark_data %>%
      filter(
        mdl=='orsf.cv',
        measure=='cstat'
      ) %>%
      ggplot( #<<
        aes(x=time, y=1-err) #<<
      ) + #<<
      coord_cartesian(ylim=c(.7,1)) + 
      scale_x_continuous(breaks=0:5) + 
      labs( 
        x=xlab, y=ylab, title=main 
      ) + 
      geom_line( 
        aes(group=seed), 
        col='grey50', alpha=0.05 
      ) + 
      geom_smooth( 
        method='gam', formula=y~s(x) 
      )
    
    ```
    
    ]

.pull-right[
  ```{r, fig.height=8, fig.width=6}
  
  jhs_benchmark_data %>%
    filter(
      mdl=='orsf.cv',
      measure=='cstat'
    ) %>%
    ggplot(
      aes(x=time, y=1-err)
    ) 
  
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    jhs_benchmark_data %>%
      filter(
        mdl=='orsf.cv',
        measure=='cstat'
      ) %>%
      ggplot(
        aes(x=time, y=1-err)
      ) +
      coord_cartesian(ylim=c(.7,1)) + #<<
      scale_x_continuous(breaks=0:5) + #<<
      labs( 
        x=xlab, y=ylab, title=main 
      ) + 
      geom_line( 
        aes(group=seed), 
        col='grey50', alpha=0.05 
      ) + 
      geom_smooth( 
        method='gam', formula=y~s(x) 
      )
    
    ```
    
    ]

.pull-right[
  ```{r, fig.height=8, fig.width=6}
  
  jhs_benchmark_data %>%
    filter(
      mdl=='orsf.cv',
      measure=='cstat'
    ) %>%
    ggplot(
      aes(x=time, y=1-err)
    ) +
    coord_cartesian(ylim=c(.7,1)) + 
    scale_x_continuous(breaks=0:5)
  
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    jhs_benchmark_data %>%
      filter(
        mdl=='orsf.cv',
        measure=='cstat'
      ) %>%
      ggplot(
        aes(x=time, y=1-err)
      ) +
      coord_cartesian(ylim=c(.7,1)) + 
      scale_x_continuous(breaks=0:5) + 
      labs( #<<
        x=xlab, y=ylab, title=main #<<
      ) + #<<
      geom_line( 
        aes(group=seed), 
        col='grey50', alpha=0.05 
      ) + 
      geom_smooth( 
        method='gam', formula=y~s(x) 
      )
    
    ```
    
    ]

.pull-right[
  ```{r, fig.height=8, fig.width=6}
  
  jhs_benchmark_data %>%
    filter(
      mdl=='orsf.cv',
      measure=='cstat'
    ) %>%
    ggplot(
      aes(x=time, y=1-err)
    ) +
    coord_cartesian(ylim=c(.7,1)) + 
    scale_x_continuous(breaks=0:5) + 
    labs(
      x=xlab, y=ylab, title=main
    ) 
  
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    jhs_benchmark_data %>%
      filter(
        mdl=='orsf.cv',
        measure=='cstat'
      ) %>%
      ggplot(
        aes(x=time, y=1-err)
      ) +
      coord_cartesian(ylim=c(.7,1)) + 
      scale_x_continuous(breaks=0:5) + 
      labs(
        x=xlab, y=ylab, title=main
      ) + 
      geom_line( #<<
        aes(group=seed), #<<
        col='grey50', alpha=0.05 #<<
      ) + #<<
      geom_smooth( 
        method='gam', formula=y~s(x) 
      )
    
    ```
    
    ]

.pull-right[
  ```{r, fig.height=8, fig.width=6}
  
  jhs_benchmark_data %>%
    filter(
      mdl=='orsf.cv',
      measure=='cstat'
    ) %>%
    ggplot(
      aes(x=time, y=1-err)
    ) +
    coord_cartesian(ylim=c(.7,1)) + 
    scale_x_continuous(breaks=0:5) + 
    labs(
      x=xlab, y=ylab, title=main
    ) + 
    geom_line( #<<
      aes(group=seed), #<<
      col='grey50', alpha=0.05 #<<
    )
  
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE, eval=FALSE}
    
    jhs_benchmark_data %>%
      filter(
        mdl=='orsf.cv', 
        measure=='cstat'
      ) %>%
      ggplot(
        aes(x=time, y=1-err)
      ) +
      coord_cartesian(ylim=c(.7,1)) + 
      scale_x_continuous(breaks=0:5) + 
      labs(
        x=xlab, y=ylab, title=main
      ) + 
      geom_line(
        aes(group=seed), 
        col='grey50', alpha=0.05
      ) +
      geom_smooth( #<<
        method='gam', formula=y~s(x) #<<
      ) #<<
    
    ```
    
    ]

.pull-right[
  ```{r, fig.height=8, fig.width=6}
  
  print(jhs_benchmark_plot_orsf_cv)
  
  ```
  ]

---
  
  ```{r, fig.width=12, fig.height=8}

jhs_benchmark_plot

```

---
  
  ```{r, fig.width=12, fig.height=8}

jhs_benchmark_data %>% 
  dplyr::filter(
    measure=='brier_score'
  ) %>% 
  group_by(mdl,seed) %>% 
  dplyr::summarise(
    err=mean(err,na.rm=TRUE)
  ) %>% 
  ungroup() %>% 
  tidyr::spread(mdl,err) %>% 
  dplyr::select(
    Reference,everything(),-seed
  ) %>%
  mutate_at(
    -1, .funs=list(~1-(./Reference))
  ) %>% 
  tidyr::gather(mdl,err) %>% 
  dplyr::filter(mdl!='Reference') %>% 
  droplevels() %>% 
  dplyr::mutate(
    mdl=fct_recode(
      mdl,
      'gbt'='xgbst',
      'gbm'='bst'
    ),
    mdl=fct_reorder(
      mdl, .x=1-err
    )
  ) %>% 
  dplyr::rename(Model=mdl) %>% 
  ggplot(
    aes(
      x = err, 
      y = reorder(
        Model,
        err,
        FUN=function(x){
          mean(x,na.rm=TRUE)
        }
      ), 
      height = ..density..,
      fill = Model
    )
  ) +
  ggridges::geom_density_ridges(
    bandwidth=0.005,na.rm=TRUE
  )+
  labs(
    x='\nIntegrated scaled Brier score at 5 years',
    y='',
    fill=''
  )+
  coord_cartesian(
    xlim=c(0.00,0.15)
  )+
  theme(
    legend.position = '',
    panel.grid.major.y = element_line(
      color='black',
      linetype=1
    ),
    text = element_text(
      size=25
    ),
    panel.border = element_blank(),
    axis.ticks = element_blank()
  )+
  scale_fill_viridis_d() 


```

---
  layout: false
background-image: url(Intro_to_Predictive_Analytics_files/baby_whoah_now.gif)
background-position: 50% 60%
  background-size: 70%

# In summary...


---
  class: inverse, center, middle
# Communication

---
  layout: true
background-image: url(hex_stickers/PNG/ggplot2.png)
background-position: 95% 2.5%
  background-size: 13%

---
  
  # Communication
  
  **Why is this important?**
  
  - build trust (relationship)

- provide insight (what's working, what's not)

- supports understanding (why is my risk high?)

- supports accountability (when predictions are wrong, you need to know why)

--
  
  **What can we do?**
  
  - Variable dependence (analogous to descriptive statistics)

- Partial variable dependence (analogous to adjusted estimates)

- Variable importance (analogous to p-values)

- Shapley values (brand new [but also old] - they can do all three!)

---
  
  # Variable dependence
  
  Suppose

- $\hat{f}(x)$ is a prediction rule for ASCVD risk 

- $x$ is a set of 100 predictor variables, including 

- left ventricular mass in $g/m^2$
  - age in years
- systolic blood pressure in mm Hg
- estimated glomerular filtration rate (eGFR) in ml/min/1.73 $m^2$
  
  - We want to know how each variable is related to ASCVD risk. 

What should we do?
  
  --
  
  - Plot $\hat{f}(x)$ as a function of each variable, separately.

---
  layout:false

```{r, fig.width=16, fig.height=12}

readRDS("Intro_to_Predictive_Analytics_files/vdep_plot.RDS")

```

---
  layout:true
background-image: url(hex_stickers/PNG/purrr.png)
background-position: 95% 2.5%
  background-size: 13%
# Partial dependence

---
  
  **Example:** 'integrate' over the effect of age to measure the age-adjusted effect of systolic blood pressure. 

.pull-left[
  ```{r, echo=TRUE, eval=TRUE}
  
  set.seed(32987)
  sbp_grid <- seq(120,180, by=10)
  
  example_data <- tibble(
    age=rnorm(n=1000, mean=60, sd=10)
  ) %>% 
    mutate(
      sbp=age+rnorm(1000,80,10),
      risk=rescale(
        age + rnorm(1000),
        to = c(0.01,0.99)
      )
    )
  
  mdl = lm(
    risk ~ age + sbp,
    data = example_data
  )
  
  ```
  
  ]

.pull-right[
  ```{r}
  
  print(sbp_grid)
  
  print(example_data[1:10,])
  
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    i = 1
    
    example_data<-example_data %>% 
      mutate(
        sbp=sbp_grid[i]
      ) %>% 
      mutate(
        yhat=predict(mdl,newdata=.)
      )
    
    partial_prediction <- 
      mean(example_data$yhat)
    
    ```
    ]

.pull-right[
  ```{r}
  print(example_data)
  print(partial_prediction)
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    i = 2
    
    example_data<-example_data %>% 
      mutate(
        sbp=sbp_grid[i]
      ) %>% 
      mutate(
        yhat=predict(mdl,newdata=.)
      )
    
    partial_prediction <- 
      mean(example_data$yhat)
    
    ```
    ]

.pull-right[
  ```{r}
  print(example_data)
  print(partial_prediction)
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    i = 3
    
    example_data<-example_data %>% 
      mutate(
        sbp=sbp_grid[i]
      ) %>% 
      mutate(
        yhat=predict(mdl,newdata=.)
      )
    
    partial_prediction <- 
      mean(example_data$yhat)
    
    ```
    ]

.pull-right[
  ```{r}
  print(example_data)
  print(partial_prediction)
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    partial_depvals <- 
      sbp_grid %>% 
      map_dbl(
        .f=function(sbp_val){
          example_data %>% 
            mutate(
              sbp=sbp_val
            ) %>% 
            mutate(
              yhat=predict(
                mdl,
                newdata=.
              )
            ) %>% 
            pluck('yhat') %>% 
            mean()
        }
      )
    
    ```
    ]

.pull-right[
  ```{r, fig.width=5, fig.height=5}
  
  round(partial_depvals[1:3],3)
  
  tibble(
    yhat=partial_depvals,
    xval=sbp_grid
  ) %>% 
    ggplot(aes(x=xval,y=yhat))+
    geom_point(size=2)+
    labs(
      x='Systolic blood pressure, mm Hg',
      y='Partial effect on risk'
    ) +
    scale_y_continuous(limits=c(0,1))
  
  ```
  ]

---
  
  .pull-left[
    ```{r, echo=TRUE}
    
    partial_depvals <- 
      seq(45,85,by=5) %>% 
      map_dbl(
        .f=function(age_val){
          example_data %>% 
            mutate(
              age=age_val
            ) %>% 
            mutate(
              yhat=predict(
                mdl,
                newdata=.
              )
            ) %>% 
            pluck('yhat') %>% 
            mean()
        }
      )
    
    ```
    ]

.pull-right[
  ```{r, fig.width=5, fig.height=5}
  
  tibble(
    yhat=partial_depvals,
    xval=seq(45,85,by=5)
  ) %>% 
    ggplot(aes(x=xval,y=yhat))+
    geom_point(size=2)+
    labs(
      x='Age, years',
      y='Partial effect on risk'
    ) +
    scale_y_continuous(limits=c(0,1))
  
  ```
  ]

---
  layout:false

```{r, fig.width=16, fig.height=12}

readRDS("Intro_to_Predictive_Analytics_files/pdep_plot.RDS")

```

---
  layout:true
background-image: url(hex_stickers/PNG/ggplot2.png)
background-position: 95% 2.5%
  background-size: 13%
# Variable importance

---
  
  **Step 1:** compute the accuracy of $\hat{f}(x)$ in a testing set.

.pull-left[
  ```{r, echo=TRUE}
  
  set.seed(730)
  test_data <- tibble(
    age=rnorm(n=10000, mean=60, sd=10)
  ) %>% 
    mutate(
      sbp=age+rnorm(10000,80,10),
      risk=rescale(
        age + rnorm(10000),
        to = c(0.01,0.99)
      )
    )
  
  original_rmse <- mdl %>% 
    predict(newdata=test_data) %>% 
    subtract(test_data$risk) %>% 
    raise_to_power(2) %>% 
    mean() %>% 
    raise_to_power(1/2)
  
  ```
  ]

.pull-right[
  ```{r}
  
  original_rmse
  
  permute_row <- function(x){
    new_x = sample(x, length(x), replace=F)
    new_x
  }
  
  ```
  ]

---
  
  **Step 2-a:** shuffle the systolic blood pressure values in the testing data

.pull-left[
  ```{r, echo=TRUE}
  
  test_shuffle <- test_data %>% 
    mutate(
      sbp = permute_row(sbp)
    )
  
  sbp_rmse <- mdl %>% 
    predict(newdata=test_shuffle) %>% 
    subtract(test_shuffle$risk) %>% 
    raise_to_power(2) %>% 
    mean() %>% 
    raise_to_power(1/2)
  
  ```
  ]

.pull-right[
  
  Importance of systolic blood pressure:
    
    ```{r, echo=TRUE}
  
  abs(original_rmse-sbp_rmse)
  
  ```
  ]

---
  
  **Step 2-b:** shuffle the age values in the testing data

.pull-left[
  ```{r, echo=TRUE}
  
  test_shuffle <- test_data %>% 
    mutate(
      age = permute_row(age)
    )
  
  age_rmse <- mdl %>% 
    predict(newdata=test_shuffle) %>% 
    subtract(test_shuffle$risk) %>% 
    raise_to_power(2) %>% 
    mean() %>% 
    raise_to_power(1/2)
  
  ```
  ]

.pull-right[
  
  Importance of age:
    
    ```{r, echo=TRUE}
  
  abs(original_rmse-age_rmse)
  
  ```
  ]