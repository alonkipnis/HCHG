# analyze survival data using R code

###########################################################################
#                  Applying the presented methods in R                    #
###########################################################################

##################################
# Load data
##################################

# #install a required package (only needed the first time)
#install.packages("coxphw")
# Load the package (always)
library(coxphw)

# Load the data from the package
data("gastric")
# Some tests require groups coded as 0 and 1:
gastric$group[gastric$group == 2] <-0 

# Significance level
alpha <- 0.05


##################################
# Log-rank test
##################################
# Load required package
#install.packages("survival")
library(survival)

# Execute test
logRank = survdiff(Surv(time, status)~ group, rho = 0, 
                   #rho indicates classical log-rank test
                   data = gastric)$chisq
1 - pchisq(logRank,1)

##################################
# Peto-Peto test
##################################
# Load required package
#install.packages("survival")
library(survival)

# Execute test
PetoPeto = survdiff(Surv(time, status)~ group, rho = 1, 
                    # Peto & Peto modification
                    data = gastric)$chisq
1 - pchisq(PetoPeto,1)


##################################
# Gorfine et al. test
##################################
# Load required package
#install.packages("KONPsurv")
library(KONPsurv)   

# Define parameters
iterations <- 999

# Execute test
KONP = konp_test(gastric$time, gastric$status, gastric$group, 
                 # as input wee need time, censoring indicator and group
                 n_perm = iterations) 
                 # and number of iterations for permutation
# The two implemented test statistics
KONP$pv_chisq
KONP$pv_lr

##################################
# Ditzhaus and Friedrich test
##################################
# Load required package
#install.packages("mdir.logrank")
library(mdir.logrank) 

# Define parameters
iterations <- 999

# Prepare data
gastric_mdir <- gastric
colnames(gastric_mdir) <- c("time", "event", "group") 
# colnames as required for D&F

# Execute test
mdir.logrank(gastric_mdir, cross =TRUE, #include crossing alternative 
                           nperm = iterations)$p_value$Perm 




##################################
# Tian et al. test
##################################
# Load required package
#install.packages("surv2sampleComp") 
library(surv2sampleComp)   

# Define parameters
iterations <- 999
tau <- 2900

# Execute test
RMST1 = surv2sample(gastric$time, gastric$status, gastric$group, 
                    npert = iterations,
                    timepoints = tau, 
                    # when to calculate difference and ratio
                    tau = tau, # restricted timepoint
                    conf.int = 1-alpha)$integrated_surv.diff
RMST1[1,4] # p-value

##################################
# Uno et al. test
##################################
# Load required package
#install.packages("survRM2")
library(survRM2)


# Define parameters
iterations <- 999
tau <- 2900

# Execute test
RMST2 = rmst2(gastric$time, gastric$status, gastric$group, 
              tau, # restricted timepoint
              covariates = NULL, alpha)
RMST2$unadjusted.result[1,4] # p-value



##################################
# Sheng et al. test
##################################
# Load required package
#install.packages("TSHRC") 
library(TSHRC) 

# Define parameters
iterations <- 999

# Execute test
twostage(gastric$time, gastric$status, gastric$group, 
                   nboot=iterations)[3]

##################################
# Plot
##################################

install.packages("survminer")
library(survminer)

# Create a survival curve
fit <- survfit(Surv(gastric$time, gastric$status)~gastric$group, 
               # create survival object from time and status with 
               # arms according to group
               data = gastric, type = "kaplan-meier") 

# Visualize with survminer
ggsurvplot(fit, data = gastric, risk.table = TRUE, 
                                # we do want a risk table
           break.time.by = 500, # where to break the x axis
           break.y.by = 0.1, # where to break the y axis
           xlim = c(0,3000), ylim = c(0,1), xlab = "\n Months"
           , ylab = "\n", 
           legend.labs = c("chemotherapy alone", 
                           "chemotherapy plus radiation"), 
           palette = c("#0073b5", "#e39144"), # Define colors
           legend = c(0.7, 0.7), # position of the legend
           legend.title="", fontsize = 5, # fontsize for the table
           font.x = c(14, "bold"), font.caption = c(14, "black"), 
           font.legend = c(14, "black"), font.tickslab = c(14, "black"),
           font.table = c(14, "black"),
           tables.theme = theme_cleantable(), 
                    # Minimalistic style of the table
           tables.height = 0.2, # change hight of the table
           risk.table.y.text = FALSE, # no group names in the table
           tables.y.text = FALSE)
