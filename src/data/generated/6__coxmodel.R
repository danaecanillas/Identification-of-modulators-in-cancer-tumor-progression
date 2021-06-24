library("survival")
library("survminer")

df <- read.csv("clean_train.csv", header = TRUE)

#df <- df[df$PAM50 == 'LumB',]
df = subset(df, select = -c(submitter,PAM50,DSSE10,DSS10,HT,RT,CT,INTCLUST))

for(i in 1:ncol(df)) {      
  if (!(colnames(df)[i] %in% c("RFS","RFSE","grade","stage","PIK3CA.mut","TP53.mut","lymph_nodes_positive","age_at_diagnosis"))) {
    print(colnames(df)[i])
    df[ , i] <- cut(df[ , i], breaks = c(-Inf,mean(df[ , i]),Inf), labels = c("Low","High"))
    res.cox <- coxph(Surv(RFS, RFSE) ~ df[ , i], data =  df)
    print(res.cox)
    print("------------------------------------------------------------------------")
  }
  else if (colnames(df)[i] == "age_at_diagnosis") {
    print(colnames(df)[i])
    df[ , i] <- cut(df[ , i], breaks = c(-Inf,45,70,Inf), labels = c("Young","Adult","Elderly"))
    res.cox <- coxph(Surv(RFS, RFSE) ~ df[ , i], data =  df)
    print(res.cox)
  }
  else if (colnames(df)[i] == "lymph_nodes_positive") {
    print(colnames(df)[i])
    df[ , i] <- cut(df[ , i], breaks = c(-Inf,10,Inf), labels = c("Low","High"))
    res.cox <- coxph(Surv(RFS, RFSE) ~ df[ , i], data =  df)
    print(res.cox)
  }
}

res.cox <- coxph(Surv(RFS, RFSE) ~ lymph_nodes_positive*NPI*stage*PROLIF*ERBB2*age_at_diagnosis, data =  df)
print(res.cox)
