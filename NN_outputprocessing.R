library(ggplot2)
library(reshape2)
df_cor = read.csv('corr_df.csv')
df_cor = df_cor[,-c(1,3)]
df_shap = read.csv('shap_dist.csv')
df_shap = df_shap[,-c(1,3)]

df_comb = df_shap*df_cor

df <- melt(df_comb)

newdf = aggregate(df$value, by = list(df$variable) ,mean)
colnames(newdf) <- c('variable', 'value')

newdf$lower = NA
newdf$upper = NA
for (i in 1:nrow(newdf)) {
  newdf$value[i] = mean(df$value[df$variable == newdf$variable[i]], na.rm = T)
  newdf$lower[i] = HDInterval::hdi(df$value[df$variable == newdf$variable[i]], credMass = 0.89)[1]
  newdf$upper[i] = HDInterval::hdi(df$value[df$variable == newdf$variable[i]], credMass = 0.89)[2]
  
}
raw_shap_df = newdf

a <- ggplot() + 
  geom_linerange(newdf,mapping= aes(x=reorder(variable,value),ymin=lower, ymax=upper), lwd = 0.8,color='blue', alpha = ifelse(newdf$lower<0 & newdf$upper>0,0.4,1)) +
  geom_point(newdf,mapping= aes(x=reorder(variable,value), y=value), size = 1.5, stroke=1, shape = 21, fill ='white',color='black') +
  geom_hline(yintercept=0, linetype="dashed", color = "red")+
  ylab('SHAP value x corr.coef')+ xlab('feature')+theme_classic()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


pdf(file="shaps.pdf", height = 6, width = 10)
plot(a)
dev.off()

#######
df_cor = read.csv('corr_df.csv')
df_cor = df_cor[,-c(1,3)]
df_shap = read.csv('weighted_shap_dist.csv')
df_shap = df_shap[,-c(1,3)]

df_comb = df_shap*df_cor

df <- melt(df_comb)

newdf = aggregate(df$value, by = list(df$variable) ,mean)
colnames(newdf) <- c('variable', 'value')

newdf$lower = NA
newdf$upper = NA
for (i in 1:nrow(newdf)) {
  newdf$value[i] = mean(df$value[df$variable == newdf$variable[i]], na.rm = T)
  newdf$lower[i] = HDInterval::hdi(df$value[df$variable == newdf$variable[i]], credMass = 0.89)[1]
  newdf$upper[i] = HDInterval::hdi(df$value[df$variable == newdf$variable[i]], credMass = 0.89)[2]
  
}

weighted_shap_df = newdf
a <- ggplot() + geom_point()+ 
  geom_linerange(newdf,mapping= aes(x=reorder(variable,value),ymin=lower, ymax=upper), lwd = 0.8,color='blue', alpha = ifelse(newdf$lower<0 & newdf$upper>0,0.4,1)) +
  geom_point(newdf,mapping= aes(x=reorder(variable,value), y=value), size = 1.5, stroke=1, shape = 21, fill ='white',color='black') +
  geom_hline(yintercept=0, linetype="dashed", color = "red")+
  ylab('Weighted SHAP value x corr.coef')+ xlab('feature')+theme_classic()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


pdf(file="weightedshaps.pdf", height = 6, width = 10)
plot(a)
dev.off()

####

preds = read.csv('latentpreds.csv', header = F)
colnames(preds) <- c('latents', 'id', 'true')
newpreds = aggregate(latents ~ id, preds, mean)
preds2 = aggregate(true ~ id, preds, mean)
newpreds$true = preds2$true
newpreds$delta = abs(scale(newpreds$latents, center = F)-scale(newpreds$true, center = F))
newpreds$latpertrue =  newpreds$latents/newpreds$true
newpreds = newpreds[order(newpreds$true, decreasing = T),]
newpreds$ranktrue = 1:nrow(newpreds)
newpreds = newpreds[order(newpreds$latent, decreasing = T),]
newpreds$ranklatent = 1:nrow(newpreds)
newpreds$rankd = abs(newpreds$ranktrue-newpreds$ranklatent)
newpreds$latent_rel = NA
newpreds$latent_rel[2:nrow(newpreds)] = newpreds$latents[2:nrow(newpreds)] / newpreds$latents[1:(nrow(newpreds)-1)]
newpreds$true_rel[2:nrow(newpreds)] = newpreds$true[2:nrow(newpreds)] / newpreds$true[1:(nrow(newpreds)-1)]
newpreds$rel_d = newpreds$true_rel-newpreds$latent_rel

hist(scale(newpreds$latents, center = F), breaks = 100)

df = read.csv('phoibledat.csv')
df = df[,-4]

df$latents = newpreds$latents[order(newpreds$id)]

pdf(file="corrplot.pdf", height = 10, width = 10)
corrplot::corrplot(corr = cor(df[,c(3:ncol(df))]), cl.pos="b", type = 'lower', method = 'circle', tl.col="black", diag = F) 
dev.off()

corrdist = data.frame(feature = colnames(df[,3:38]))
corrdist$cumdist = NA
counter = 1
for (i in c(3:38)){
  corrdist$cumdist[counter] = sum(cor(df[,c(3:ncol(df))], df[,i]))-1
  counter = counter + 1
}
corrdist$latents = cor(df[,c(3:38)], df[,ncol(df)])

corrdist$cumdist_s = scale(corrdist$cumdist, center = F)
corrdist$latents_s = scale(corrdist$latents, center = F)
corrdist$cumdist_n = corrdist$cumdist/max(corrdist$cumdist)
corrdist$latents_n = corrdist$latents/max(corrdist$latents)

corrdist$d = corrdist$latents_n-corrdist$cumdist_n

corrdist$rank_cumdist[order(corrdist$cumdist_n)] = 1:nrow(corrdist)
corrdist$rank_latents[order(corrdist$latents_n)] = 1:nrow(corrdist)

df[c(1004, 2543),]
df[c(1220, 2671),]

xtable::xtable(df[c(1220, 2671), 3:38])

rowSums(df[df$Name == 'Japanese', 3:38])
df[df$Name == 'Japanese', 'latents']


estimate_mode <- function(x) {
  d <- density(x)
  d$x[which.max(d$y)]
}
estimate_mode(scale(df$latents, center = F))

####
preds = read.csv('latentpreds.csv', header = F)
colnames(preds) <- c('latents', 'id', 'true')

preds$runs =rep(1:100, times=1, each=2886)
preds$runs = as.factor(preds$runs)

for (i in 1:100) {
  preds$latents[preds$runs == i] = scale(preds$latents[preds$runs == i], center = F)
}

newpreds = aggregate(latents ~ id, preds, mean)
newpreds$lower = NA
newpreds$upper = NA
for (i in 1:nrow(newpreds)) {
  newpreds$lower[i] = HDInterval::hdi(preds$latents[preds$id == i], credMass = 0.89)[1]
  newpreds$upper[i] = HDInterval::hdi(preds$latents[preds$id == i], credMass = 0.89)[2]
  
}

newpreds$hdidist = newpreds$upper-newpreds$lower

cor(newpreds$hdidist, newpreds$latents)

df = read.csv('phoibledat.csv')
df = df[,-4]
df = cbind(df, newpreds[,c(2,3,4)])

print(df[order(df$latents, decreasing = T),][1:5,c('Name', 'latents')])
print(df[order(df$latents, decreasing = F),][1:5,c('Name', 'latents')])
print(df[order(df$latents, decreasing = F),][1:2, 3:38])


plot_df = df[df$Name %in% c('English', 'French', 'Pirahã', 'South-Eastern Ju', 'Xhosa'),]
a <- ggplot() + 
  geom_linerange(plot_df,mapping= aes(x=reorder(Name,latents),ymin=lower, ymax=upper), lwd = 0.8,color='blue') +
  geom_point(plot_df,mapping= aes(x=reorder(Name,latents), y=latents), size = 1.5, stroke=1, shape = 21, fill ='white',color='black')+
  xlab('Language') + ylab('Complexity (scaled)')+theme_classic()

pdf(file="langs.pdf", height = 5, width = 7)
plot(a)
dev.off()

library(brms)
df$latents_s = scale(df$latents, center = F)

distdf = data.frame()
m_log = brm(latents_s ~ 0 + Intercept, df, family = 'lognormal', cores = 4)
f_log = as.data.frame(m_log$fit)
interv = HDInterval::hdi(f_log$lp__)
distdf = rbind(distdf, data.frame('model' = 'log-normal', 'mean' = mean(f_log$lp__), 'lower' = interv[[1]], 'upper' = interv[[2]]))
m_gamma = brm(latents_s ~ 0 + Intercept, df, family = 'gamma', cores = 4)
f_gamma = as.data.frame(m_gamma$fit)
interv = HDInterval::hdi(f_gamma$lp__)
distdf = rbind(distdf, data.frame('model' = 'gamma', 'mean' = mean(f_gamma$lp__), 'lower' = interv[[1]], 'upper' = interv[[2]]))
m_inv = brm(latents_s ~ 0 + Intercept, df, family = 'inverse.gaussian', iter = 4000, cores = 4)
f_inv = as.data.frame(m_inv$fit)
interv = HDInterval::hdi(f_inv$lp__)
distdf = rbind(distdf, data.frame('model' = 'inverse gaussian', 'mean' = mean(f_inv$lp__), 'lower' = interv[[1]], 'upper' = interv[[2]]))
m_exgaussian = brm(latents_s ~ 0 + Intercept, df, family = 'exgaussian', cores = 4)
f_exgaussian = as.data.frame(m_exgaussian$fit)
interv = HDInterval::hdi(f_exgaussian$lp__)
distdf = rbind(distdf, data.frame('model' = 'exgaussian', 'mean' = mean(f_exgaussian$lp__), 'lower' = interv[[1]], 'upper' = interv[[2]]))
m_exp = brm(latents_s ~ 0 + Intercept, df, family = 'exponential', cores = 4)
f_exp = as.data.frame(m_exp$fit)
interv = HDInterval::hdi(f_exp$lp__)
distdf = rbind(distdf, data.frame('model' = 'exponential', 'mean' = mean(f_exp$lp__), 'lower' = interv[[1]], 'upper' = interv[[2]]))


loo_m_log <- loo(m_log, reloo = T)
loo_m_gamma <- loo(m_gamma, reloo = T)
loo_m_inv <- loo(m_inv, reloo = T)
loo_m_exgaussian <- loo(m_exgaussian, reloo = T)
loo_m_exp <- loo(m_exp, reloo = T)

loo_compare(loo_m_log, loo_m_gamma, loo_m_inv, loo_m_exgaussian, loo_m_exp)
cmp = loo_compare(loo_m_log, loo_m_gamma, loo_m_inv, loo_m_exgaussian, loo_m_exp)
xtable::xtable(cmp)


pdf(file="comphist.pdf", height = 5, width = 7)
hist(df$latents_s, lwd = 0.1, col = rgb(0,0,1,1/4), ylim = c(0,1),
     border=F, probability = T, breaks = 60, xlab = 'Complexity (scaled)', main = '')
curve(dgamma(x, mean(f_gamma[,2]), mean(f_gamma[,2])/exp(mean(f_gamma[,1]))), add = T, col = rgb(1,0,0,.7), lwd = 3)
dev.off()


##
library(factoextra)
library(blavaan)
preds = read.csv('latentpreds.csv', header = F)
colnames(preds) <- c('latents', 'id', 'true')

preds$runs =rep(1:100, times=1, each=2886)
preds$runs = as.factor(preds$runs)

for (i in 1:100) {
  preds$latents[preds$runs == i] = scale(preds$latents[preds$runs == i], center = F)
}
preds$latents = preds$latents/max(preds$latents)

newpreds = aggregate(latents ~ id, preds, mean)
base_df = read.csv('phoibledat.csv')
base_df = base_df[,-4]
base_df = base_df[,c(3:38)]
basesums = rowSums(base_df)
basesums = basesums/max(basesums)
pca <- prcomp(base_df, scale = F, rank. = 1)
pca_vec = pca$x+abs(min(pca$x))
pca_vec = pca_vec/max(pca_vec)



model <- paste(c('latent =~', paste(colnames(base_df), collapse = '+')), collapse = '')
fit <- bsem(model, data = base_df, save.lvs=TRUE)
smry = summary(fit)

baylats = predict(fit)
baylat_df = data.frame(ID=1:2886)
for(i in 1:length(baylats)){
  baylat_df = cbind(baylat_df, data.frame(baylats[[i]]))
}
baylat_df = baylat_df[,-1]
baylats2 = rowMeans(baylat_df)
baylats2 = baylats2+abs(min(baylats2))
baylats2 = baylats2/max(baylats2)
  

cor_df = data.frame(
  'NN' = newpreds$latents,
  'raw' = basesums,
  'PCA' = pca_vec,
  'Bayesian' = baylats2
)
cor_df$NN = cor_df$NN-min(cor_df$NN)
cor_df$raw = cor_df$raw-min(cor_df$raw)
cor(cor_df)

pdf(file="comphist_compare.pdf", height = 5, width = 7)
hist(cor_df$NN, lwd = 0.1, col = rgb(0,0,1,1/4),
     border=F, probability = T, breaks = 60, xlab = 'Complexity (scaled) / cumulative feature counts (scaled)', main = '')
lines(density(cor_df$raw), col = rgb(1,0,0,.7), lwd = 3)
lines(density(cor_df$PC1), col = rgb(.5,0,.8,.7), lwd = 3)
lines(density(cor_df$Bayesian), col = rgb(.5,0,.8,.7), lwd = 3)
dev.off()


vars = get_pca_var(pca)
bayvars = as.data.frame(smry)
bayvars = bayvars[-nrow(bayvars),]
contrib_df = data.frame('shaps'= colMeans(df_shap), 'pca' = rowMeans(vars$contrib), 'bayvars' = as.numeric(bayvars$Estimate)[1:36], 'raw' = colMeans(base_df/rowSums(base_df)))

cor(contrib_df)
plot(contrib_df)

contrib_df2 = data.frame('shaps'= colMeans(df_shap)/mean(df_shap$tone), 'pca' = rowMeans(vars$contrib)/mean(vars$contrib[1,]), 'bayvars' = as.numeric(bayvars$Estimate)[1:36], 'raw' = colMeans(base_df/rowSums(base_df))/colMeans(base_df/rowSums(base_df))[1])

cor(contrib_df2)
plot(contrib_df2)


featimpdf = melt(data.frame(
  NN = scale(colMeans(df_shap)), pca = scale(rowMeans(vars$contrib)), bay = scale(as.numeric(bayvars$Estimate)[1:36]),
  raw = scale(colMeans(base_df/rowSums(base_df))), 'ID' = colnames(df_shap)
))
a <- ggplot() + geom_point()+ 
  geom_point(featimpdf,mapping= aes(x=ID, y=value, fill = variable), size = 1.5, stroke=1, shape = 21)+theme_classic()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


#dists

dist_df_means = data.frame()
dist_df_max = data.frame()
for (i in 1:length(cor_df$NN)) {
  tmp_df = data.frame(
    'NN' = abs(cor_df$NN-cor_df$NN[i]),
    'raw' = abs(cor_df$raw-cor_df$raw[i]),
    'PCA' = abs(cor_df$PC1-cor_df$PC1[i]),
    'Bayesian' = abs(cor_df$Bayesian-cor_df$Bayesian[i])
  )
  
  dist_df_means = rbind(dist_df_means, data.frame(
    'NN_raw' = mean(abs(tmp_df$NN-tmp_df$raw)),
    'NN_PCA' = mean(abs(tmp_df$NN-tmp_df$PCA)),
    'NN_Bayesian' = mean(abs(tmp_df$NN-tmp_df$Bayesian)),
    'raw_PCA' = mean(abs(tmp_df$raw-tmp_df$PCA)),
    'raw_Bayesian' = mean(abs(tmp_df$raw-tmp_df$Bayesian)),
    'Bayesian_PCA' = mean(abs(tmp_df$PCA-tmp_df$Bayesian))
  ))
  
  dist_df_max = rbind(dist_df_max, data.frame(
    'NN_raw' = max(abs(tmp_df$NN-tmp_df$raw)),
    'NN_PCA' = max(abs(tmp_df$NN-tmp_df$PCA)),
    'NN_Bayesian' = max(abs(tmp_df$NN-tmp_df$Bayesian)),
    'raw_PCA' = max(abs(tmp_df$raw-tmp_df$PCA)),
    'raw_Bayesian' = max(abs(tmp_df$raw-tmp_df$Bayesian)),
    'Bayesian_PCA' = max(abs(tmp_df$PCA-tmp_df$Bayesian))
  ))
  
}
mean(dist_df_max$NN_raw)
mean(dist_df_max$NN_PCA)
mean(dist_df_max$NN_Bayesian)
mean(dist_df_max$raw_PCA)

mean(dist_df_means$NN_raw)
mean(dist_df_means$NN_PCA)
mean(dist_df_means$NN_Bayesian)
mean(dist_df_means$raw_PCA)

ggplot(melt(dist_df_max), aes(x = value, fill = variable)) +
  geom_histogram(position = "identity", alpha = 0.4, bins = 50) + facet_wrap(~variable)

