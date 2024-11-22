library(ggplot2)
library(reshape2)
require(MASS)
library(brms)
set.seed(42)

base_df = read.csv('phoibledat.csv')
base_df = base_df[,-4]
base_df2 = base_df[,c(3:38)]
long_df = melt(base_df2)

p = ggplot() + geom_histogram(data = long_df, mapping = aes(value, after_stat(density)), bins = 30) + facet_wrap(~variable, scales = 'free') +
  theme_classic() + ylab('Density') + xlab('Variable')

pdf(file = 'allfeathist.pdf', width = 12, height = 8)
p
dev.off()

##Central Limit Theorem Simulation


shuffle_df = base_df2
for(col in 1:ncol(shuffle_df)){
  shuffle_df[,col] = sample(shuffle_df[,col])
}
sums_vals = rowSums(shuffle_df)

plot_df = data.frame('vals' = c(sums_vals, rowSums(base_df2)), 'type' = rep(c('Shuffled', 'Real'), each = nrow(base_df2)) )

p = ggplot() +
  geom_histogram(data = plot_df, mapping = aes(vals, after_stat(density), fill = type), bins = 100) + xlab('Feature sum') + 
  theme_classic()+ guides(fill=guide_legend(title="Type")) + ylab('Density')

pdf(file = 'shufflesum.pdf', width = 6, height = 4)
p
dev.off()

plot_df$vals = scale(plot_df$vals, center = F)
out1 = brm(vals ~ 0 + Intercept , plot_df[plot_df$type == 'Real',], family = 'lognormal', cores = 4)
out2 = brm(vals ~ 0 + Intercept , plot_df[plot_df$type == 'Shuffled',], family = 'lognormal', cores = 4)
summary(out1)
summary(out2)


############
#####NN
############

df_cor = read.csv('corr_df_new.csv')
df_cor = df_cor[,-c(1,3)]
df_shap = read.csv('shap_dist_new.csv')
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
feature_df = newdf
feature_df$model = 'NN'
feature_df$weighted = 'no'


#######
df_cor = read.csv('corr_df_new.csv')
df_cor = df_cor[,-c(1,3)]
df_shap = read.csv('weighted_shap_dist_new.csv')
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

newdf$model = 'NN'
newdf$weighted = 'yes'

feature_df = rbind(feature_df, newdf)


####



preds = read.csv('newlatentpreds.csv', header = F)
colnames(preds) <- c('latents', 'id', 'true')
newdf = read.csv('phoibledat.csv')
newdf = newdf[, c('ID', 'Name')]

newdf$value = NA
newdf$lower = NA
newdf$upper = NA
for (i in 1:2886) {
  newdf$value[i] = mean(preds$latents[preds$id == i])
  newdf$lower[i] = HDInterval::hdi(preds$latents[preds$id == i], credMass = 0.89)[1]
  newdf$upper[i] = HDInterval::hdi(preds$latents[preds$id == i], credMass = 0.89)[2]
  
}
latent_df = newdf
latent_df$model = 'NN'


############
#####Raw sums
############


base_df = read.csv('phoibledat.csv')
base_df = base_df[,-4]
base_df2 = base_df[,c(3:38)]
base_language_sums = rowSums(base_df2)
base_feature_sums = colSums(base_df2)

newdf = base_df[, c('ID', 'Name')]
newdf$value = base_language_sums
newdf$lower = NA
newdf$upper = NA
newdf$model = 'Raw sums'
latent_df = rbind(latent_df, newdf)

newdf = data.frame( 'variable' = feature_df[1:36, 'variable'], 'value' = base_feature_sums, 'lower' = NA, 
                    'upper' = NA, 'model' = 'Raw sums', 'weighted' = 'no')
rownames(newdf) <- NULL
feature_df = rbind(feature_df, newdf)

############
#####Bayesian
############

library(factoextra)
library(blavaan)

base_df = read.csv('phoibledat.csv')
base_df = base_df[,-4]
base_df = base_df[,c(3:38)]
model <- paste(c('latent =~', paste(colnames(base_df), collapse = '+')), collapse = '')
#fit <- bsem(model, data = base_df, save.lvs=TRUE)
#save(fit, file= 'Bayfit.rda')
load('Bayfit.rda')
baylats = predict(fit)

baylat_df = data.frame(ID=1:2886)
for(i in 1:length(baylats)){
  baylat_df = cbind(baylat_df, data.frame(baylats[[i]]))
}
baylat_df = t(baylat_df[,-1])

newdf = read.csv('phoibledat.csv')
newdf = newdf[, c('ID', 'Name')]

newdf$value = NA
newdf$lower = NA
newdf$upper = NA
for (i in 1:2886) {
  newdf$value[i] = mean(baylat_df[,i])
  newdf$lower[i] = HDInterval::hdi(baylat_df[,i], credMass = 0.89)[1]
  newdf$upper[i] = HDInterval::hdi(baylat_df[,i], credMass = 0.89)[2]
  
}


newdf$model = 'Bayesian'
latent_df = rbind(latent_df, newdf)

####

smry = summary(fit)
bayvars = as.data.frame(smry)
bayvars = bayvars[-nrow(bayvars),]


newdf = data.frame( 'variable' = feature_df[1:36, 'variable'], 'value' = as.numeric(bayvars$Estimate)[1:36], 'lower' = as.numeric(bayvars$pi.lower)[1:36], 
                    'upper' = as.numeric(bayvars$pi.upper)[1:36], 'model' = 'Bayesian', 'weighted' = 'no')

feature_df = rbind(feature_df, newdf)


newdf = data.frame( 'variable' = feature_df[1:36, 'variable'], 'value' = newdf$value/feature_df$value[feature_df$model == 'Raw sums'], 'lower' = newdf$lower/feature_df$value[feature_df$model == 'Raw sums'], 
                    'upper' = newdf$upper/feature_df$value[feature_df$model == 'Raw sums'], 'model' = 'Bayesian', 'weighted' = 'yes')

feature_df = rbind(feature_df, newdf)

############
#####PCA
############


#visualization

slope_coef = 1

corr_matrix = matrix(c(1,0,0,1), ncol = 2)
tau = c(0.3, 0.3)
covm = diag(tau) %*% corr_matrix %*% diag(tau)
randmnorm = rbind(MASS::mvrnorm(10, c(-1,-1), covm),MASS::mvrnorm(10, c(1,1), covm))

viz_df = data.frame(Dim1 = randmnorm[,1], Dim2 = randmnorm[,2], 'id' = 1:20)
proj_df1 =  data.frame( Dim1 = (randmnorm[,1] + slope_coef * randmnorm[,2]) / (slope_coef^2 + 1) , 'id' = 1:20)
proj_df1$Dim2 = slope_coef * proj_df1$Dim1
joint_df1 = rbind(viz_df, proj_df1)

proj_df2 = data.frame( Dim1 = (randmnorm[,1] + -slope_coef * randmnorm[,2]) / ((-slope_coef)^2 + 1) , 'id' = 1:20)
proj_df2$Dim2 = -slope_coef * proj_df2$Dim1
joint_df2 = rbind(viz_df, proj_df2)

p1 <- ggplot() + 
  geom_line(joint_df1, mapping = aes(y = Dim2,x = Dim1, group = id), color = 'black', linewidth = 0.8, alpha = 0.3, lineend='round', linetype = 'dotted')+
  geom_point(viz_df,mapping= aes(x=Dim1, y=Dim2), fill = 'white', size = 1.5, stroke=1, shape = 21,color='red', alpha = 0.8)+
  theme_classic() + geom_abline(mapping= aes(intercept = 0, slope = slope_coef), lwd = 1, alpha = 0.9, color = 'blue') + theme(axis.title.x=element_blank())

p2 <- ggplot() + 
  theme_classic() + geom_abline(mapping= aes(intercept = 0, slope = 0), lwd = 1, alpha = 0.9, color = 'blue')+
  geom_point(proj_df1,mapping= aes(x=Dim2, y=rep(0,20)), fill = 'white', size = 3, stroke=1, shape = 21,color='red', alpha = 0.8) + xlim(-4,4)+ theme(axis.title.x=element_blank())+ theme(axis.title.y=element_blank())+xlab('PC')

p3 <- ggplot() + 
  geom_line(joint_df2, mapping = aes(y = Dim2,x = Dim1, group = id), color = 'black', linewidth = 0.8, alpha = 0.3, lineend='round', linetype = 'dotted')+
  geom_point(viz_df,mapping= aes(x=Dim1, y=Dim2), fill = 'white', size = 1.5, stroke=1, shape = 21,color='red', alpha = 0.8)+
  theme_classic() + geom_abline(mapping= aes(intercept = 0, slope = -slope_coef), lwd = 1, alpha = 0.9, color = 'blue')

p4 <- ggplot() + 
  theme_classic() + geom_abline(mapping= aes(intercept = 0, slope = 0), lwd = 1, alpha = 0.9, color = 'blue')+
  geom_point(proj_df2,mapping= aes(x=Dim2, y=rep(0,20)), fill = 'white', size = 3, stroke=1, shape = 21,color='red', alpha = 0.8)+ xlim(-4,4)+ theme(axis.title.y=element_blank()) +xlab('PC')

p = ggpubr::ggarrange(p1,p2,p3, p4, ncol = 2, nrow = 2)

pdf(file = 'pca_explain.pdf', width = 6, height = 6)
p
dev.off()

base_df = read.csv('phoibledat.csv')
base_df = base_df[,-4]
base_df = base_df[,c(3:38)]
pca <- prcomp(base_df, scale = F, rank. = 1)

newdf = data.frame( 'variable' = feature_df[1:36, 'variable'], 'value' = pca$rotation[,1], 'lower' = NA, 
                    'upper' = NA, 'model' = 'PCA', 'weighted' = 'no')

feature_df = rbind(feature_df, newdf)

newdf = data.frame( 'variable' = feature_df[1:36, 'variable'], 'value' = pca$rotation[,1]/feature_df$value[feature_df$model == 'Raw sums'], 'lower' = NA, 
                    'upper' = NA, 'model' = 'PCA', 'weighted' = 'yes')

feature_df = rbind(feature_df, newdf)


#normalization


feature_df_norm = feature_df

for (i in c('yes', 'no')) {
  for (e in c('NN', 'PCA', 'Raw sums', 'Bayesian')) {
    #sd = sd(feature_df_norm$value[feature_df_norm$model == e & feature_df_norm$weighted == i])
    sd = max(feature_df_norm$value[feature_df_norm$model == e & feature_df_norm$weighted == i])
    feature_df_norm$upper[feature_df_norm$model == e & feature_df_norm$weighted == i] = feature_df_norm$upper[feature_df_norm$model == e & feature_df_norm$weighted == i]/sd
    feature_df_norm$lower[feature_df_norm$model == e & feature_df_norm$weighted == i] = feature_df_norm$lower[feature_df_norm$model == e & feature_df_norm$weighted == i]/sd
    feature_df_norm$value[feature_df_norm$model == e & feature_df_norm$weighted == i] = feature_df_norm$value[feature_df_norm$model == e & feature_df_norm$weighted == i]/sd
  }
}

plot_df = feature_df_norm[feature_df_norm$weighted =='no',]
library(RColorBrewer)
clrs = RColorBrewer::brewer.pal(3, 'Set1')
p1 <- ggplot() + 
  geom_linerange(plot_df[plot_df$model == 'NN',],mapping= aes(x=reorder(variable,value),ymin=lower, ymax=upper), color = clrs[1], lwd = 0.8) +
  geom_point(plot_df[plot_df$model == 'NN',],mapping= aes(x=reorder(variable,value), y=value), fill = clrs[1], size = 1.5, stroke=1, shape = 21,color='black') + 
  geom_hline(yintercept=0, linetype="dashed", color = "black")+
  xlab('feature')+theme_classic()+ ggtitle('Neural Network') + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

p2 <- ggplot() + 
  geom_linerange(plot_df[plot_df$model == 'Bayesian',],mapping= aes(x=reorder(variable,value),ymin=lower, ymax=upper), color = clrs[2], lwd = 0.8) +
  geom_point(plot_df[plot_df$model == 'Bayesian',],mapping= aes(x=reorder(variable,value), y=value), fill = clrs[2], size = 1.5, stroke=1, shape = 21,color='black') + 
  geom_hline(yintercept=0, linetype="dashed", color = "black")+
  xlab('feature')+theme_classic()+ggtitle('Bayesian model') + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), axis.title.y=element_blank())

p3 <- ggplot() + 
  geom_linerange(plot_df[plot_df$model == 'PCA',],mapping= aes(x=reorder(variable,value),ymin=lower, ymax=upper), color = clrs[3], lwd = 0.8) +
  geom_point(plot_df[plot_df$model == 'PCA',],mapping= aes(x=reorder(variable,value), y=value), fill = clrs[3], size = 1.5, stroke=1, shape = 21,color='black') + 
  geom_hline(yintercept=0, linetype="dashed", color = "black")+
  xlab('feature')+theme_classic()+ggtitle('PCA') + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), axis.title.y=element_blank())

p = ggpubr::ggarrange(p1,p2,p3, ncol = 3)
pdf(file = 'raw_features.pdf', width = 12, height = 4)
p
dev.off()


plot_df = feature_df_norm[feature_df_norm$weighted =='yes',]
library(RColorBrewer)
clrs = RColorBrewer::brewer.pal(3, 'Set1')
p1 <- ggplot() + 
  geom_linerange(plot_df[plot_df$model == 'NN',],mapping= aes(x=reorder(variable,value),ymin=lower, ymax=upper), color = clrs[1], lwd = 0.8) +
  geom_point(plot_df[plot_df$model == 'NN',],mapping= aes(x=reorder(variable,value), y=value), fill = clrs[1], size = 1.5, stroke=1, shape = 21,color='black') + 
  geom_hline(yintercept=0, linetype="dashed", color = "black")+
   xlab('feature')+theme_classic()+ ggtitle('Neural Network') + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


p2 <- ggplot() + 
  geom_linerange(plot_df[plot_df$model == 'Bayesian',],mapping= aes(x=reorder(variable,value),ymin=lower, ymax=upper), color = clrs[2], lwd = 0.8) +
  geom_point(plot_df[plot_df$model == 'Bayesian',],mapping= aes(x=reorder(variable,value), y=value), fill = clrs[2], size = 1.5, stroke=1, shape = 21,color='black') + 
  geom_hline(yintercept=0, linetype="dashed", color = "black")+
  xlab('feature')+theme_classic()+ggtitle('Bayesian model') + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), axis.title.y=element_blank())

p3 <- ggplot() + 
  geom_linerange(plot_df[plot_df$model == 'PCA',],mapping= aes(x=reorder(variable,value),ymin=lower, ymax=upper), color = clrs[3], lwd = 0.8) +
  geom_point(plot_df[plot_df$model == 'PCA',],mapping= aes(x=reorder(variable,value), y=value), fill = clrs[3], size = 1.5, stroke=1, shape = 21,color='black') + 
  geom_hline(yintercept=0, linetype="dashed", color = "black")+
  xlab('feature')+theme_classic()+ggtitle('PCA') + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), axis.title.y=element_blank())

p = ggpubr::ggarrange(p1,p2,p3, ncol = 3)
pdf(file = 'weighted_features.pdf', width = 12, height = 4)
p
dev.off()

correlation = cor(data.frame('NN' = plot_df$value[plot_df$model == 'NN'],
               'PCA' = plot_df$value[plot_df$model == 'PCA'],
               'Bayesian' = plot_df$value[plot_df$model == 'Bayesian']), method = 'spearman')

xtable::xtable(correlation)
###latent dimensions


correlation = cor(data.frame('NN' = latent_df$value[latent_df$model == 'NN'],
               'Raw sums' = latent_df$value[latent_df$model == 'Raw sums'],
               'Bayesian' = latent_df$value[latent_df$model == 'Bayesian']), method = 'spearman')
xtable::xtable(correlation)

latent_df$rank = NA
latent_df$rank[latent_df$model == 'NN'][order(latent_df$value[latent_df$model == 'NN'], decreasing = T)]= 1:nrow(base_df)
latent_df$rank[latent_df$model == 'Raw sums'][order(latent_df$value[latent_df$model == 'Raw sums'], decreasing = T)]= 1:nrow(base_df)
latent_df$rank[latent_df$model == 'Bayesian'][order(latent_df$value[latent_df$model == 'Bayesian'], decreasing = T)] = 1:nrow(base_df)

latent_df_norm = latent_df

for (e in c('NN', 'Raw sums', 'Bayesian')) {
  tmp = latent_df_norm$value[latent_df_norm$model == e ]
  tmp = tmp + abs(min(tmp))
  tmp = tmp/max(tmp)
  latent_df_norm$value[latent_df_norm$model == e ] = tmp
}

models = c('NN', 'Raw sums', 'Bayesian')
exact_rank = matrix(NA, 3,3)
for(m1 in 1:3){
  for(m2 in 1:3){
    count = 0
    for (rank in 1:nrow(base_df)){
        count = count + as.numeric(latent_df_norm$ID[latent_df_norm$rank == rank & latent_df_norm$model == models[m1]] == latent_df_norm$ID[latent_df_norm$rank == rank & latent_df_norm$model == models[m2]])
    }
    exact_rank[m1,m2] = count
  }
}

exact_rank = exact_rank/2286

latent_df_norm[latent_df_norm$model == 'NN',][order(latent_df_norm$value[latent_df_norm$model == 'NN'], decreasing = T),][1:5,]
latent_df_norm[latent_df_norm$model == 'NN',][order(latent_df_norm$value[latent_df_norm$model == 'NN'], decreasing = T),][2882:2886,]

latent_df_norm[latent_df_norm$model == 'Bayesian',][order(latent_df_norm$value[latent_df_norm$model == 'Bayesian'], decreasing = T),][1:5,]
latent_df_norm[latent_df_norm$model == 'Bayesian',][order(latent_df_norm$value[latent_df_norm$model == 'Bayesian'], decreasing = T),][2882:2886,]

latent_df_norm[latent_df_norm$model == 'Raw sums',][order(latent_df_norm$value[latent_df_norm$model == 'Raw sums'], decreasing = T),][1:5,]
latent_df_norm[latent_df_norm$model == 'Raw sums',][order(latent_df_norm$value[latent_df_norm$model == 'Raw sums'], decreasing = T),][2882:2886,]


#
comp_df = data.frame(ID = latent_df_norm$ID, NN =latent_df_norm$value[latent_df_norm$model == 'NN'], Bayesian =latent_df_norm$value[latent_df_norm$model == 'Bayesian'] , RawSums =latent_df_norm$value[latent_df_norm$model == 'Raw sums'])

comp_df$NN_Bay = comp_df$NN - comp_df$Bayesian
comp_df$NN_Raw = comp_df$NN - comp_df$RawSums
comp_df$Bay_Raw = comp_df$Bayesian - comp_df$RawSums


###shape determination


library(brms)
determine_shape <- function(latents, model){
  df = data.frame('latents' = latents+0.001)
  m_log = brm(latents ~ 0 + Intercept, df, family = 'lognormal', cores = 4)
  loo_log = loo(m_log)
  m_gamma = brm(latents ~ 0 + Intercept, df, family = 'gamma', cores = 4)
  loo_gamma = loo(m_gamma)
  m_inv = brm(latents ~ 0 + Intercept, df, family = 'inverse.gaussian', iter = 4000, cores = 4)
  loo_inv = loo(m_inv)
  m_exgaussian = brm(latents ~ 0 + Intercept, df, family = 'exgaussian', cores = 4)
  loo_exgaussian = loo(m_exgaussian)
  m_exp = brm(latents ~ 0 + Intercept, df, family = 'exponential', cores = 4)
  loo_exp = loo(m_exp)
  cmpr = as.data.frame(loo_compare(loo_log, loo_gamma, loo_inv, loo_exgaussian, loo_exp))
  cmpr$model = model
  cmpr$shape = c('log-normal', 'gamma', 'inverse gaussian', 'exgaussian', 'exponential')
  rownames(cmpr) <- NULL
  return(cmpr)
}

distdf_NN = determine_shape(latent_df_norm$value[latent_df_norm$model == 'NN'], model = 'NN')
distdf_Bayesian = determine_shape(latent_df_norm$value[latent_df_norm$model == 'Bayesian'], model = 'Bayesian')
distdf_RawSums = determine_shape(latent_df_norm$value[latent_df_norm$model == 'Raw sums'], model = 'Raw sums')

joint_df = rbind(distdf_NN,distdf_Bayesian, distdf_RawSums)
save(joint_df, file = 'joint_df.rda')

load(file = 'joint_df.rda')

p <- ggplot() + 
  geom_linerange(joint_df,mapping= aes(x=reorder(shape, elpd_diff),ymin=elpd_diff-se_diff, ymax=elpd_diff+se_diff, color = model), lwd = 0.8, position = position_dodge(.3)) +
  geom_point(joint_df,mapping= aes(x=reorder(shape, elpd_diff), y=elpd_diff, fill = model), size = 1.5, stroke=1, shape = 21,color='black', position = position_dodge(.3)) + 
  xlab('feature')+theme_classic() +facet_wrap(~model)

mytable = joint_df[,c('elpd_diff', 'se_diff', 'model', 'shape')]
xtable::xtable(mytable)

m_log_NN = brm(value+0.001 ~ 0 + Intercept, latent_df_norm[latent_df_norm$model == 'NN',], family = 'lognormal', cores = 4)
m_log_Bayesian = brm(value+0.001 ~ 0 + Intercept, latent_df_norm[latent_df_norm$model == 'Bayesian',], family = 'lognormal', cores = 4)
m_log_NN_RawSums = brm(value+0.001 ~ 0 + Intercept, latent_df_norm[latent_df_norm$model == 'Raw sums',], family = 'lognormal', cores = 4)

lognorm_dens_NN = dlnorm(seq(0,1,0.001), summary(m_log_NN$fit)$summary[1,1], summary(m_log_NN$fit)$summary[2,1])
lognorm_dens_Bayesian = dlnorm(seq(0,1,0.001), summary(m_log_Bayesian$fit)$summary[1,1], summary(m_log_Bayesian$fit)$summary[2,1])
lognorm_dens_RawSums = dlnorm(seq(0,1,0.001), summary(m_log_NN_RawSums$fit)$summary[1,1], summary(m_log_NN_RawSums$fit)$summary[2,1])


dens_df = rbind( data.frame(value = seq(0,1,0.001), dens = lognorm_dens_NN, model = 'NN'),
                data.frame(value = seq(0,1,0.001), dens = lognorm_dens_Bayesian, model = 'Bayesian'),
                data.frame(value = seq(0,1,0.001), dens = lognorm_dens_RawSums, model = 'Raw sums')
                
)

latent_df_norm$logvalue = log(latent_df_norm$value)
p1 = ggplot() +
  geom_histogram(data = latent_df_norm, mapping = aes(value, after_stat(density), fill = model), bins = 100) + 
  geom_line(data= dens_df, mapping = aes(value, dens), color = 'black', lwd = 0.6)+ xlab('Latent value (norm.)') + 
  theme_classic()+ facet_wrap(~model, ncol = 1)+ guides(fill=guide_legend(title="Model")) + ylab('Density')
p2 = ggplot() +
  geom_histogram(data = latent_df_norm, mapping = aes(logvalue, after_stat(density), fill = model), bins = 100) + 
  xlab('Log-latent value (norm.)') + 
  theme_classic()+ facet_wrap(~model, ncol = 1) + guides(fill=guide_legend(title="Model"))+ ylab('Density')+ theme(axis.title.y=element_blank())


pdf(file = 'latdens.pdf', width = 5, height = 5)
ggpubr::ggarrange(p1,p2, common.legend = T,legend = 'bottom')
dev.off()
