import numpy as np
from scipy.stats import multivariate_normal


__all__ = ["gaussian_generator_ind"]



# def gaussian_generator_ind(means, variances, dim=10, sample_size = 500):
# 	sample_size = int(sample_size/len(means))
# 	data = []
# 	labels = []
# 	n_classes= len(means)
# 	for i in range(n_classes):
# 		mean = [means[i]]*dim
# 		cov = np.eye(dim)*variances[i]
# 		print(mean)
# 		print(cov)
# # mean=[2]*10, cov=np.eye(10)*1
# 		mn = multivariate_normal(mean=mean, cov=cov)
# 		data += mn.rvs(size=sample_size).tolist()
# 		#data +=np.random.multivariate_normal(mean, cov, sample_size).tolist()
# 		# print(mn.pdf(data))
# 		labels += [i]*sample_size 
# 	data = np.array(data)
# 	labels = np.array(labels)
	
# 	return data, labels


	
def gaussian_generator_ind(means, variances, dim=10, sample_size = 500):
	sample_size = int(sample_size/len(means))
	data = []
	labels = []
	n_classes= len(means)

	mn_neg = multivariate_normal(mean=[means[0]]*dim, cov=np.eye(dim)*variances[0])
	mn_pos = multivariate_normal(mean=[means[1]]*dim, cov=np.eye(dim)*variances[1])
	labels = [0]*sample_size +[1]*sample_size
	neg_data = mn_neg.rvs(size=sample_size).tolist()
	pos_data = mn_pos.rvs(size=sample_size).tolist()
	data = neg_data + pos_data
	data = np.array(data)
	labels = np.array(labels)
	posterior = get_posterior(data, mn_neg, mn_pos)
	return data, labels, posterior


def get_posterior(x,mn_neg,mn_pos):
  
	neg_density = mn_neg.pdf(x)
	pos_density = mn_pos.pdf(x)
	x_density = neg_density+pos_density
	neg_post = neg_density/x_density
	pos_post = pos_density/x_density
	dist = np.array([neg_post,pos_post])
	dist = dist.T
	

	return dist
