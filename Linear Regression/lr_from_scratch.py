from statistics import mean
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import random

style.use('fivethirtyeight')

#xs = np.asarray([1,2,3,4,5,6], dtype=np.float64)
#ys = np.asarray([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, step, variance, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
	xs = [i for i in range(len(ys))]
	
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
	m = ((mean(xs) * mean(ys)) - mean(xs*ys))/((mean(xs)*mean(xs)) - mean(xs*xs))  # PEMDAS
	
	b = mean(ys) - (m * mean(xs))
	return m, b


def squared_error(ys_orig, ys_line):
	return sum((ys_orig - ys_line) ** 2)


def coeff_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_mean_line = squared_error(ys_orig, y_mean_line)
	return (1 - (squared_error_regr/squared_error_mean_line))

xs, ys = create_dataset(40, 2, 40, correlation='pos')
m,b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m*x) + b for x in xs]
r_squared = coeff_of_determination(ys, regression_line)

predict_xs = 8
predict_ys = (m*predict_xs) + b

plt.scatter(xs, ys)
plt.scatter(predict_xs, predict_ys, color='g', s='100')
plt.plot(xs, regression_line)
plt.show()
