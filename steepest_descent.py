# modified steepest descent porgram for the course "Basics of Machine Learning"
# original program uses straight line as fitting model and fixed step for steepest descent iterations
# modified to use a parabola as fitting model and backtracking line search for steepest descent iterations
# modified by Henrik Kurkela, Sept 2019

import matplotlib.pyplot as plt
import numpy as np

# lets make first random data to which we try to fit a parabola
M = 100
x = 5*np.random.randn(M)
y = 0.05*np.random.randn(M)

plt.figure(1)
plt.plot(x,y,'*r')
plt.title('Our random data')

# now let's try to fit a parabola to that known data
# our hypothesis function h(x) = t2*x^2 + t1*x + t0, where
# t2, t1 and t0 are randomly initialized parameters
# we multiply t0's random value by 0.1 so it has a greater chance of being closer to zero than t1 or t2
# this is done because the t0 parameter is by far the slowest to converge
t0 = 0.1 * np.random.randn(1)
t1 = np.random.randn(1)
t2 = np.random.randn(1) 

# let's now iterate many times. I.e. each time
# we calculate a new gradient value for our cost function
# and use bactracking line search to select the new values for t0, t1, t2
t0_adjusted_values = []
t1_adjusted_values = []
t2_adjusted_values = []
rounds_total = 500
rate = 1.00
for rounds in range(rounds_total):
    grad_t0 = 0.00
    grad_t1 = 0.00
    grad_t2 = 0.00
    # lets calculate new gradient values for t0 and t1 based on
    # M known values
    for i in range(M):
        # our cost function c = (1/2M)*(h(x)-y)^2
        # where M = number of known values
        # partial derivatives for t1 and t0 are
        # t0 =>(1/M)*(h(x)-y)*1 => ((t2*x(i)^2 + t1*x(i) + t0) - y(i)) * 1
        # t1 =>(1/M)*(h(x)-y)*x => ((t2*x(i)^2 + t1*x(i) + t0) - y(i)) * x(i)
        # partial derivative for t2 is
        # (1/M) * ((t2*x(i)^2 + t1*x(i) + t0) - y(i)) * x(i)^2
        # where i goes from 1 => M
        grad_t0 = grad_t0 + (1.00 / M) * (t2 * pow(x[i], 2) + t1 * x[i] + t0 - y[i])
        grad_t1 = grad_t1 + (1.00 / M) * (t2 * pow(x[i], 2) + t1 * x[i] + t0 - y[i]) * x[i]
        grad_t2 = grad_t2 + (1.00 / M) * (t2 * pow(x[i], 2) + t1 * x[i] + t0 - y[i]) * pow(x[i], 2)
    are_we_satisfied = False
    rate = rate * 1.1
    current_cost = 0.00
    # calculate the value of the cost function at current position
    for i in range(M):
        current_cost = current_cost + (0.5 / M) * pow((t2 * pow(x[i], 2) + t1 * x[i] + t0 - y[i]),2)
    # calculate the value of the cost function at a candidate jump position
    # if the value of the cost function increases we reject the candidate and reduce the jump size
    # this will be done until a satisfactory candidate jump position is discovered
    while are_we_satisfied == False:
        value_of_cost = 0.00
        for i in range(M):
            value_of_cost = value_of_cost + (0.5 / M) * pow(((t2 - rate * grad_t2) * pow(x[i], 2) + (t1 - rate * grad_t1) * x[i] + (t0 - rate * grad_t0) - y[i]), 2)
        # satisfactory value. we can continue
        if current_cost > value_of_cost:
            are_we_satisfied = True
        # unsatisfactory value, we reduce the step and recalculate the cost
        else:
            rate = rate * 0.9
    # make the step
    t0 = t0 - rate * grad_t0
    t1 = t1 - rate * grad_t1
    t2 = t2 - rate * grad_t2
    # save the current location to a list 
    t0_adjusted_values.append(t0)
    t1_adjusted_values.append(t1)
    t2_adjusted_values.append(t2)
    # and finally lets visualize results
    xmin = np.min(x)
    xmax = np.max(x)
    x1 = np.linspace(xmin,xmax,100)
    h = t2*x1*x1 + t1*x1 + t0

plt.figure(1)
plt.clf()
plt.plot(x,y,'*r')
plt.plot(x1, h, 'b')
plt.title('Known datapoints and fitting (Current cost: {})'.format(current_cost))

plt.figure(2)
plt.plot(t0_adjusted_values,'-b')
plt.title('Adjusted t0 values(Current value: {})'.format(t0))

plt.figure(3)
plt.plot(t1_adjusted_values,'-g')
plt.title('Adjusted t1 values (Current value: {})'.format(t1))

plt.figure(4)
plt.plot(t2_adjusted_values,'-m')
plt.title('Adjusted t2 values (Current value: {})'.format(t2))
plt.show()
