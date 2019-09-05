# Modified steepest descent porgram for the course "Basics of Machine Learning".
# Original program uses straight line as fitting model and fixed step for steepest descent iterations.
# Modified to use a parabola as fitting model and backtracking line search for steepest descent iterations.
# Modified to use an acceptability criterion for iteration termination.
# Modified by Henrik Kurkela, Sept. 2019.

import matplotlib.pyplot as plt
import numpy as np

# Adjustable variables
rounds_total = 10000 # Maximum amount of iterations before termination.
acceptable_fit = 0.002 # Acceptability criterion. When reached we can stop iterating and accept the values for t0, t1 and t2.
learning_rate = 0.01 # Learning rate is initially set as this value in the beginning of each iteration.

# Let's make first random data to which we try to fit a parabola.
M = 100
x = 5*np.random.randn(M)
y = 0.05*np.random.randn(M)

plt.figure(1)
plt.plot(x,y,'*r')
plt.title('Our random data')

# Now let's try to fit a parabola to that known data.
# Our hypothesis function h(x) = t2*x^2 + t1*x + t0, where
# t2, t1 and t0 are randomly initialized parameters.
t0 = np.random.randn(1)
t1 = np.random.randn(1)
t2 = np.random.randn(1) 

# Let's now iterate many times. I.e. each time
# we calculate a new gradient value for our cost function
# and use bactracking line search to select the new values for t0, t1, t2.
t0_adjusted_values = []
t1_adjusted_values = []
t2_adjusted_values = []

# Backtracking line search variables.
number_of_misses = 0
ready_to_report = False

for rounds in range(rounds_total):
    grad_t0 = 0.00
    grad_t1 = 0.00
    grad_t2 = 0.00
    are_we_satisfied = False
    rate = learning_rate
    current_cost = 0.00
    # Let's calculate new gradient values for t0 and t1 based on M known values:
    for i in range(M):
        # Our cost function c = (1/2M)*(h(x)-y)^2
        # where M = number of known values.
        # Partial derivatives for t1 and t0 are:
        # t0 =>(1/M)*(h(x)-y)*1 => ((t2*x(i)^2 + t1*x(i) + t0) - y(i)) * 1
        # t1 =>(1/M)*(h(x)-y)*x => ((t2*x(i)^2 + t1*x(i) + t0) - y(i)) * x(i)
        # Partial derivative for t2 is:
        # (1/M) * ((t2*x(i)^2 + t1*x(i) + t0) - y(i)) * x(i)^2
        # where i goes from 1 => M
        grad_t0 = grad_t0 + (1.00 / M) * (t2 * pow(x[i], 2) + t1 * x[i] + t0 - y[i])
        grad_t1 = grad_t1 + (1.00 / M) * (t2 * pow(x[i], 2) + t1 * x[i] + t0 - y[i]) * x[i]
        grad_t2 = grad_t2 + (1.00 / M) * (t2 * pow(x[i], 2) + t1 * x[i] + t0 - y[i]) * pow(x[i], 2)
    # Calculate the value of the cost function at current position:
    for i in range(M):
        current_cost = current_cost + (0.5 / M) * pow((t2 * pow(x[i], 2) + t1 * x[i] + t0 - y[i]),2)
    # Calculate the value of the cost function at a candidate jump position.
    # If the value of the cost function increases we reject the candidate and reduce the jump size.
    # This will be done until a satisfactory candidate jump position is discovered
    # or the maximum value for acceptable fit is reached.
    while are_we_satisfied == False:
        value_of_cost = 0.00
        for i in range(M):
            value_of_cost = value_of_cost + (0.5 / M) * pow(((t2 - rate * grad_t2) * pow(x[i], 2) + (t1 - rate * grad_t1) * x[i] + (t0 - rate * grad_t0) - y[i]), 2)
        # Acceptable fit. we can stop iterating and report findings.
        if value_of_cost < acceptable_fit:
            are_we_satisfied = True
            ready_to_report = True
        # Satisfactory value. we can continue.
        elif current_cost > value_of_cost:
            are_we_satisfied = True
        # Unsatisfactory value, we reduce the step, recalculate the cost and increment the miss counter.
        else:
            rate = rate * 0.5
            number_of_misses = number_of_misses + 1
    # Make the step:
    t0 = t0 - rate * grad_t0
    t1 = t1 - rate * grad_t1
    t2 = t2 - rate * grad_t2
    # Save the current location to a list:
    t0_adjusted_values.append(t0)
    t1_adjusted_values.append(t1)
    t2_adjusted_values.append(t2)
    # If satisfactory fit is reached we can stop iterating:
    if ready_to_report == True:
        break
# Print calculation details in the console window:
print('How many times the learning rate had to be reduced due to back-tracking line search constraints: {}'.format(number_of_misses))
print('How many iterations were calculated: {}'.format(len(t0_adjusted_values)))
# And finally lets visualize results:
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
