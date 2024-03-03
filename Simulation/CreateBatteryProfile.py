import numpy as np
import matplotlib.pyplot as plt

def i(t, b_a, R_plus_a, tau):
    return np.where(t <= b_a, R_plus_a, R_plus_a * np.exp(-(t - b_a) / tau))

def B(t, b_a, Q_a, R_plus_a, theta_a, tau):
    c = t.copy()
    c[t <= b_a] = t[t <= b_a] * R_plus_a
    c[t > b_a] = Q_a - Q_a * i(t, b_a, R_plus_a, tau)[t > b_a] * (1 - theta_a) / R_plus_a
    return c

# Values for R and theta
R_values = [1, 1, 2, 2]
theta_values = [0.6, 0.8, 0.6, 0.8]

# Create a figure and axis
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#fig.suptitle('R and $\\theta$ Combinations')

plot = False
ttt = 20
# Loop through R and theta combinations
if plot:
    for idx in range(4):
        R_plus_a = R_values[idx]
        theta_a = theta_values[idx]

        b_a = theta_a * 100 / R_plus_a  # Adjusted b_a for better visualization
        tau = ttt/(R_plus_a*theta_a)
        Q_a = 100

        # Generate x values
        t_values = np.linspace(0, 150, 1000)
        i_values = i(t_values, b_a, R_plus_a, tau)
        B_values = B(t_values, b_a, Q_a, R_plus_a, theta_a, tau)

        # Plot on corresponding subplot
        ax = axs[idx // 2, idx % 2]

        # Plot B on the left y-axis
        ax.plot(t_values, B_values, label='$R={}, \\theta={}$'.format(R_plus_a, theta_a), color=plt.cm.viridis(0.5))
        ax.set_xlabel('t')
        ax.set_ylabel('B(t)', color=plt.cm.viridis(0.5))
        ax.tick_params('y', colors=plt.cm.viridis(0.5))

        # Create a secondary y-axis for i
        ax2 = ax.twinx()
        ax2.plot(t_values, i_values, label='i(t)', linestyle='--', color=plt.cm.viridis(0.1))
        ax2.set_ylabel('i(t)', color=plt.cm.viridis(0.1))
        ax2.tick_params('y', colors=plt.cm.viridis(0.1))

        # Vertical line at the expansion point
        ax.axvline(x=b_a, color='black', linestyle='--', label='bp')

        # Set title
        ax.set_title(f'$R^+={R_plus_a}, \\theta={theta_a}, b = {b_a}$')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    plt.savefig('./R_theta_combinations.png')
    plt.show()
else:
    # Loop through R and theta combinations
    for idx in range(4):
        R_plus_a = R_values[idx]
        theta_a = theta_values[idx]

        b_a = theta_a * 100 / R_plus_a  # Adjusted b_a for better visualization
        tau = ttt/(R_plus_a*theta_a)
        Q_a = 100

        # Generate x values
        t_values = np.linspace(0, 150, 1000)
        i_values = i(t_values, b_a, R_plus_a, tau)
        B_values = B(t_values, b_a, Q_a, R_plus_a, theta_a, tau)

        print(f'R_plus_a={R_plus_a}, theta_a={theta_a}')
        #print(t_values)
        #print(i_values)
        #print(B_values)


        print("I")
        #print(" ".join([f"({xx}, {yy if yy > 0.0001 else 0})" for xx, yy in zip(t_values, i_values)]))
        print("B")
        #print(" ".join([f"({xx}, {yy})" for xx, yy in zip(t_values, B_values)]))
        # Print B values
        #for t, B_val in zip(t_values, B_values):
        #    print(f't={t}, B(t)={B_val}')
#
        ## Print i values
        #for t, i_val in zip(t_values, i_values):
        #    print(f't={t}, i(t)={i_val}')

        print(f'b_a={b_a}, tau={tau}')

        # Vertical line at the expansion point
        print(f'Expansion point: b_a={b_a}')

        print('------------------------------------')

