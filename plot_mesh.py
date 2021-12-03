import numpy
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm

mu_1_mean = 0.6

file = open('plotting_data/experiments_{}/test_cases.pkl'.format(mu_1_mean), 'rb')
test_cases = pickle.load(file)

width, delta, thompson_diff_regret, ucb_diff_regret = [], [], [], []

for test_case in test_cases:
    width_i, delta_i = test_case[0], test_case[1]
    width.append(width_i)
    delta.append(delta_i)

    file = open('plotting_data/experiments_{}/thompson_data_delta{}_width{}.pkl'.format(mu_1_mean, delta_i, width_i), 'rb')
    woRejection, wRejection, ucb, ucbwConfidence = pickle.load(file)
    woRejection, wRejection = numpy.array(woRejection), numpy.array(wRejection)
    ucb, ucbwConfidence =numpy.array(ucb), numpy.array(ucbwConfidence)

    thompson_diff_i = woRejection.mean(axis=0)[-1] - wRejection.mean(axis=0)[-1]
    ucb_diff_i = ucb.mean(axis=0)[-1] - ucbwConfidence.mean(axis=0)[-1]

    thompson_diff_regret.append(abs(thompson_diff_i))
    ucb_diff_regret.append(abs(ucb_diff_i))


fig_1 = plt.figure()
fig_1.set_size_inches(5,5)
ax = fig_1.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(width, delta, thompson_diff_regret, cmap=cm.jet, linewidth=0)
# ax.pcolor(width, delta, thompson_diff_regret, cmap=cm.jet, linewidth=0)
# plt.title('Regret Difference for TS')
plt.xlabel('Estimation error $\epsilon$', fontsize=13,labelpad=10)
plt.ylabel('Gap between arms $\Delta$', fontsize=13, labelpad=10)
ax.grid(False)
ax.set_zticks([])
ax.view_init(89.9, -90.1)
fig_1.savefig('figures/TS-comparison.png')


fig_2 = plt.figure()
ax = fig_2.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(width, delta, ucb_diff_regret, cmap=cm.jet, linewidth=0)
plt.xlabel('Estimation error $\epsilon$', fontsize=13,labelpad=10)
plt.ylabel('Gap between arms $\Delta$', fontsize=13, labelpad=10)
ax.grid(False)
ax.set_zticks([])
ax.view_init(89.9, -90.1)
fig_2.savefig('figures/UCB-comparison.png')
