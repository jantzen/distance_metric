# file: visualize.py

import numpy as np
from matplotlib import pyplot as plt
import eugene as eu
from moment_analysis_with_tuning_and_LOO import load_data


def plot_moments(data):
    # get data range
    ymin = 0.
    ymax = 0.
    for key in sorted(data.keys()):
        lo = np.min(data[key])
        hi = np.max(data[key])
        if lo < ymin:
            ymin = lo
        if hi > ymax:
            ymax = hi
    plts = []
    plts.append(plt.figure())
    num_subplots = len(data.keys())
    rows = round(num_subplots / 2.)
    counter = 1
    for key in sorted(data.keys()):
        plt.subplot(rows, 2, counter)
        plt.plot(data[key][0,:], 'b-')
        plt.plot(data[key][1,:], 'g-')
        plt.ylim([ymin, ymax])
        plt.title(key, fontsize=48)
        plt.legend(['3rd moment', '4th moment'], fontsize=24)
        counter += 1
    plts.append(plt.figure())
    legend_txt = []
    for key in sorted(data.keys()):
        plt.plot(data[key][0,:], alpha=0.4)
        legend_txt.append(key)
    plt.legend(legend_txt, fontsize=24)
    plt.title('All events, 3rd moment', fontsize=48)
    plts.append(plt.figure())
    legend_txt = []
    for key in sorted(data.keys()):
        plt.plot(data[key][1,:], alpha=0.4)
        legend_txt.append(key)
    plt.legend(legend_txt, fontsize=24)
    plt.title('All events, 4th moment', fontsize=48)
    return plts


def plot_initials(data, frags, reps, colors, mu_spec):
    plts = []
    moments_norm_list = []
    labels = []
    for key in sorted(data.keys()):
        moments_norm_list.append(data[key])
        labels.append(key)

    fragmented = eu.fragment_timeseries.split_timeseries(moments_norm_list, frags)
    plts.append(plt.figure())
    for ii, event in enumerate(fragmented):
        initials_x = []
        initials_y = []
        for fragment in event:
            initials_x.append(fragment[0,0])
            initials_y.append(fragment[1,0])
        initials_x = np.array(initials_x)
        initials_y = np.array(initials_y)
        plt.scatter(initials_x, initials_y, c=colors[ii], s=200, alpha=0.4, edgecolors='none')

    untrans, trans, error_flag = eu.initial_conditions.choose_untrans_trans(fragmented, reps, alpha=0.3,
            beta=0.1, mu_spec=mu_spec, report=True)
    for ii, event in enumerate(untrans):
        initials_x = []
        initials_y = []
        for fragment in event:
            initials_x.append(fragment[0,0])
            initials_y.append(fragment[1,0])
        initials_x = np.array(initials_x)
        initials_y = np.array(initials_y)
        plt.scatter(initials_x, initials_y, c=colors[ii], s=200, alpha=1.0, marker='+')
    for ii, event in enumerate(trans):
        initials_x = []
        initials_y = []
        for fragment in event:
            initials_x.append(fragment[0,0])
            initials_y.append(fragment[1,0])
        initials_x = np.array(initials_x)
        initials_y = np.array(initials_y)
        plt.scatter(initials_x, initials_y, c=colors[ii], s=200, alpha=1.0, marker='x')

    plt.title("Initial values for {} fragments.".format(frags), fontsize=48)
    plt.legend(labels, fontsize=24)

    return plts


def main(
        events=['07', '08', '09', '10', '12', '13']
        ):
    data = dict([])
    # plot data
    for ev in events:
        data['event' + ev] = np.loadtxt('./processed/moments_norm_' + ev)
    plot_moments(data)
    # plot initials
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
            'salmon']
    mu_untrans = np.array([0., 2.2]).reshape(-1,1)
#    mu_trans = np.array([1.0, 3.7]).reshape(-1,1)
#    mu_trans = np.array([0.45, 2.51]).reshape(-1,1)
#    mu_spec = [mu_untrans, mu_trans]
    mu_spec = None
    # optimal
#    plot_initials(data, 862, 10, colors)
    frags = 470
    reps = 10
    plot_initials(data, frags, reps, colors, mu_spec)
#    #suboptimal
##    plot_initials(data, 191, 10, colors)
#    frags = 862
#    reps = 10
#    plot_initials(data, frags, reps, colors)
    # plot initials without event 11
    del(data['event13'])
    del(colors[2])
    # optimal
#    plot_initials(data, 862, 10, colors)
    frags = 470
    reps = 10
    plot_initials(data, frags, reps, colors, mu_spec)
    #suboptimal
#    plot_initials(data, 191, 10, colors)
#    frags = 862
#    reps = 10
#    plot_initials(data, frags, reps, colors)
    plt.show()

if __name__ == '__main__':
#    main()
#    main(events=['06', '08', '11', '12', '13','14','15','16','17','18','19'])
    main(events=['07', '08', '09', '10', '12', '13'])
