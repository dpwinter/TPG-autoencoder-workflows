import matplotlib.pyplot as plt

colors = ['#2C7BB6', '#D7191C']
labels = ['train', 'validation']

def set_boxplot_color_idx(boxplot, i):
    """Set the color of boxplot elements.
    i: index into colors/labels arrays."""
        
    plt.setp(boxplot['boxes'], color=colors[i])
    plt.setp(boxplot['whiskers'], color=colors[i])
    plt.setp(boxplot['caps'], color=colors[i])
    plt.setp(boxplot['medians'], color=colors[i])
    plt.plot([], c=colors[i], label=labels[i])

def locs_to_labels(locs):
    """Covert list of floats to list of strings."""
    loc_labels = ["{:0.0f}".format(l) for l in locs]
    plt.xticks(locs, loc_labels)
