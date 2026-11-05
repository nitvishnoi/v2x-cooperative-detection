from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from matplotlib.figure import figaspect
import matplotlib as mpl

import numpy as np


stats_file = 'data/usyd-v2x/usyd_dataset_stats.pkl'
with open(stats_file, 'rb') as f:
    dataset_stats = pickle.load(f)

colors_for_classes = {
    'car': '#604CC3',
    'bike': '#80C4E9',
    'ped': '#FF7F3E'
}

names_for_classes = {
    'car': 'Vehicle',
    'bike': 'Bike',
    'ped': 'Pedestrian'
}

# fontsize = 10

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 24}

mpl.rc('font', **font)


class CustomColorMap(object):
    def __init__(self) -> None:
        self.car_hex_list = ['#FFFFFF', '#dedaf3', '#c2bae8', '#aca1e0', '#988cd8', '#8778d2', '#7463cb', '#604CC3']
        self.bike_hex_list = ['#FFFFFF', '#dceff9', '#acd8f1', '#85c7ea', '#76bfe7', '#67b9e5', '#58b2e2', '#49abe0']
        self.ped_hex_list = ['#FFFFFF', '#ffdfcf', '#ffc6a9', '#ffa578', '#ff9a66', '#ff8d52', '#ff7f3e', '#ff7128']
        self.dict_cmp = {
            'car': self.get_continuous_cmap(self.car_hex_list),
            'bike': self.get_continuous_cmap(self.bike_hex_list),
            'ped': self.get_continuous_cmap(self.ped_hex_list),
        }
        

    def hex_to_rgb(self, value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


    def rgb_to_dec(self, value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v/256 for v in value]
    
    def get_continuous_cmap(self, hex_list, float_list=None):
        ''' creates and returns a color map that can be used in heat map figures.
            If float_list is not provided, colour map graduates linearly between each color in hex_list.
            If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
            
            Parameters
            ----------
            hex_list: list of hex code strings
            float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
            
            Returns
            ----------
            colour map'''
        rgb_list = [self.rgb_to_dec(self.hex_to_rgb(i)) for i in hex_list]
        if float_list:
            pass
        else:
            float_list = list(np.linspace(0,1,len(rgb_list)))
            
        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp



def make_fig_classes_distribution_polar_coord(save_to_dir: str, partition: str):
    classes_distribution_polar_coord = \
        dataset_stats['classes_distribution_polar_coord'][partition]
    
    save_to_dir = Path(save_to_dir)
    save_to_dir.mkdir(exist_ok=True)

    w, h = figaspect(1/1)

    colormaps = CustomColorMap()

    for cls_name, cls_info in classes_distribution_polar_coord.items():
        print(f"cls_name: {cls_name}")
        r = cls_info['r']
        theta = cls_info['theta']

        # define binning
        r_bins = np.arange(0, 70.0, 3.0)
        theta_bins = np.linspace(-np.pi, np.pi, 60)    
    
        #calculate histogram
        hist, _, _ = np.histogram2d(theta, r, bins=(theta_bins, r_bins))
        hist = hist.astype(float) / float(len(r))
        assert hist.sum() > 0.9, f"hist.sum() = {hist.sum()}"

        thth, rr = np.meshgrid(theta_bins, r_bins)

        fig, ax = plt.subplots(figsize=(2*h, 2*w), subplot_kw=dict(projection="polar"))
        

        print('hist min: ', np.min(hist))
        print('hist max: ', np.max(hist))
        norm = mpl.colors.PowerNorm(gamma=0.25)
        pc = ax.pcolormesh(thth, rr, hist.T, cmap=colormaps.dict_cmp[cls_name], norm=norm)

        # ax.set_title(f'{cls_name_to_full[cls_name]}')
        ax.set_yticks([25, 50, 70], labels=['25m', '50m', '70m'])
        ax.tick_params(axis='x', which='major', pad=17)

        cbar = fig.colorbar(pc, orientation="horizontal", pad=0.1)

        colorbar_ticks = np.linspace(0, hist.max(), 3)
        cbar.set_ticks(colorbar_ticks)
        cbar.set_ticklabels([f"{np.round(val * 100, 1)}%" for val in colorbar_ticks])

        fig.tight_layout()
        # fig.savefig(save_to_dir / f"polar_distrib_{cls_name}_{partition}.svg", format = 'svg', dpi=300)
        fig.savefig(save_to_dir / f"polar_distrib_{cls_name}_{partition}.pdf", bbox_inches='tight')

    # plt.show()


def make_fig_classes_heading(save_to_dir: str, partition: str):
    def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
        """Draws a bar plot with multiple bars per data point.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            The axis we want to draw our plot on.

        data: dictionary
            A dictionary containing the data we want to plot. Keys are the names of the
            data, the items is a list of the values.

            Example:
            data = {
                "x":[1,2,3],
                "y":[1,2,3],
                "z":[1,2,3],
            }

        colors : array-like, optional
            A list of colors which are used for the bars. If None, the colors
            will be the standard matplotlib color cyle. (default: None)

        total_width : float, optional, default: 0.8
            The width of a bar group. 0.8 means that 80% of the x-axis is covered
            by bars and 20% will be spaces between the bars.

        single_width: float, optional, default: 1
            The relative width of a single bar within a group. 1 means the bars
            will touch eachother within a group, values less than 1 will make
            these bars thinner.

        legend: bool, optional, default: True
            If this is set to true, a legend will be added to the axis.
        """

        # Check if colors where provided, otherwhise use the default color cycle
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Number of bars per group
        n_bars = len(data)

        # The width of a single bar
        bar_width = total_width / n_bars

        # List containing handles for the drawn bars, used for the legend
        bars = []

        # Iterate over all data
        for i, (name, values) in enumerate(data.items()):
            # The offset in x direction of that bar
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

            # Draw a bar for every value of that type
            for x, y in enumerate(values):
                bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

            # Add a handle to the last drawn bar, which we'll need for the legend
            bars.append(bar[0])

        # Draw legend if we need
        if legend:
            ax.legend(bars, data.keys())



    save_to_dir = Path(save_to_dir)
    save_to_dir.mkdir(exist_ok=True)

    classes_heading = dataset_stats['classes_heading'][partition]
    xticks = [0.05, 0.1, 0.17]
    
    w, h = figaspect(1/1)
    scale = 2
    w *= scale
    h *= scale
    fig, ax = plt.subplots(figsize=(h, w), subplot_kw=dict(projection="polar"))  # subplot_kw=dict(projection="polar")
    theta_bins = np.linspace(-np.pi, np.pi, 60)
    
    n_bars = theta_bins.shape[0]
    width = float(w) / float(n_bars)  # width of bar
    
    i = 0
    
    for cls_name, cls_info in classes_heading.items():
        print(f"cls_name: {cls_name}")    
        hist, _  = np.histogram(cls_info, bins=theta_bins)
        hist = hist / float(len(cls_info))
        assert hist.sum() > 0.99, f"hist.sum() = {hist.sum()}"

        x_offset = (i - n_bars / 2) * width + width / 2

        ax.bar(theta_bins[:-1], 
               hist, 
               width=width,
               label=names_for_classes[cls_name], 
               color=colors_for_classes[cls_name], 
               alpha=0.75)
        
        i += 1

        # ax.hist(cls_info, theta_bins, 
        #         label=names_for_classes[cls_name], 
        #         color=colors_for_classes[cls_name],
        #         weights=np.diff(theta_bins),
        #         density=True,
        #         alpha=0.5)
    ax.set_ylim([0, 0.17])
    ax.set_yticks(xticks, labels=[f"{int(val * 100.0)}%" for val in xticks])
    ax.tick_params(axis='x', which='major', pad=17)
    ax.legend(loc=(-0.1, -0.2), ncol=3)
    # fig.savefig(save_to_dir / f"heading_{cls_name}_{partition}.svg", format = 'svg', dpi=300)

    fig.tight_layout()
    fig.savefig(save_to_dir / f"heading_all_three_{partition}.pdf", bbox_inches='tight')

    # plt.show()


def make_size_distribution(save_to_dir: str, partition: str):
    save_to_dir = Path(save_to_dir)
    save_to_dir.mkdir(exist_ok=True)

    classes_size = dataset_stats['classes_size'][partition]

    max_dx, max_dy, max_dz = 0, 0, 0
    for cls_name, cls_info in classes_size.items():
        max_dx = max(max_dx, np.max(cls_info['dx']))
        max_dy = max(max_dy, np.max(cls_info['dy']))
        max_dz = max(max_dz, np.max(cls_info['dz']))
    
    max_dx = 8.0
    max_size = (max_dx, max_dy, max_dz)

    name_size = ['Length', 'Width', 'Height']

    w, h = figaspect(2/1)

    for ii, type_size in enumerate(('dx', 'dy', 'dz')):
        fig, ax = plt.subplots(figsize=(h, w))    
        bins = np.linspace(0, max_size[ii], 20)
        for cls_name, cls_info in classes_size.items():
            current_size = cls_info[type_size]
            hist, _ = np.histogram(current_size, bins=bins)
            hist = 100. * hist / float(len(current_size))
            assert hist.sum() > 0.99, f"hist.sum() = {hist.sum()}"
            ax.plot(
                bins[:-1], hist, color=colors_for_classes[cls_name], 
                label=names_for_classes[cls_name])
            ax.fill_between(bins[:-1], 0, hist, alpha=0.3, color=colors_for_classes[cls_name])

        ax.set_xlabel(f'{name_size[ii]} (m)')
        ax.set_yticks(
            np.arange(0, 100, 20), [f"{i}%" for i in np.arange(0, 100, 20)])
        ax.set_ylim([0, 80])
        ax.legend(loc='upper right')
        fig.tight_layout()
        # fig.savefig(save_to_dir / f"size_distrib_{type_size}_{partition}.svg", format = 'svg', dpi=300)
        fig.savefig(save_to_dir / f"size_distrib_{type_size}_{partition}.pdf", bbox_inches='tight')


def make_class_distribution_in_traintest(save_to_dir: str):
    save_to_dir = Path(save_to_dir)
    save_to_dir.mkdir(exist_ok=True)

    classes_by_splits = dataset_stats['classes_by_splits']
    train = classes_by_splits['train']
    test = classes_by_splits['test']

    color_train = "#70d6ff"
    color_test = "#ff9770"
    
    x_labels = ['Vehicle', 'Bike', 'Pedestrian']
    x = np.arange(len(x_labels))
    width = 0.4 # width of bar

    w, h = figaspect(5/2)
    fig, ax = plt.subplots(figsize=(int(h * 1.5), int(w * 1.5)))

    ax.bar(x, [train['num_cars'], train['num_bikes'], train['num_peds']], width, color=color_train, label='Train')
    ax.bar(x + width, [test['num_cars'], test['num_bikes'], test['num_peds']], width, color=color_test, label='Test')

    ax.set_yscale('log')
    ax.set_xticks(x + width / 2.0, x_labels)
    ax.set_ylabel('Number of boxes')
    ax.legend(loc='upper right')
    fig.savefig(save_to_dir / f"class_distribution_in_traintest.pdf", bbox_inches='tight')
    # plt.show()

    num_labels = np.array([train['num_cars'], train['num_bikes'], train['num_peds']] + [test['num_cars'], test['num_bikes'], test['num_peds']])
    print('num labels: ', np.sum(num_labels))

    percentage_vru = np.array([train['num_bikes'], train['num_peds'], 
                               test['num_bikes'], test['num_peds']]).sum() / float(np.sum(num_labels))
    print('percentage VRU: ', percentage_vru)


def medium():
    def hex_to_rgb(value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


    def rgb_to_dec(value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v/256 for v in value]
    
    def get_continuous_cmap(hex_list, float_list=None):
        ''' creates and returns a color map that can be used in heat map figures.
            If float_list is not provided, colour map graduates linearly between each color in hex_list.
            If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
            
            Parameters
            ----------
            hex_list: list of hex code strings
            float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
            
            Returns
            ----------
            colour map'''
        rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
        if float_list:
            pass
        else:
            float_list = list(np.linspace(0,1,len(rgb_list)))
            
        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    hex_list = ['#e9e6f7', '#cec8ed', '#bab2e5', '#a69bde', '#9486d6', '#8373d0', '#715fc9', '#604CC3']

    x, y = np.mgrid[-5:5:0.05, -5:5:0.05]
    z = (np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2))

    fig, ax = plt.subplots(1,1)
    im = ax.imshow(z, cmap=get_continuous_cmap(hex_list))
    fig.colorbar(im)
    ax.yaxis.set_major_locator(plt.NullLocator()) # remove y axis ticks
    ax.xaxis.set_major_locator(plt.NullLocator()) # remove x axis ticks
    plt.show()


def main():
    # make_fig_classes_distribution_polar_coord('figs_for_paper', 'all')
    make_fig_classes_heading('figs_for_paper', 'all')
    # make_size_distribution('figs_for_paper', 'all')
    # make_class_distribution_in_traintest('figs_for_paper')

    # medium()


if __name__ == '__main__':
    main()
