"""Stolen from Felix Fiedler.
"""
import matplotlib as mpl
from pathlib import Path

file_pth = Path(__file__).parent.resolve()

fontsize = 14
legend_fontsize = 12
mpl.rcParams.update({
    'lines.linewidth':1,
    'font.size': fontsize,
    'legend.fontsize': legend_fontsize,
    'axes.titlesize'  : fontsize,
    'axes.labelsize'  : fontsize,
    'xtick.labelsize' : fontsize, 
    'ytick.labelsize' : fontsize,
    'pdf.fonttype' : 42, # get rid of type 3 fonts
    'ps.fonttype' : 42, # get rid of type 3 fonts
    'font.serif': [],
    'axes.titlepad':  10,
    'axes.labelsize': 'medium',
    'figure.figsize': (7.15, 4),
    'figure.autolayout': True,
    'axes.grid': False,
    'lines.markersize': 5,
    'pgf.rcfonts': False,    # don't setup fonts from rc parameters
    'axes.unicode_minus': False,
    "text.usetex": False,     # use inline math for ticks
    "pgf.texsystem" : "xelatex",
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
})


# # Load notation from file. The same notation is used in the paper.
# with open(file_pth.joinpath('notation_LL.tex'), 'r') as f:
#     tex_preamble = f.readlines()

# tex_preamble = ''.join(tex_preamble)

# mpl.rcParams.update({
#     'text.latex.preamble': tex_preamble,
#     'pgf.preamble': tex_preamble,
# })

color = mpl.rcParams['axes.prop_cycle'].by_key()['color']
boxprops = dict(boxstyle='round', facecolor='white', alpha=0.5, pad = .4)