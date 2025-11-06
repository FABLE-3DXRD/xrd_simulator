from itertools import cycle, islice
import sys
import random
import numpy as np

def select_phases(original_list, N):
    # Create a cycling iterator over the original list
    original_cycle = cycle(original_list)
    
    # Use islice to select N elements from the cycling iterator
    selected_elements = list(islice(original_cycle, N))
    selected_elements.sort()
    return selected_elements

def map_to_range(number,noptions):
    return (number - 1) % noptions + 1

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def printvars(vars):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(vars.items())), key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        
def run_name():
    names = [
    "Dragon", "Phoenix", "Unicorn", "Griffin", "Pegasus",
    "Fairy", "Centaur", "Mermaid", "Basilisk", "Hydra",
    "Chimera", "Hippogriff", "Werewolf", "Elf", "Goblin",
    "Gnome", "Leviathan", "Kelpie", "Cerberus", "Sphinx",
    "Leprechaun", "Yeti", "Thunderbird", "Pixie", "Nymph",
    "Dog", "Cat", "Horse", "Elephant", "Lion",
    "Tiger", "Bear", "Giraffe", "Monkey", "Dolphin",
    "Kangaroo", "Zebra", "Ostrich", "Penguin", "Panda",
    "Rabbit", "Squirrel", "Deer", "Fox", "Wolf",
    "Sheep", "Cow", "Chicken", "Eagle", "Whale"
    ]
    adjectives = [
        "Beautiful", "Brave", "Clever", "Dazzling", "Elegant",
        "Fearless", "Graceful", "Handsome", "Intelligent", "Jolly",
        "Kind", "Lovely", "Magnificent", "Noble", "Optimistic",
        "Playful", "Quick", "Radiant", "Strong", "Talented",
        "Unique", "Vivacious", "Wise", "Zealous", "Adventurous",
        "Bold", "Curious", "Determined", "Energetic", "Friendly",
        "Generous", "Happy", "Imaginative", "Joyful", "Lively",
        "Mischievous", "Nutty", "Outgoing", "Polite", "Quirky",
        "Resourceful", "Sociable", "Trustworthy", "Unwavering",
        "Vibrant", "Witty", "Youthful", "Zeal", "Adorable", "Blissful"
    ]
    return random.sample(adjectives,1)[0] + '_' + random.sample(names,1)[0]


def plot_3d_surface(data,q):
    """
    Plot a 3D surface from the given data.
    
    Parameters:
        data (numpy.ndarray): 2D array representing the surface data.
    
    Returns:
        None
    """

    # Define the x and y coordinates
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])

    # Create meshgrid for x and y coordinates
    X, Y = np.meshgrid(x, y)

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(x=q.T, y=Y, z=data.T)])

    # Update layout
    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis=dict(title='X coordinate', range=[0, q.max()]),
            yaxis=dict(title='Y coordinate', range=[0, data.shape[1]-1]),
            zaxis=dict(title='Value'),
        )
    )
    # Show the plot
    fig.show()

def interval(array, x_interval, y_interval):
    mask = (array[:, 1] >= x_interval[0]) & (array[:, 1] <= x_interval[1]) & \
           (array[:, 0] >= y_interval[0]) & (array[:, 0] <= y_interval[1])
    return mask