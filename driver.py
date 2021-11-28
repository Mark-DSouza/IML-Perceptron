import sys

from linearlySeparable_main import linearlySeparable
from nonLinearlySeparable_main import nonLinearlySeparable

configs = {
    1: "linearlySeparable/",
    2: "nonLinearlySeparable/",
    3: "overlapping/",
}

selection = int(sys.argv[1])
chosen_config = configs[selection]

if selection == 1:
    linearlySeparable(configs[1])
elif selection == 2:
    nonLinearlySeparable(configs[2])
else:
    linearlySeparable(configs[3])