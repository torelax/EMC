from copy import deepcopy
import sys

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    print(params)
    print(__file__)