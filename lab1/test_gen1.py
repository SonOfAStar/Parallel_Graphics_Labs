import sys
import random as rnd
import numpy as np


def create_test(dimensions):
    test = np.zeros((dimensions[0]))
    ans = np.zeros((dimensions[0]))
    for i in range(0, dimensions[0]):
        elem = rnd.random() * 1000
        test[i] = elem
        ans[i] = elem * elem

    test_name = '.\\tests\\test' + str(dimensions[0]) + '.txt'
    ans_name = '.\\answs\\ans' + str(dimensions[0]) + '.txt'

    test_out = open(test_name, 'w')
    ans_out = open(ans_name, 'w')
    test_out.write(str(dimensions[0]) + "\n")

    test = '\t'.join('%0.5f' %y for y in test)
    ans = '\t'.join('%0.5f' %y for y in ans)
    test_out.write(test)
    ans_out.write(ans)

    test_out.close()
    ans_out.close()
    return 0


def main():
    argv = []
    alfa = show = False
    for arg in sys.argv[1:]:
        argv.append(int(arg))
        print(arg)

    if len(argv) == 1:
        create_test(argv)
    else:
        print('Error argv')


if __name__ == '__main__':
    main()