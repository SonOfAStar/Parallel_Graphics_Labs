import sys
import random as rnd
import numpy as np


def create_test(dimensions):
    test = np.zeros((dimensions[0], dimensions[1]))
    ans = np.zeros((dimensions[1], dimensions[0]))
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            elem = rnd.random() * 1000
            test[i][j] = elem
            ans[j][i] = elem

    test_name = '.\\tests\\test' + str(dimensions[0]) + 'x' + str(dimensions[1]) + '.txt'
    test_out = open(test_name, 'w')
    ans_name = '.\\answs\\ans' + str(dimensions[0]) + 'x' + str(dimensions[1]) + '.txt'
    ans_out = open(ans_name, 'w')
    test_out.write(str(dimensions[0]) + " " + str(dimensions[1]) + "\n")

    test = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in test)
    ans = '\n'.join('\t'.join('%0.3f' % x for x in y) for y in ans)
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

    if len(argv) == 2:
        create_test(argv)
    else:
        print('Error argv')


if __name__ == '__main__':
    main()