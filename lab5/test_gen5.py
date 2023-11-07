import sys
import random as rnd
import numpy as np


def create_test(dimensions):
    n = dimensions[0]
    if n > (135 * 1000000):
        n = 135 * 1000000
    test = np.zeros((n))

    for i in range(0, n):
        elem = rnd.randint(1, 16777215)
        test[i] = elem

    ans = np.sort(test)

    test_name = '.\\tests\\test' + str(n) + '.txt'
    ans_name = '.\\answs\\ans' + str(n) + '.txt'

    test_out = open(test_name, 'w')
    ans_out = open(ans_name, 'w')
    test_out.write(str(dimensions[0]) + "\n")

    test = '\t'.join('%d' % y for y in test)
    ans = '\t'.join('%d' % y for y in ans)
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