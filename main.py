import torch


# ! main file for driver code


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="sir", choices=["sir"], help='which model to use for epidemic simulation (default: %(default)s)')
    parser.add_argument('--S0', type=int, default=1, help='initial condition for [S]usceptible (default: %(default)s)')
    parser.add_argument('--I0', type=int, default=1, help='initial condition for [I]nfected (default: %(default)s)')
    parser.add_argument('--R0', type=int, default=1, help='initial condition for [R]ecovered (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1e-3, help='infection likelihood parameter (default: %(default)s)')
    parser.add_argument('--gamma', type=float, default=1e-3, help='recovery rate parameter (default: %(default)s)')
    args = parser.parse_args()

    print('\n# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
    print("")