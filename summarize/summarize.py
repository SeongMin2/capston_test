import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a',type=int,default=1)
    parser.add_argument('--b',type=int,default=2)
    args = parser.parse_args()

    c = args.a + args.b

    print(c)
