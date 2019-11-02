import argparse

parser = argparse.ArgumentParser(description='Test an image.')
parser.add_argument('image', type=int, nargs=1, help='name of image')

args = parser.parse_args()

print(args.image[0])