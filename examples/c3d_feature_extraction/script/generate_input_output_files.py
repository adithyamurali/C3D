import argparse
import IPython
parser = argparse.ArgumentParser()
parser.add_argument("input_list_name", help = "input list to be populated")
parser.add_argument("input_folder", help = "Location of input images")
parser.add_argument("output_list_name", help = "ouput list to be populated")
parser.add_argument("output_folder", help = "Location of output features")
parser.add_argument("num_minutes", help = "Number of 16-frame batches")
parser.add_argument("fps", help = "Frame rate - Frames per second", default = 1)

args = parser.parse_args()

input_list_path = "input/frm/"
output_list_path = "output/c3d/"
prototxt_path = "../prototxt/"

input_list = open(prototxt_path + args.input_list_name, 'w')
output_list = open(prototxt_path + args.output_list_name, 'w')
i = 1
fps = int(args.fps)
while i < (int(args.num_minutes) - 16):
	input_list.write(input_list_path + args.input_folder + "/ " + str(i) + " 0\n")
	output_list.write(output_list_path + args.input_folder + "/"+ str(i)+"\n")
	i += fps