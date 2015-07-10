Useful command line arguments:

To read video properties:

exiftool ___

Image/video Pre-processing:
ffmpeg -i <video> -r <fps> <path>/%6d.jpg
ffmpeg -i <input video> -filter:v "crop=640:480:x:y" <output video>
ffmpeg -i <input video> -vf scale=640:480 <output video>

To find total number of files in the folder:

find . -type f | wc -l

matlab -nosplash -nodesktop -nojvm -r "read_binary_blob('/home/animesh/C3D/examples/c3d_feature_extraction/output/c3d/v_ApplyEyeMakeup_g01_c01/000001.conv2a');exit"

Example use of script to generate input_output files:

python generate_input_output_files.py input_list_frm_suturing.txt suturing output_list_prefix_suturing.txt suturing 187