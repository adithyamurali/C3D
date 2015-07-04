Useful command line arguments:

To read video properties:

exiftool ___

To convert video file to images and control frame rate:

ffmpeg -i Suturing_B004_capture1.avi -r 16 frm_test_16/image-%3d.jpg

To find total number of files in the folder:

find . -type f | wc -l

matlab -nosplash -nodesktop -nojvm -r "read_binary_blob('/home/animesh/C3D/examples/c3d_feature_extraction/output/c3d/v_ApplyEyeMakeup_g01_c01/000001.conv2a');exit"

Example use of script to generate input_output files:

python generate_input_output_files.py input_list_frm_suturing.txt suturing output_list_prefix_suturing.txt suturing 187