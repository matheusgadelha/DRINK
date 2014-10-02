img_files=`ls $1 | grep .ppm`

test_name=`echo $1 | cut -d'/' -f 2`

for file in $img_files 
do 
	match_file=`echo $file | cut -c1-4`	
	match_file="${test_name}_${match_file}.matches"
	./bin/tst-desc $1'img1.ppm' $1$file > $match_file
done


