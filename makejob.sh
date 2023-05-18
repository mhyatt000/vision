# makes job files that you can submit

file_name="$1"
new_file_name="${file_name#configs/}"
new_file_name="${file_name%.yaml}.sh"

sed  "s|<yaml>|$1|g" multinode.sh > "jobscripts/$new_file_name"
