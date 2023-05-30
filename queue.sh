# makes job files that you can submit

file_name="$1"
file_name="${file_name#configs/}"
file_name="${HOME}/cs/vision/jobs/${file_name%.yaml}.sh"

echo $file_name

sed "s|<yaml>|${HOME}/cs/vision/${1}|g" multinode.sh > "$file_name"
