# build the docs as html: then drag and drop the build/html/index.html file in a browser

# remove the directory source/_autosummary
dir_name="source/_autosummary"
if [ -d $dir_name ]; then
  rm -rf $dir_name
fi

make clean
make html
