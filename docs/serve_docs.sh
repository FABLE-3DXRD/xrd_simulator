# custom bash script to build the docs and put them in a place that is 
# visible to github pages, which unfortunately cannot see into the 
# build folder.
make html
cp -r build/html/* .
echo "Copied all docs to path visible by github pages"