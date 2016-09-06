#!/bin/bash

# download and unpack data for assignment 1
mkdir data
cd data
wget https://static.dato.com/files/coursera/course-4/people_wiki.gl.zip
wget https://static.dato.com/files/coursera/course-4/people_wiki_map_index_to_word.gl.zip
wget https://static.dato.com/files/coursera/course-4/people_wiki_word_count.npz
wget https://static.dato.com/files/coursera/course-4/people_wiki_tf_idf.npz
wget https://static.dato.com/files/coursera/course-4/kmeans-arrays.npz

unzip people_wiki.gl.zip
unzip people_wiki_map_index_to_word.gl.zip
