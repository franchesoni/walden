
### Installation
- clone the repo
- download bftools (bioformats command line tools)
- unzip it
- put the folder inside the repo
- download the necessary sam checkpoints 

requires installing openjdk (e.g. using conda), ask chatgpt for help if needed

### Data processing pipeline
[x] receive image path
[x] extract metadata (dimensions, z planes)
[x] extract a center crop of the image 4096 x 4096 (at all z planes) and save it
[x] load the center z planes and compute the sharpness for each plane to select the consecutive levels with best sharpness (for now take only one)
[x] for the selected focus(es), save big tiles 16384 x 16384 in jpeg
[x] load and downsample each tile to create a 16x downsampled composite image
_now we can zip the out/ folder and transfer it elsewhere, as it's less than 2GB_

[x] extract overlapping tiles 
- run SAM2 over them, saving the masks mostly inside a center crop
[x] compute average dinov2reg feats for each mask, saving them along with the mask bounding box coordinates (absolute)
[x] unsupervised learning using kmeans
[x] understand import / export annotation qupath (export in geojson at file / export and import similarly), get classes from imported files, add points and save points as tsv
[x] visualize cell classification in qupath (visualize img + bbox with class)
[] do some example annotations using qupath (e.g. points), import them
[] assign points to masks, run k-nn
[] return counts


### Notes
- zipping the images is not worth it

### Future
- run 
- do focus stacking
- tune SAM2


