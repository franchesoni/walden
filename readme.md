
### Installation
- clone the repo
- download bftools (bioformats command line tools)
- unzip it
- put the folder inside the repo

requires installing openjdk (e.g. using conda), ask chatgpt for help if needed

### Data processing pipeline
- receive image path
- extract metadata (dimensions, z planes)
- extract a center crop of the image 4096 x 4096 (at all z planes) and save it
- load the center z planes and compute the sharpness for each plane to select the consecutive levels with best sharpness (for now take only one)
- for the selected focus(es), save big tiles 16384 x 16384 in jpeg
- load and downsample each tile to create a 16x downsampled composite image
_now we can zip the out/ folder and transfer it elsewhere, as it's less than 2GB_

- extract overlapping tiles 
- run SAM2 over them, saving the masks far away from the border 
- compute average dinov2reg feats for each mask, saving them along with the mask bounding box coordinates (absolute)
- do some example annotations using qupath (e.g. points), import them
- assign points to masks, run k-nn
- return counts, visualize img + bbox with class


### Notes

- zipping the images is not worth it

### Future
- do focus stacking
- tune SAM2


