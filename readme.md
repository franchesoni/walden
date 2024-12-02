
### Installation
- clone the repo
- download bftools (bioformats command line tools), from the repo: `wget https://downloads.openmicroscopy.org/bio-formats/8.0.1/artifacts/bftools.zip; unzip bftools.zip; rm bftools.zip`
- modify the ram available in `bftools/config.sh`
- download the necessary sam checkpoints 

requires installing openjdk (e.g. using conda), ask chatgpt for help if needed

### Usage
run
- `vsi_to_jpegs.py` to convert the vsi to big jpegs, do it for every vsi
- `jpegs_to_overlapping_tiles.py` (or `get_dataset.sh`) to convert the big jpegs to smaller jpegs

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
[x] do some example annotations using qupath (e.g. points), import them
[x] parallelize <- somewhat, sam2 inference is hard to parallelize
[] extract all bboxes and cells
[] load the supervision and run knn to order the bboxes
[] create an util to display and annotate the sequence of neighbors


### Notes
- zipping the images is not worth it
- in the composite image, in qupath, and in the patches we saved, there are some tiling effects visible

### Future
- run 
- do focus stacking
- tune SAM2


# already run
quyet tmux 1
img1 
weird tmux 0
img2
quyet tmux 2
img5 
weird tmux 1
img3
# running...
quyet tmux 3
img6 


