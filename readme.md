
### Installation
- clone the repo
- download bftools (bioformats command line tools)
- unzip it
- put the folder inside the repo

requires installing openjdk (e.g. using conda), ask chatgpt for help if needed

### Data processing pipeline
- receive image path
- extract metadata (dimensions, z planes)
- extract a few random crops of the image (at all z planes)
- computes the sharpness for each plane
- take the consecutive levels with the best sharpness
- visualize them

### Notes

- zipping the images is not worth it

