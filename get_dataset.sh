python jpegs_to_tiles.py out/img1/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img2/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img3/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img4/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img5/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img6/ --reset --parallel --small_tile_size=256 --small_tile_step=256

mkdir dataset
mkdir dataset/img1
cp -r out/img1/tiles dataset/img1/tiles
cp out/img1/composite_image.jpeg out/img1/extracted_metadata.txt dataset/img1/

mkdir dataset/img2
cp -r out/img2/tiles dataset/img2/tiles
cp out/img2/composite_image.jpeg out/img2/extracted_metadata.txt dataset/img2/

mkdir dataset/img3
cp -r out/img3/tiles dataset/img3/tiles
cp out/img3/composite_image.jpeg out/img3/extracted_metadata.txt dataset/img3/

mkdir dataset/img4
cp -r out/img4/tiles dataset/img4/tiles
cp out/img4/composite_image.jpeg out/img4/extracted_metadata.txt dataset/img4/

mkdir dataset/img5
cp -r out/img5/tiles dataset/img5/tiles
cp out/img5/composite_image.jpeg out/img5/extracted_metadata.txt dataset/img5/

mkdir dataset/img6
cp -r out/img6/tiles dataset/img6/tiles
cp out/img6/composite_image.jpeg out/img6/extracted_metadata.txt dataset/img6/

tar -cf slides_dataset.tar dataset/
rm -rf dataset/