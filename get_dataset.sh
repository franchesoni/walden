python jpegs_to_tiles.py out/img1/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img2/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img3/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img4/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img5/ --reset --parallel --small_tile_size=256 --small_tile_step=256
python jpegs_to_tiles.py out/img6/ --reset --parallel --small_tile_size=256 --small_tile_step=256

mkdir slides_dataset
mkdir slides_dataset/img1
cp -r out/img1/tiles slides_dataset/img1/tiles
cp out/img1/composite_image.jpeg out/img1/extracted_metadata.txt slides_dataset/img1/

mkdir slides_dataset/img2
cp -r out/img2/tiles slides_dataset/img2/tiles
cp out/img2/composite_image.jpeg out/img2/extracted_metadata.txt slides_dataset/img2/

mkdir slides_dataset/img3
cp -r out/img3/tiles slides_dataset/img3/tiles
cp out/img3/composite_image.jpeg out/img3/extracted_metadata.txt slides_dataset/img3/

mkdir slides_dataset/img4
cp -r out/img4/tiles slides_dataset/img4/tiles
cp out/img4/composite_image.jpeg out/img4/extracted_metadata.txt slides_dataset/img4/

mkdir slides_dataset/img5
cp -r out/img5/tiles slides_dataset/img5/tiles
cp out/img5/composite_image.jpeg out/img5/extracted_metadata.txt slides_dataset/img5/

mkdir slides_dataset/img6
cp -r out/img6/tiles slides_dataset/img6/tiles
cp out/img6/composite_image.jpeg out/img6/extracted_metadata.txt slides_dataset/img6/

tar -cf slides_dataset.tar slides_dataset/