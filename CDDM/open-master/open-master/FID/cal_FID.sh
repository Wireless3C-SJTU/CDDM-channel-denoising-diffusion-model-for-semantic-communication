export gpu=0
export real_path=./real_image
export fake_path=./fake_image
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu
