pip install -r requirements.txt
python main.py --dataset-path ./ --serialization-path ./tb/stacked_bpe --optimizer radam --drop 0.2 --batch 2 --bpe
python main.py --dataset-path ./ --serialization-path ./tb/stacked --optimizer radam --drop 0.2 --batch 2

for i in $(seq 0.01 0.01 0.15)
    do
    python mistakes_validation.py --dataset-path ./test_data.csv --file logs.txt --model-path ./tb/stacked/best.th --vocabulary-path ./tb/stacked/vocabulary --mistakes-rate $i --random-seed 10
    python mistakes_validation.py --dataset-path ./test_data.csv --file logs.txt --model-path ./tb/stacked_bpe/best.th --vocabulary-path ./tb/stacked_bpe/vocabulary --mistakes-rate $i --random-seed 10 --bpe --bpe-path ./bpe.model
 done

