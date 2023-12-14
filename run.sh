# python main.py --cfg thumos --stage 1 --mode train
# python main.py --cfg thumos --stage 1 --mode test
# python main.py --cfg thumos --stage 2 --mode train
# python main.py --cfg thumos --stage 2 --mode test
python main.py --cfg thumos --stage 1 --mode train --seed 6
python main.py --cfg thumos --stage 1 --mode test  --seed 6
python main.py --cfg thumos --stage 2 --mode train  --seed 0
python main.py --cfg thumos --stage 2 --mode test   --seed 0