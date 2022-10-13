class Config:
    BATCHSIZE = 200
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=26
    START_EPOCHES=0
    SAVEDIR="./output/char"

class DigitNLConfig:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=9
    START_EPOCHES=0
    SAVEDIR="./output/digit/DigitNLConfig"

class DigitConfig5Point:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=9
    START_EPOCHES=0
    SAVEDIR="./output/DigitConfig5Point"

class DigitConfig6Point:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=9
    START_EPOCHES=0
    SAVEDIR="./output/DigitConfig6Point"

class DigitConfig5Adj:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=9
    START_EPOCHES=0
    SAVEDIR="./output/DigitConfig5Adj"

class DigitConfig6Adj:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=9
    START_EPOCHES=0
    SAVEDIR="./output/DigitConfig6Adj"


class CharConfig5Point:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=5000
    EPOCHES=6
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=26
    START_EPOCHES=0
    SAVEDIR="./output/CharConfig5Point"

class CharConfig6Point:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=26
    START_EPOCHES=0
    SAVEDIR="./output/CharConfig6Point"

class CharConfig5Adj:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=26
    START_EPOCHES=0
    SAVEDIR="./output/CharConfig5Adj"

class CharConfig6Adj:
    BATCHSIZE = 100
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=26
    START_EPOCHES=0
    SAVEDIR="./output/CharConfig6Adj"

class CharConfigResnet:
    BATCHSIZE = 50
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=26
    START_EPOCHES=0
    SAVEDIR="./output/CharConfigResnet"

class CharConfigNeural:
    BATCHSIZE = 50
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=30
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=26
    START_EPOCHES=0
    SAVEDIR="./output/CharConfigNeural"

class CharConfigResnet:
    BATCHSIZE = 50
    PRINT_INTERVAL=1
    SAVS_INTERVAL=1000
    EPOCHES=100
    DEVICE='cuda'
    LR=3E-5
    GAMA=0.1
    NUM_CLASS=26
    START_EPOCHES=0
    SAVEDIR="./output/resnet5m"