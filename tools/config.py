class Config:
    Window_Length=600
    Create_Gesture_Length = 180
    FLEX_PORT="com4"
    UHAND_PORT="com7"
    FREQUENCY=9600
    UPDATE_TIME_INTERVAL=20
    UHAND_CONNECTION_TIMEOUT=0.2
    Char_MLPMIXER_WEIGHT=r".\ai\output\weights_classify\finalcharTrans4m5p.pth"
    Digit_MLPMIXER_WEIGHT=r".\ai\output\weights_classify\finaldigitTrans4m5p.pth"

    TEXT_CHARS='./outputpic/train.txt'
    PRECHAR_WEIGHTS="./ai/output/weights_prechar/twolayLSTM_params.pth"
    #上位机存储目录
    IMAGE_FOLDER="../data/Image/"    
    FLEX_FOLDER="../data/flexSensor/"
    RUNTEMP_BASEFOLDER="../data/temp/"
    FaceFolder="../data/face"
    ValidationFile="../data/validation/"

    HANDLENGTH=180
    UPTIME_INTERVAL=20
    