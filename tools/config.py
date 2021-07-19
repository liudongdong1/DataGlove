class Config:
    Window_Length=180
    Create_Gesture_Length = 180
    FLEX_PORT="com16"
    UHAND_PORT="com7"
    FREQUENCY=9600
    UPDATE_TIME_INTERVAL=20
    UHAND_CONNECTION_TIMEOUT=0.2
    MLPMIXER_WEIGHT="./ai/output/weights_classify/weights-34-6946-[0.9471].pth"
    TEXT_CHARS='./Tools/pngresult/train.txt'
    PRECHAR_WEIGHTS="./ai/output/weights_prechar/twolayLSTM_params.pth"
    #上位机存储目录
    IMAGE_FOLDER="../data/Image/"    
    FLEX_FOLDER="../data/flexSensor/"
    RUNTEMP_BASEFOLDER="../data/temp/"

    HANDLENGTH=180
    UPTIME_INTERVAL=20
    