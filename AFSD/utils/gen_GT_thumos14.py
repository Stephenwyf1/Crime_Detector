import json

with open("thumos_annotations/thumos_gt.json", 'r') as fobj:
    database = json.load(fobj)
data = database['database']
abnormalList = ["CleanAndJerk", "BaseballPitch", "CricketBowling", "FrisbeeCatch", "JavelinThrow", "SoccerPenalty",
                  "Diving", "CliffDiving", "CricketShot"]
for y in data:
    video_level_anno = data[y]['annotations']
    for i in video_level_anno:
        if i['label'] in abnormalList:
            i['label'] = 'abnormal'
        else:
            i['label'] = 'normal'

with open("thumos_annotations/thumos_gt_VAD.json", 'w') as vad:
    json.dump(database, vad)
