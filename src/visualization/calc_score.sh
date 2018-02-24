NAME="model-$2-clustering-$3.txt" #.sjis

DIST="data/processed/csj/$1"
ANSWER="data/external/CJLC-0.1/$1.txt"
RECOGNITION_RESULT_PATH=$DIST/recognition.txt.utf8
RE_RECOGNITION_RESULT_PATH=$DIST/re_recognition-$NAME

nkf -w $RE_RECOGNITION_RESULT_PATH > $RE_RECOGNITION_RESULT_PATH.utf8

RE_RECOGNITION_RESULT_PATH=$RE_RECOGNITION_RESULT_PATH.utf8

python3 src/visualization/shaping.py $RECOGNITION_RESULT_PATH > $RECOGNITION_RESULT_PATH'.shaped'
mecab -O wakati $ANSWER > $ANSWER.wakati  # without | kakasi -JH -i utf8 | nkf -w
#mecab -O wakati $ANSWER > $ANSWER.kanji"
mecab -O wakati $RECOGNITION_RESULT_PATH".shaped" > $RECOGNITION_RESULT_PATH".shaped.wakati" # without | kakasi -JH -p -i utf8 | nkf -w
#mecab -O wakati $RECOGNITION_RESULT_PATH".shaped" > $RECOGNITION_RESULT_PATH".shaped.kanji"
python3 src/visualization/calc_score.py $ANSWER.wakati $RECOGNITION_RESULT_PATH".shaped.wakati" $DIST > $DIST/"result_before.txt"
echo 'save at '$DIST/"result_before-$NAME"

python3 src/visualization/shaping.py $RE_RECOGNITION_RESULT_PATH > $RE_RECOGNITION_RESULT_PATH'.shaped'
mecab -O wakati $RE_RECOGNITION_RESULT_PATH".shaped" > $RE_RECOGNITION_RESULT_PATH".shaped.wakati" #(ry
mecab -O wakati $RE_RECOGNITION_RESULT_PATH".shaped" > $RE_RECOGNITION_RESULT_PATH".shaped.kanji"
python3 src/visualization/calc_score.py $ANSWER.wakati $RE_RECOGNITION_RESULT_PATH".shaped.wakati" $DIST > $DIST/"result_after.txt"
echo 'save at '$DIST/"result_after-$NAME"
