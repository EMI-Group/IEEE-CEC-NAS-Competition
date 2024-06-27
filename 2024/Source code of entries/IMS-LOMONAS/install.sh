pip install -r requirements.txt

mkdir -p cec

gdown https://drive.google.com/uc?id=11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO -O ./cec/database.zip
gdown https://drive.google.com/uc?id=1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp -O ./cec/data.zip

unzip ./cec/database.zip -d ./cec/database
unzip ./cec/data.zip -d ./cec/data

cp -R ./cec/database/database ./database
cp -R ./cec/data/data20240229/data ./data

rm -r cec

mkdir -p exp