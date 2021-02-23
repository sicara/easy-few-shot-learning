mkdir data/CUB
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx" -O data/CUB/images.tgz
rm -rf /tmp/cookies.txt
tar  --exclude='._*' -zxvf data/CUB/images.tgz -C data/CUB/