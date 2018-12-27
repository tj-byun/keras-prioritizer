# Download EMNIST dataset and retain `emnist-digits.mat` file
wget http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip
unzip matlab.zip
mv matlab/emnist-digits.mat .
rm -r matlab
