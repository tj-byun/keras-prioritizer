function run() {
  echo "run $1 $2"
  python kerascov.py exp_entropy emnist mnist_models/emnist_cnn_small_10.h5 $1 $2 2>&1 >> tmp.log
}

run 0 20
run 20 40
run 40 60

return
COUNTER=0
while [ $COUNTER -lt 12 ]; do
  let a=20*COUNTER
  let ct=COUNTER+1
  let b=20*ct
  echo "Running $a $b"
  run $a $b
  let COUNTER=COUNTER+1 
done
