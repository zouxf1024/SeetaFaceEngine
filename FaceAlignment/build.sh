
CUR_DIR="$(pwd)"
export SDK_ROOT="$CUR_DIR/../"

logfile="/dev/null"
check_cmd(){
    "$@" >> $logfile 2>&1
}
check_cc(){
  check_cmd arm-linux-gnueabihf-gcc -v
}

check_cc
if [ $? -eq 127 ];then
        export PATH=$PATH:$SDK_ROOT/prebuilts/toolschain/bin
fi

echo "$SDK_ROOT"

[ -d build ] && rm -rf build 2>>/dev/null
mkdir build

cd build
cmake ../
make -j4
cd ../

