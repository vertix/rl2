pushd `dirname $0` > /dev/null
SCRIPTDIR=`pwd`
popd > /dev/null

java -Xms2G -Xmx8G -cp ".:*:$SCRIPTDIR/*" -jar repeater.jar $1