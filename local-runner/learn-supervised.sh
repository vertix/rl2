#!/bin/bash
COUNTER=0

while [  $COUNTER -lt 1000 ]; do
    echo Run number $COUNTER
    let COUNTER=COUNTER+1
    java -Xms512m -Xmx1G -server -jar "local-runner.jar" "local-runner-console-supervised.properties" local-runner-console.default.properties
done
