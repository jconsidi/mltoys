#!/bin/sh

set -e

cd `dirname $0`

for EXAMPLE in $(ls example-*.py)
do
    "./$EXAMPLE"
    echo ''
done
