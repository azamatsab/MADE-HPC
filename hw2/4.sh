#!/bin/bash

FILE=Linux
if test -f "$FILE"; then
    echo "course"
else
    echo "very easy"
    echo "course is easy" > $FILE
fi