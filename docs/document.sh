#!/bin/bash
# used to make and update sphinx documentation

sphinx-apidoc -o ./source ../general
make html
