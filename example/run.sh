#!/bin/sh

cd "$(dirname "$0")"

printf '\n> Running example:\n\n'
CGO_ENABLED=1 go run . "$@"
