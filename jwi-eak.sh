#!/usr/bin/env bash
# Jwi-eak launcher: ./jwi-eak.sh [FILE] [options]   (./jwi-eak.sh --help)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec julia --project="$DIR" -t auto "$DIR/Jwi-eak.jl" "$@"
