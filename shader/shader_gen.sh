#!/bin/sh

glslangValidator -V correction.comp -o ./spv/correction.comp.spv
glslangValidator -V fft_col.comp -o ./spv/fft_col.comp.spv
glslangValidator -V fft_row.comp -o ./spv/fft_row.comp.spv
glslangValidator -V propagate.comp -o ./spv/propagate.comp.spv
glslangValidator -V ocean.frag -o ./spv/ocean.frag.spv
glslangValidator -V ocean.vert -o ./spv/ocean.vert.spv
