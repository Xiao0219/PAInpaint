#!/bin/bash

# 打包整个推理代码目录
cd ..
tar -czvf PAInpaint_inference.tar.gz PAInpaint_inference/

echo "推理代码已打包成 PAInpaint_inference.tar.gz" 