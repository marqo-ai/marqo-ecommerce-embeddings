git clone https://github.com/marqo-ai/GCL
cd ./GCL
MODEL=hf-hub:Marqo/marqo-ecommerce-B
outdir=/MarqoModels/GE/marqo-ecommerce-B/ap-title2image
hfdataset=Marqo/amazon-products-eval
python  evals/eval_hf_datasets_v1.py \
      --model_name $MODEL \
      --hf-dataset $hfdataset \
      --output-dir $outdir \
      --batch-size 1024 \
      --num_workers 8 \
      --left-key "['title']" \
      --right-key "['image']" \
      --img-or-txt "[['txt'], ['img']]" \
      --left-weight "[1]" \
      --right-weight "[1]" \
      --run-queries-cpu \
      --top-q 4000 \
      --doc-id-key item_ID \
      --context-length "[[64], [0]]"