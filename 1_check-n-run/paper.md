# Check-n-run
[PAPER OFFICIAL](https://arxiv.org/pdf/2010.08679v2.pdf)

**Tags**: - Distributed Learning, Checkpoint, Embedding

**Highlights & Notes**

What?
- Checkpoint embeddings tables in 
  - cheaper way, wrt. storage and network cost, given that accuracy degradatio can't be more than 0.01%.
  - faster way, with minimal overhead on training. it means stalling the training for minimal duration.
  - minimizing the re-training time, it means the gap between failure time and the most recent checkpoint timestamp
- Use Model Parallelization for Embedding tables
  - as embedding tables are big, that it can't be stored in 1 gpu
  - so it's divided and stored in multiple gpu's for training (FB used 128 GPUS)

Why?
- Recommendation models are very big in size (for Facebook it's in TBs)
- most of this size belongs to embedding tables of sparse features.
- Recommendation models training data is also very big (for Facebook it's in Billions). 
- In this scenario, Distributed learning is required for Training, which often led to failures while training the model
- failures can be due to network, hardware, system(OOM), code and power outages.
- After the failure Training model from start is very costly here, That's why Checkpointing is crucial.
- But Checkpointing here is costly and challenging due to large embedding tables

How?
- `Incremental Checkpointing`
  - Embeddings of only small fraction are updated in a epoch. (here epoch means a fraction of dataset.)
  - with this insight, we don't need to checkpoint whole model in a epoch
  - we just need to save the embeddings corresponding to the categorical variable unique values used in training
  - we flag the categorical variable unique values in forward propagation, in GPU storing the embedding table

- Quantization
  - Quantization reduces the size of embedding tables but it impacts accuracy. 
  - accuracy degradation due to quantization is inversely proportional to wrt no of bits used for quantization.
  - tried different quantization: 
    - `uniform quantization` (symmetric, asymmetric).uniform means if quantized values are evenly spaced
    - `non-uniform quantization` tried KNN based method but it's costly and slow, so dropped it
    - `adaptive assymetric quantization` (best method)

- Decoupling
  - to minimize the run time overhead and
training stalls, model is shifted from gpu to host cpu and then training is unstalled. training was stalled for 7 seconds (between which the model was shifted from gput to host cpu)

- Avoiding trainer-reader state gap
  - When a training job resumes from a checkpoint, the run should still train the same training dataset as the original run.
  - Hence, the checkpoint must also include the reader state (i.e., which parts have been read).
  -  This is important, for example, to avoid training the same sample twice.
