# Lost in the Source Language

This is the repository for the paper "Lost in the Source Language: How Large Language Models Evaluate the Quality of Machine Translation" accepted by ACL 2024 findings. We provide our source code, data and results for easy reimplementation.

## Requirements
- Python >= 3.8.0
- Pytorch >= 2.1.2
- langchain >= 0.1.0
- langchain-core >= 0.1.9
- pandas
- openai
- *Optional* vllm

We recommend to use [vllm](https://github.com/vllm-project/vllm) to accelerate the inference.

## Coarse-grained Score Prediction([GEMBA](https://arxiv.org/pdf/2302.14520))
We use [GEMBA](https://github.com/MicrosoftTranslator/GEMBA)'s source code to predict scores. The results of our experiments are in the [gemba_results](gemba_results) folder. We compute the correlations between the metrics scores and human scores using [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval).

## Fine-grained Error Detection([AutoMQM](https://arxiv.org/pdf/2308.07286))
We implement AutoMQM for fine-grained error detection. For example, if you want to use GPT-3.5 in the S-R-T mode, simply run automqm.py as follows:
``` bash
python automqm.py --model-name gpt-3.5-turbo-0613 --lang-pair en-de --prefix gpt3.5-turbo_ref_stratified_wmt22_ende_3200 --example-selector stratified --has-source --has-reference --prompt-path prompts/prompt_ref_sample.json
```

To evaluate the output of AutoMQM, use the evaluate.py with the corresponding subcommand like sf1_mf1, mcc, etc., or just use test_all subcommand. If you want to convert the results to MQM scores, use the save_scores subcommand in evaluate.py.

## Fine-tune Llama2
The processed training data is in the data folder, which is derived from wmt21 MQM data. The output format is similar to that of [InstructScore](https://arxiv.org/pdf/2305.14282). To fine-tune Llama2 model, simply run the finetune_llama2.sh. Don't forget to configure some of the parameters like $MODEL_PATH_OR_NAME.

After training, use the inference.py to generate the answers of the testset.

Finally, use postprocess_inference.py to compute the MQM scores of the answers.