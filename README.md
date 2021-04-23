# Analysis of BERT performance on Question Answering systems 

# Team Members: Paras Tehria, Tushar Gupta

**\*\*\*\*\* April 23 2021 \*\*\*\*\***

This github repo was created as a part of final project in class COMSE6998 Practical Deep Learning Systems Performance taught by professor Parijat Dube at Columbia University.
The main aim of this project was to compare the performance of efficient BERT models with baseline using different paradigms. 
We have taken 3 BERT Models (BERT Mini, BERT Medium, BERT Large)and compared there performance on question answering tasks.

Most of the code is taken from google research's BERT repo.
 https://github.com/google-research/bert

You can download the BERT models from here:

[**4/256 (BERT-Mini)**][4_256]

[**8/512 (BERT-Medium)**][8_512]

[**12/768 (BERT-Base)**][12_768]

[4_256]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip                   
[8_512]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip     
[12_768]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip  


#Download the SQUAD train and dev dataset

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json


(You will need to unzip the bert model)

Commands to run for training:
```buildoutcfg
python run_squad.py \
  --vocab_file=$uncased_L-24_H-1024_A-16/vocab.txt \
  --bert_config_file=$uncased_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=$uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --do_train=True \
  --train_file=train-v2.0.json \
  --do_predict=True \
  --predict_file=dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=15.0 \
```

Learning Rate Paradigms:
We have used two learning rate paradigms for this task: Constant LR and polynomial decay LR. Add --use_lr_decay = False to use constant LR. By default we'll be using polynomial decay LR.

BERT Embeddings:
We have used two embedding schemes for this task: taking embeddings of last hidden state and taking average of embeddings of all BERT hidden states. Add --use_modified_embed = True to use average embeddings. By default we'll be using last hidden layer's embeddings.

Commands to run fto generate prediction file:
```buildoutcfg 
   python run_squad.py \
     --vocab_file=uncased_L-24_H-1024_A-16/vocab.txt \
     --bert_config_file=uncased_L-24_H-1024_A-16/bert_config.json \
     --init_checkpoint=model.ckpt-10859 \
     --do_train=False \
     --max_query_length=30  \
     --do_predict=True \
     --predict_file=input_file.json \
     --predict_batch_size=8 \
     --n_best_size=3 \
     --max_seq_length=384 \
     --doc_stride=128 \
     --output_dir=output/    
```

This command will generate a prediction file named prediction.json.

To get accuracy score on test set. Run the command:
```buildoutcfg
   python evaluate.py data.json prediction.json
```

This will give you EM and F1 score on the dev set.

We highly recommend everyone to refer to this blog for understanding how to run this code:

https://www.pragnakalp.com/case-study/question-answering-system-in-python-using-bert-nlp/
