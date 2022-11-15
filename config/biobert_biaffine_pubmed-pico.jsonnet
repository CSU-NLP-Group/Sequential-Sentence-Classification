local transformer_model = "monologg/biobert_v1.1_pubmed";
local base_dir = "./datasets/pubmed-pico/";
local is_pubmed_rct = false; # 使用的是否是pubmed_rct的数据集
local debug = false; #the swith to turn on the debug mode


#是否采用warm_up，如果是，对num_epochs、patience和学习率都有影响
local dropout = 0.3 ;
local hidden_size = 512;
local ga_steps =  2 ;
local biaf_input_size = 128 ;

{
    "dataset_reader":{
         "type": "biaffine_reader",
         "is_pubmed": is_pubmed_rct,
         "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
                "namespace":"tokens",
            }
         },
         "tokenizer": {
              "type": "pretrained_transformer",
              "model_name": transformer_model,
              "max_length": 60,
              "tokenizer_kwargs":{
                  "do_lower_case": true,
              }
         },
    },
    "model": {
        "type": "biaffine_classifier",
        "biaf_input_size": biaf_input_size,
        "dropout": dropout,

        "sent_encoder":{
            "type":"my_att",
            "input_size":768,
            "num_query":1,
            "attention":{
                "type":"dot_product",
            },
        },

        "ctx_sent_encoders": [
            {
                "type": "lstm",
                "bidirectional": true,
                "hidden_size": hidden_size,
                "input_size": 768,
                "num_layers": 1
            },
        ],

        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": transformer_model,
                }
            }
        },

    },
    "train_data_path": base_dir + (if debug then "small_train.txt" else "train.txt"),
    "validation_data_path": base_dir  + (if debug then "small_dev.txt" else "dev.txt"),
    "test_data_path": base_dir + (if debug then "small_test.txt" else "test.txt"),
    "evaluate_on_test": true,

    "trainer": {
        "cuda_device": 1,
        "grad_norm": 1,

        "num_epochs":  40,
        "patience": 10,
        "num_gradient_accumulation_steps": ga_steps,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr":  5e-06,
            'eps':1e-06,
            "weight_decay": 0.01,
            "parameter_groups": [
                [["transformer_model\\.*\\.bias",
                  "LayerNorm\\.*\\.weight",
                  "layer_norm\\.*\\.weight"], {"weight_decay": 0}],
                [["ctx_sent_encoders", "biaffine_scorer"], {"lr":3e-4}]
            ],
        },

        "callbacks":[
            {
                "type":"FGM",
                "emb_name":"word_embeddings",
            },
        ],

        "validation_metric": "+span-f1",
    },
    "data_loader": {
        "batches_per_epoch": 2000,
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 2,
            "sorting_keys": [
                "tags"
            ]
        },
    },
    "validation_data_loader":{
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8,
            "sorting_keys": [
                "tags"
            ]
        },
    }
}
