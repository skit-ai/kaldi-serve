config = {
    "hi": {
        "tdnn": {
            "script": "./scripts/inference_tdnn_online.sh",
            "args": [
                "models/hindi/exp/chain/tdnn1g_sp_online/conf/online.conf",
                "models/hindi/exp/chain/tdnn1g_sp_online/final.mdl",
                "models/hindi/exp/chain/tree_a_sp/graph/HCLG.fst",
                "models/hindi/exp/chain/tree_a_sp/graph/words.txt"
            ]
        },
        "gmm":{
            "script": "./scripts/inference_gmm_online.sh",
            "args": [
                "models/hindi/exp/tri3b_online/conf/online_decoding.conf",
                "models/hindi/exp/tri3b/graph/HCLG.fst",
                "models/hindi/exp/tri3b/graph/words.txt"
            ]
        }
    },
    "en": {
        
    }
}
